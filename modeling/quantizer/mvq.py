import torch
from typing import List, Tuple
from torch.nn import functional as F
from torch import distributed as tdist, nn as nn

from .quantizer import VectorQuantizer

def get_entropy_loss(latent_embed, codebook_embed, inv_entropy_tau):
    E_dist = latent_embed.square().sum(dim=1, keepdim=True) + codebook_embed.square().sum(dim=1, keepdim=False)
    E_dist.addmm_(latent_embed, codebook_embed.T, alpha=-2, beta=1)  # E_dist: (N, vocab_size)
    logits = -E_dist.float().mul_(inv_entropy_tau)
    # calc per_sample_entropy
    prob, log_prob = logits.softmax(dim=-1), logits.log_softmax(dim=-1)  # both are (N, vocab_size)
    per_sample_entropy = torch.mean((-prob * log_prob).sum(dim=-1))
    # calc codebook_entropy
    avg_prob = prob.mean(dim=0)  # (vocab_size,)
    log_avg_prob = torch.log(avg_prob + 1e-7)
    codebook_entropy = (-avg_prob * log_avg_prob).sum()
    # calc entropy_loss
    entropy_loss = per_sample_entropy - codebook_entropy
    return entropy_loss


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # self.norm_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, idx):
        return F.embedding(
            idx, F.normalize(self.weight, dim=1), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

    def get_norm_weight(self):
        return F.normalize(self.weight, dim=1)


class ResConv(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3 if quant_resi < 0 else 1
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class VectorQuantizerMVQ(nn.Module):
    def __init__(
        self,
        codebook_size,
        token_size,
        commitment_cost=0.25,
        use_l2_norm=False,
        # entropy_temp=0.01, # we do not use this
        clustering_vq=False,
        num_codebooks=16
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList()
        for _ in range(num_codebooks):
            codebook = VectorQuantizer(
                codebook_size=codebook_size // num_codebooks,
                token_size=token_size // num_codebooks,
                commitment_cost=commitment_cost,
                use_l2_norm=use_l2_norm,
                clustering_vq=clustering_vq,
            )
            self.codebooks.append(codebook)

    def init_vocab(self, eini: float):
        for codebook in self.codebooks:
            codebook.init_vocab(eini)

    def f_to_idx(self, features):
        indices = []
        chunk_size = features.shape[-1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=-1)
        for i, codebook in enumerate(self.codebooks):
            indices.append(codebook.f_to_idx(splited_features[i]))
        indices = torch.stack(indices, dim=1)
        return indices

    def idx_to_f(self, indices):
        assert indices.shape[1] == self.num_codebooks
        latent_features = []
        for i, codebook in enumerate(self.codebooks):
            sub_indices = indices[:, i].flatten(start_dim=1)
            latent_feature = codebook.codebook(sub_indices)
            latent_features.append(latent_feature)
        latent_features = torch.cat(latent_features, dim=-1)
        return latent_features

    def get_codebook_entry(self, indices):
        """Get codebook entries for multi-codebook indices.
        
        Args:
            indices: Tensor of shape (N, num_codebooks) or (N, num_codebooks, H, W)
        
        Returns:
            z_quantized: Quantized features
        """
        if len(indices.shape) == 2:
            # indices shape: (N, num_codebooks)
            latent_features = []
            for i, codebook in enumerate(self.codebooks):
                sub_indices = indices[:, i]
                latent_feature = codebook.get_codebook_entry(sub_indices)
                latent_features.append(latent_feature)
            return torch.cat(latent_features, dim=-1)
        elif len(indices.shape) == 4:
            # indices shape: (B, num_codebooks, H, W)
            batch_size, _, height, width = indices.shape
            latent_features = []
            for i, codebook in enumerate(self.codebooks):
                sub_indices = indices[:, i]  # (B, H, W)
                latent_feature = codebook.get_codebook_entry(sub_indices.flatten())
                # Reshape to (B, H, W, token_size // num_codebooks)
                latent_feature = latent_feature.view(batch_size, height, width, -1)
                latent_features.append(latent_feature)
            # Concatenate along the last dimension and rearrange to (B, C, H, W)
            latent_features = torch.cat(latent_features, dim=-1)  # (B, H, W, C)
            return latent_features.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        else:
            raise NotImplementedError(f"Unsupported indices shape: {indices.shape}")

    def forward(self, features):
        latent_features = []
        all_result_dicts = []
        chunk_size = features.shape[1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=1)

        for i, codebook in enumerate(self.codebooks):
            latent_feature, result_dict = codebook(splited_features[i].float())
            latent_features.append(latent_feature.to(features.dtype))
            all_result_dicts.append(result_dict)
        
        # Concatenate latent features
        z_quantized = torch.cat(latent_features, dim=1)  # Concatenate along channel dimension
        
        # Calculate global losses
        global_quantizer_loss = sum(rd['quantizer_loss'] for rd in all_result_dicts) / self.num_codebooks
        global_commitment_loss = sum(rd['commitment_loss'] for rd in all_result_dicts) / self.num_codebooks
        global_codebook_loss = sum(rd['codebook_loss'] for rd in all_result_dicts) / self.num_codebooks
        
        # Collect all min_encoding_indices
        # Each codebook returns indices of shape (B, H, W)
        # Stack them to get shape (B, num_codebooks, H, W)
        all_indices = torch.stack([rd['min_encoding_indices'] for rd in all_result_dicts], dim=1)
        
        result_dict = dict(
            quantizer_loss=global_quantizer_loss,
            commitment_loss=global_commitment_loss,
            codebook_loss=global_codebook_loss,
            min_encoding_indices=all_indices
        )
        
        return z_quantized, result_dict