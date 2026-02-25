import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Mapping, Text, Tuple
from einops import rearrange
from torch.cuda.amp import autocast


class SoftVectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 clustering_vq: bool = False,
                 entropy_loss_ratio: float = 0.01,
                 tau: float = 0.07,
                 num_codebooks: int = 1,
                 show_usage: bool = False
                 ):
        super().__init__()
        # Map new parameter names to internal names for compatibility
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        self.clustering_vq = clustering_vq
        
        # Keep soft quantization specific parameters
        self.num_codebooks = num_codebooks
        self.n_e = codebook_size
        self.e_dim = token_size
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = use_l2_norm
        self.show_usage = show_usage
        self.tau = tau
        
        # Single embedding layer for all codebooks
        self.embedding = nn.Parameter(torch.randn(num_codebooks, codebook_size, token_size))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(num_codebooks, 65536))

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        original_shape = z.shape
        
        # Handle input reshaping to match VectorQuantizer format
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z = z.view(z.size(0), -1, z.size(-1))
        
        batch_size, seq_length, _ = z.shape
        
        # Ensure sequence length is divisible by number of codebooks
        assert seq_length % self.num_codebooks == 0, \
            f"Sequence length ({seq_length}) must be divisible by number of codebooks ({self.num_codebooks})"
        
        segment_length = seq_length // self.num_codebooks
        z_segments = z.view(batch_size, self.num_codebooks, segment_length, self.e_dim)
        
        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)
            
        z_flat = z_segments.permute(1, 0, 2, 3).contiguous().view(self.num_codebooks, -1, self.e_dim)
        
        logits = torch.einsum('nbe, nke -> nbk', z_flat, embedding.detach())
        
        # Calculate probabilities (soft quantization)
        probs = F.softmax(logits / self.tau, dim=-1)  
        
        # Soft quantize
        z_q = torch.einsum('nbk, nke -> nbe', probs, embedding)
        
        # Reshape back
        z_q = z_q.view(self.num_codebooks, batch_size, segment_length, self.e_dim).permute(1, 0, 2, 3).contiguous()
        
        # Calculate cosine similarity
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_segments.view(-1, self.e_dim),
                z_q.view(-1, self.e_dim),
                dim=-1
            ).mean()
        
        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (num_codebooks, batch_size * segment_length)
        indices = indices.transpose(0, 1).contiguous()  # (batch_size * segment_length, num_codebooks)
        
        # Track codebook usage
        if self.show_usage and self.training:
            for k in range(self.num_codebooks):
                cur_len = indices.size(0)
                self.codebook_used[k, :-cur_len].copy_(self.codebook_used[k, cur_len:].clone())
                self.codebook_used[k, -cur_len:].copy_(indices[:, k])
        
        # Calculate losses if training
        if self.training:
            # Soft quantization doesn't have traditional commitment/codebook loss
            # Map entropy loss to quantizer_loss for compatibility
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(logits.view(-1, self.n_e))
            quantizer_loss = entropy_loss
            commitment_loss = torch.tensor(0.0, device=z.device)
            codebook_loss = torch.tensor(0.0, device=z.device)
        else:
            quantizer_loss = torch.tensor(0.0, device=z.device)
            commitment_loss = torch.tensor(0.0, device=z.device)
            codebook_loss = torch.tensor(0.0, device=z.device)
        
        # Calculate codebook usage
        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        z_q = z_q.view(batch_size, -1, self.e_dim)
        
        # Reshape back to original input shape to match VectorQuantizer
        z_q = z_q.view(batch_size, original_shape[2], original_shape[3], original_shape[1])
        z_quantized = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        # Calculate average probabilities
        avg_probs = torch.mean(torch.mean(probs, dim=-1))
        max_probs = torch.mean(torch.max(probs, dim=-1)[0])
        
        # Return format matching VectorQuantizer
        result_dict = dict(
            quantizer_loss=quantizer_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=indices.view(batch_size, self.num_codebooks, segment_length).view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )
        
        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        """Added for compatibility with VectorQuantizer API"""
        if len(indices.shape) == 1:
            # For single codebook case
            z_quantized = self.embedding[0][indices]
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding[0])
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss