"""Multi-codebook Vector Quantizer (MVQ) for VibeToken.

Uses multiple independent codebooks for richer discrete representations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from .vector_quantizer import VectorQuantizer


class VectorQuantizerMVQ(nn.Module):
    """Multi-codebook Vector Quantizer.
    
    Splits the latent representation into multiple parts, each quantized
    by an independent codebook. This allows for more expressive discrete
    representations.
    """
    
    def __init__(
        self,
        codebook_size: int,
        token_size: int,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
        num_codebooks: int = 8,
    ):
        """Initialize MVQ.
        
        Args:
            codebook_size: Total codebook size (divided among codebooks).
            token_size: Total token dimension (divided among codebooks).
            commitment_cost: Weight for commitment loss.
            use_l2_norm: Whether to L2-normalize embeddings.
            num_codebooks: Number of independent codebooks.
        """
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList()
        
        for _ in range(num_codebooks):
            codebook = VectorQuantizer(
                codebook_size=codebook_size // num_codebooks,
                token_size=token_size // num_codebooks,
                commitment_cost=commitment_cost,
                use_l2_norm=use_l2_norm,
            )
            self.codebooks.append(codebook)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize features using multiple codebooks.
        
        Args:
            features: Input features of shape (B, C, H, W).
            
        Returns:
            z_quantized: Quantized features of shape (B, C, H, W).
            result_dict: Dictionary with losses and indices.
        """
        latent_features = []
        all_result_dicts = []
        chunk_size = features.shape[1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=1)

        for i, codebook in enumerate(self.codebooks):
            latent_feature, result_dict = codebook(splited_features[i].float())
            latent_features.append(latent_feature.to(features.dtype))
            all_result_dicts.append(result_dict)
        
        # Concatenate quantized features
        z_quantized = torch.cat(latent_features, dim=1)
        
        # Aggregate losses
        global_quantizer_loss = sum(rd['quantizer_loss'] for rd in all_result_dicts) / self.num_codebooks
        global_commitment_loss = sum(rd['commitment_loss'] for rd in all_result_dicts) / self.num_codebooks
        global_codebook_loss = sum(rd['codebook_loss'] for rd in all_result_dicts) / self.num_codebooks
        
        # Stack indices: shape (B, num_codebooks, H, W)
        all_indices = torch.stack([rd['min_encoding_indices'] for rd in all_result_dicts], dim=1)
        
        result_dict = dict(
            quantizer_loss=global_quantizer_loss,
            commitment_loss=global_commitment_loss,
            codebook_loss=global_codebook_loss,
            min_encoding_indices=all_indices
        )
        
        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook entries for multi-codebook indices.
        
        Args:
            indices: Tensor of shape:
                - (B, num_codebooks): single token per codebook
                - (B, num_codebooks, seq_len): sequence of tokens per codebook
                - (B, num_codebooks, H, W): 2D spatial tokens per codebook
                - (B, seq_len, 1): generator format (single codebook index per position)
            
        Returns:
            z_quantized: Quantized features.
        """
        if len(indices.shape) == 2:
            # Shape: (B, num_codebooks) - each entry is a token index
            latent_features = []
            for i, codebook in enumerate(self.codebooks):
                sub_indices = indices[:, i]
                latent_feature = codebook.get_codebook_entry(sub_indices)
                latent_features.append(latent_feature)
            return torch.cat(latent_features, dim=-1)
            
        elif len(indices.shape) == 3:
            batch_size, dim1, dim2 = indices.shape
            
            # Check if this is (B, num_codebooks, seq_len) or (B, seq_len, 1)
            if dim1 == self.num_codebooks:
                # Shape: (B, num_codebooks, seq_len) - from encode()
                seq_len = dim2
                latent_features = []
                for i, codebook in enumerate(self.codebooks):
                    sub_indices = indices[:, i, :]  # (B, seq_len)
                    latent_feature = codebook.get_codebook_entry(sub_indices.flatten())
                    latent_feature = latent_feature.view(batch_size, seq_len, -1)
                    latent_features.append(latent_feature)
                
                # Concatenate along feature dimension: (B, seq_len, C)
                z_quantized = torch.cat(latent_features, dim=-1)
                # Reshape to (B, C, 1, seq_len) for decoder
                z_quantized = z_quantized.permute(0, 2, 1).unsqueeze(2)
                return z_quantized
            
            elif dim2 == 1:
                # Shape: (B, seq_len, 1) - common format from generator
                indices = indices.squeeze(-1)  # (B, seq_len)
                seq_len = dim1
                
                # For generator format, all codebooks use the same indices
                latent_features = []
                for i, codebook in enumerate(self.codebooks):
                    latent_feature = codebook.get_codebook_entry(indices.flatten())
                    latent_feature = latent_feature.view(batch_size, seq_len, -1)
                    latent_features.append(latent_feature)
                
                z_quantized = torch.cat(latent_features, dim=-1)  # (B, seq_len, C)
                z_quantized = z_quantized.permute(0, 2, 1).unsqueeze(2)
                return z_quantized
            else:
                raise ValueError(f"Ambiguous 3D indices shape: {indices.shape}. "
                               f"Expected (B, {self.num_codebooks}, seq_len) or (B, seq_len, 1)")
            
        elif len(indices.shape) == 4:
            # Shape: (B, num_codebooks, H, W)
            batch_size, _, height, width = indices.shape
            latent_features = []
            for i, codebook in enumerate(self.codebooks):
                sub_indices = indices[:, i]  # (B, H, W)
                latent_feature = codebook.get_codebook_entry(sub_indices.flatten())
                latent_feature = latent_feature.view(batch_size, height, width, -1)
                latent_features.append(latent_feature)
            
            # Concatenate and permute to (B, C, H, W)
            latent_features = torch.cat(latent_features, dim=-1)
            return latent_features.permute(0, 3, 1, 2).contiguous()
        else:
            raise NotImplementedError(f"Unsupported indices shape: {indices.shape}")

    def f_to_idx(self, features: torch.Tensor) -> torch.Tensor:
        """Convert features directly to indices without quantization.
        
        Args:
            features: Input features.
            
        Returns:
            indices: Token indices for each codebook.
        """
        indices = []
        chunk_size = features.shape[-1] // self.num_codebooks
        splited_features = features.split(chunk_size, dim=-1)
        for i, codebook in enumerate(self.codebooks):
            indices.append(codebook.f_to_idx(splited_features[i]))
        indices = torch.stack(indices, dim=1)
        return indices
