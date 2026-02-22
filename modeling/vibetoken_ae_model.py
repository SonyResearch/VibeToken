"""VibeTokenAE model definition.

Replaces vector quantization with spherify projection and noise-based training.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange
import json
from pathlib import Path
from omegaconf import OmegaConf

from huggingface_hub import PyTorchModelHubMixin

from modeling.modules.base_model import BaseModel
from modeling.modules.encoder_decoder import ResolutionEncoder, ResolutionDecoder


def spherify(z, eps=1e-8):
    """Project latents onto a sphere of radius sqrt(L) using RMS normalization.

    Args:
        z: Tensor of shape (B, N, D).

    Returns:
        Tensor of same shape with ||z||_2 = sqrt(N*D).
    """
    rms = torch.sqrt(torch.mean(z ** 2, dim=(1, 2), keepdim=True) + eps)
    return z / rms


def sample_noise_angle(batch_size, max_angle, high_angle_min, high_angle_max,
                        high_angle_prob, device):
    """Sample noise angles per the VibeTokenAE noise strategy.

    With probability (1 - high_angle_prob): uniform in [0, max_angle].
    With probability high_angle_prob: uniform in [high_angle_min, high_angle_max].

    Returns:
        alpha: Tensor of shape (B,) in degrees.
    """
    use_high = torch.rand(batch_size, device=device) < high_angle_prob
    alpha_normal = torch.rand(batch_size, device=device) * max_angle
    alpha_high = (
        high_angle_min
        + torch.rand(batch_size, device=device) * (high_angle_max - high_angle_min)
    )
    alpha = torch.where(use_high, alpha_high, alpha_normal)
    return alpha


class VibeTokenAEModel(BaseModel, PyTorchModelHubMixin,
                        tags=["image-tokenization", "vibetoken-ae"]):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self.encoder = ResolutionEncoder(config)
        self.decoder = ResolutionDecoder(config)

        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))

        self.apply(self._init_weights)

        ae_cfg = config.model.vq_model.get("ae", {})
        self.max_angle = ae_cfg.get("max_angle", 85.0)
        self.high_angle_min = ae_cfg.get("high_angle_min", 85.0)
        self.high_angle_max = ae_cfg.get("high_angle_max", 89.0)
        self.high_angle_prob = ae_cfg.get("high_angle_prob", 0.1)

    def _save_pretrained(self, save_directory: Path) -> None:
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _to_3d(self, z_4d):
        """(B, D, 1, N) -> (B, N, D)"""
        return rearrange(z_4d, 'b d 1 n -> b n d')

    def _to_4d(self, z_3d):
        """(B, N, D) -> (B, D, 1, N)"""
        return rearrange(z_3d, 'b n d -> b d 1 n')

    def _encode_raw(self, x, attention_mask=None, encode_patch_size=None, train=True):
        """Run the encoder and return the raw 4D output (B, D, 1, N)."""
        z = self.encoder(
            pixel_values=x,
            latent_tokens=self.latent_tokens,
            attention_mask=attention_mask,
            encode_patch_size=encode_patch_size,
            train=train,
        )
        return z

    def encode(self, x, attention_mask=None, encode_patch_size=None, train=True, **kwargs):
        """Encode and spherify. Returns (z_spherified_4d, result_dict)."""
        z_4d = self._encode_raw(x, attention_mask=attention_mask,
                                encode_patch_size=encode_patch_size, train=train)
        v = spherify(self._to_3d(z_4d))
        v_4d = self._to_4d(v)
        return v_4d, {}

    def decode(self, z, attention_mask=None, height=None, width=None,
               decode_patch_size=None, train=True):
        return self.decoder(
            z, attention_mask=attention_mask,
            height=height, width=width,
            decode_patch_size=decode_patch_size, train=train,
        )

    def forward(self, x, key_attention_mask=None, height=None, width=None, train=True):
        if height is None:
            batch_size, _, height, width = x.shape

        # 1. Encode + spherify
        z_4d = self._encode_raw(x, attention_mask=key_attention_mask, train=train)
        v = spherify(self._to_3d(z_4d))  # (B, N, D)

        if not train:
            v_4d = self._to_4d(v).to(self.decoder.decoder_embed.weight.dtype)
            decoded = self.decode(v_4d, attention_mask=key_attention_mask,
                                  height=height, width=width, train=False)
            return decoded, {}

        # 2. Sample shared noise direction
        e = torch.randn_like(v)

        # 3. Sample angle and compute sigma
        alpha_deg = sample_noise_angle(
            v.shape[0], self.max_angle, self.high_angle_min,
            self.high_angle_max, self.high_angle_prob, v.device,
        )
        alpha_rad = alpha_deg * (math.pi / 180.0)
        sigma = torch.tan(alpha_rad)  # (B,)
        sigma = sigma[:, None, None]  # (B, 1, 1)

        # 4. Sub-noise factor
        s = torch.rand(v.shape[0], 1, 1, device=v.device) * 0.5
        sigma_sub = s * sigma

        # 5. Create noisy latents (same noise direction for both)
        v_noisy = spherify(v + sigma_sub * e)
        v_NOISY = spherify(v + sigma * e)

        # 6. Decode both
        dtype = self.decoder.decoder_embed.weight.dtype
        x_small = self.decode(
            self._to_4d(v_noisy).to(dtype),
            attention_mask=key_attention_mask,
            height=height, width=width, train=True,
        )
        x_large = self.decode(
            self._to_4d(v_NOISY).to(dtype),
            attention_mask=key_attention_mask,
            height=height, width=width, train=True,
        )

        # 7. Re-encode x_large for latent consistency
        z_reenc_4d = self._encode_raw(x_large, attention_mask=key_attention_mask, train=train)
        v2 = spherify(self._to_3d(z_reenc_4d))

        extra_dict = {
            "x_large": x_large,
            "v": v,
            "v2": v2,
        }
        return x_small, extra_dict

    @torch.no_grad()
    def generate(self, batch_size, height, width, num_steps=1,
                 refine_noise_deg=5.0):
        """Generate images from random noise with optional iterative refinement.

        Args:
            batch_size: Number of images to generate.
            height: Output image height.
            width: Output image width.
            num_steps: 1 for one-shot generation, >1 for iterative refinement.
            refine_noise_deg: Small noise angle (degrees) added during refinement steps.

        Returns:
            List of T image tensors [(B,3,H,W), ...], one per step.
        """
        was_training = self.training
        self.eval()
        dtype = self.decoder.decoder_embed.weight.dtype
        device = self.decoder.decoder_embed.weight.device

        # Step 0: sample random noise, spherify (no noise), decode
        e = torch.randn(batch_size, self.num_latent_tokens, self.token_size,
                        device=device)
        v = spherify(e)
        x = self.decode(self._to_4d(v).to(dtype), height=height, width=width,
                        train=False)
        step_images = [x.clamp(0, 1)]

        # Refinement steps 1..T-1
        sigma_refine = math.tan(refine_noise_deg * math.pi / 180.0)
        for _ in range(num_steps - 1):
            z_4d = self._encode_raw(x, train=False)
            z = self._to_3d(z_4d)
            noise = torch.randn_like(z)
            v = spherify(z + sigma_refine * noise)
            x = self.decode(self._to_4d(v).to(dtype), height=height,
                            width=width, train=False)
            step_images.append(x.clamp(0, 1))

        if was_training:
            self.train()
        return step_images
