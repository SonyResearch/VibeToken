"""Training loss implementation.

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

from modeling.modules.blocks import SimpleMLPAdaLN
from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class ReconstructionLoss_Stage1(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        loss_config = config.losses
        self.quantizer_weight = loss_config.quantizer_weight
        self.target_codebook_size = 1024

    def forward(self,
                target_codes: torch.Tensor,
                reconstructions: torch.Tensor,
                quantizer_loss: torch.Tensor,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        return self._forward_generator(target_codes, reconstructions, quantizer_loss)

    def _forward_generator(self,
                           target_codes: torch.Tensor,
                           reconstructions: torch.Tensor,
                           quantizer_loss: Mapping[Text, torch.Tensor],
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        reconstructions = reconstructions.contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        batch_size = reconstructions.shape[0]
        reconstruction_loss = loss_fct(reconstructions.view(batch_size, self.target_codebook_size, -1),
                                        target_codes.view(batch_size, -1))
        total_loss = reconstruction_loss + \
            self.quantizer_weight * quantizer_loss["quantizer_loss"]

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss["quantizer_loss"]).detach(),
            commitment_loss=quantizer_loss["commitment_loss"].detach(),
            codebook_loss=quantizer_loss["codebook_loss"].detach(),
        )

        return total_loss, loss_dict


class ReconstructionLoss_Stage2(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


class ReconstructionLoss_Single_Stage(ReconstructionLoss_Stage2):
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        loss_config = config.losses
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        
        if self.quantize_mode == "vae":
            self.kl_weight = loss_config.get("kl_weight", 1e-6)
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        if self.quantize_mode in ["vq", "mvq", "softvq"]:
            # Compute quantizer loss.
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
        elif self.quantize_mode == "vae":
            # Compute kl loss.
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            posteriors = extra_result_dict
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                kl_loss=(self.kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
        else:
            raise NotImplementedError

        return total_loss, loss_dict


class ReconstructionLoss_AE(torch.nn.Module):
    """Loss module for VibeTokenAE: pixel recon + pixel consistency + latent consistency + GAN."""

    def __init__(self, config):
        super().__init__()
        loss_config = config.losses

        self.discriminator = NLayerDiscriminator()
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()

        self.discriminator_iter_start = loss_config.discriminator_start
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight

        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.recon_l1_weight = loss_config.get("recon_l1_weight", 25.0)
        self.recon_vgg_weight = loss_config.get("recon_vgg_weight", 1.0)
        self.pixcon_l1_weight = loss_config.get("pixcon_l1_weight", 10.0)
        self.pixcon_vgg_weight = loss_config.get("pixcon_vgg_weight", 1.0)
        self.latcon_weight = loss_config.get("latcon_weight", 0.1)

        self.config = config

    def should_discriminator_be_trained(self, global_step: int):
        return global_step >= self.discriminator_iter_start

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        x = inputs.contiguous()
        x_small = reconstructions.contiguous()

        x_large = extra_result_dict.get("x_large")
        v = extra_result_dict.get("v")
        v2 = extra_result_dict.get("v2")

        # (A) Pixel reconstruction loss: SmoothL1 + perceptual on x_small vs original
        recon_l1 = F.smooth_l1_loss(x_small, x, reduction="mean")
        recon_percep = self.perceptual_loss(x_small, x).mean()
        l_recon = self.recon_l1_weight * recon_l1 + self.recon_vgg_weight * recon_percep

        # (B) Pixel consistency loss: SmoothL1 + perceptual on x_large vs sg(x_small)
        l_pixcon = torch.zeros((), device=x.device)
        pixcon_l1_val = torch.zeros((), device=x.device)
        pixcon_percep_val = torch.zeros((), device=x.device)
        if x_large is not None:
            x_small_sg = x_small.detach()
            x_large_f = x_large.float().contiguous()
            pixcon_l1_val = F.smooth_l1_loss(x_large_f, x_small_sg, reduction="mean")
            pixcon_percep_val = self.perceptual_loss(x_large_f, x_small_sg).mean()
            l_pixcon = self.pixcon_l1_weight * pixcon_l1_val + self.pixcon_vgg_weight * pixcon_percep_val

        # (C) Latent consistency loss: 1 - cosine(v, v2)
        l_latcon = torch.zeros((), device=x.device)
        if v is not None and v2 is not None:
            cos_sim = F.cosine_similarity(v.flatten(1), v2.flatten(1), dim=1)
            l_latcon = self.latcon_weight * (1.0 - cos_sim).mean()

        # (D) Adversarial loss (on x_small, the main reconstruction)
        generator_loss = torch.zeros((), device=x.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(x_small)
            generator_loss = -torch.mean(logits_fake)
        d_weight *= self.discriminator_weight

        total_loss = (
            l_recon
            + l_pixcon
            + l_latcon
            + d_weight * discriminator_factor * generator_loss
        )

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=l_recon.detach(),
            recon_l1=recon_l1.detach(),
            recon_percep=recon_percep.detach(),
            l_recon=l_recon.detach(),
            pixcon_l1=pixcon_l1_val.detach(),
            pixcon_percep=pixcon_percep_val.detach(),
            l_pixcon=l_pixcon.detach(),
            l_latcon=l_latcon.detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )
        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake)

        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = (
                self.ema_real_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            )
            self.ema_fake_logits_mean = (
                self.ema_fake_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
            )

        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict