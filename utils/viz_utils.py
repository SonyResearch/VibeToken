"""Utils functions for visualization."""

import torch
import torchvision.transforms.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

def make_viz_from_samples(
    original_images,
    reconstructed_images
):
    """Generates visualization images from original images and reconstructed images.

    Args:
        original_images: A torch.Tensor, original images.
        reconstructed_images: A torch.Tensor, reconstructed images.

    Returns:
        A tuple containing two lists - images_for_saving and images_for_logging.
    """
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    reconstructed_images = reconstructed_images * 255.0
    reconstructed_images = reconstructed_images.cpu()
    
    original_images = torch.clamp(original_images, 0.0, 1.0)
    original_images *= 255.0
    original_images = original_images.cpu()

    diff_img = torch.abs(original_images - reconstructed_images)
    to_stack = [original_images, reconstructed_images, diff_img]

    images_for_logging = rearrange(
            torch.stack(to_stack),
            "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
            l1=1).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_from_samples_generation(
    generated_images,
):
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging


def make_viz_from_refinement_steps(step_images):
    """Create a grid visualization of iterative refinement steps.

    Each row is one sample, columns are refinement steps [step_0 | step_1 | ... | step_T-1].

    Args:
        step_images: List of T tensors, each (B, 3, H, W) in [0, 1].

    Returns:
        (images_for_saving, images_for_logging) matching the existing pattern.
        images_for_saving is a single PIL image with the full grid.
        images_for_logging is the tensor (C, grid_H, grid_W).
    """
    T = len(step_images)
    B = step_images[0].shape[0]

    # stack to (T, B, C, H, W), then rearrange to grid: rows=samples, cols=steps
    grid = torch.stack([img.clamp(0, 1).cpu() for img in step_images])  # (T, B, C, H, W)
    grid = (grid * 255.0).byte()
    # rows = B samples, cols = T steps
    images_for_logging = rearrange(grid, 't b c h w -> c (b h) (t w)')
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging


def make_viz_from_samples_t2i_generation(
    generated_images,
    captions,
):
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    # Create a new image with space for captions
    width, height = images_for_saving.size
    caption_height = 20 * len(captions) + 10
    new_height = height + caption_height
    new_image = Image.new("RGB", (width, new_height), "black")
    new_image.paste(images_for_saving, (0, 0))

    # Adding captions below the image
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()

    for i, caption in enumerate(captions):
        draw.text((10, height + 10 + i * 20), caption, fill="white", font=font)

    return new_image, images_for_logging