import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

class FuzzyEmbedding(nn.Module):
    def __init__(self, grid_size, scale, width, apply_fuzzy=False):
        super(FuzzyEmbedding, self).__init__()
        assert grid_size == 1024, "grid_size must be 1024 for now"
        
        self.grid_size = grid_size
        self.scale = scale
        self.width = width
        self.apply_fuzzy = apply_fuzzy
        # grid_size is the minimum possible token size
        # then we can use grid_sample to get the fuzzy embedding for any resolution
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(grid_size, width))
        
        self.class_positional_embedding = nn.Parameter(
                scale * torch.randn(1, width))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, grid_height, grid_width, train=True, dtype=torch.float32):
        meshx, meshy = torch.meshgrid(
            torch.tensor(list(range(grid_height)), device=self.positional_embedding.device), 
            torch.tensor(list(range(grid_width)), device=self.positional_embedding.device)
        )
        meshx = meshx.to(dtype)
        meshy = meshy.to(dtype)

        # Normalize coordinates to [-1, 1] range
        meshx = 2 * (meshx / (grid_height - 1)) - 1
        meshy = 2 * (meshy / (grid_width - 1)) - 1
        
        if self.apply_fuzzy:
            # Add uniform noise in range [-0.0004, 0.0004] to x and y coordinates
            if train:
                noise_x = torch.rand_like(meshx) * 0.0008 - 0.0004
                noise_y = torch.rand_like(meshy) * 0.0008 - 0.0004
            else:
                noise_x = torch.zeros_like(meshx)
                noise_y = torch.zeros_like(meshy)

            # Apply noise to the mesh coordinates
            meshx = meshx + noise_x
            meshy = meshy + noise_y
        
        grid = torch.stack((meshy, meshx), 2).to(self.positional_embedding.device)
        grid = grid.unsqueeze(0) # add batch dim
        
        positional_embedding = einops.rearrange(self.positional_embedding, "(h w) d -> d h w", h=int(math.sqrt(self.grid_size)), w=int(math.sqrt(self.grid_size)))
        positional_embedding = positional_embedding.to(dtype)
        positional_embedding = positional_embedding.unsqueeze(0) # add batch dim

        fuzzy_embedding = F.grid_sample(positional_embedding, grid, align_corners=False)
        fuzzy_embedding = fuzzy_embedding.to(dtype)
        fuzzy_embedding = einops.rearrange(fuzzy_embedding, "b d h w -> b (h w) d").squeeze(0)

        final_embedding = torch.cat([self.class_positional_embedding, fuzzy_embedding], dim=0)
        return final_embedding


if __name__ == "__main__":
    fuzzy_embedding = FuzzyEmbedding(256, 1.0, 1024)
    grid_height = 16
    grid_width = 32
    fuzzy_embedding = fuzzy_embedding(grid_height, grid_width, dtype=torch.bfloat16)
    print(fuzzy_embedding.shape)