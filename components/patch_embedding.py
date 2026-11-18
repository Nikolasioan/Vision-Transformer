import torch
import torch.nn as nn

def _pair(v):
    # turn 32 into (32, 32); leave (32, 48) as-is
    return v if isinstance(v, (tuple, list)) else (v, v)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(32, 48), patch_size=4, in_chans=3, dim=192):
        super().__init__()
        H, W = _pair(img_size)
        Ph, Pw = _pair(patch_size)
        assert H % Ph == 0 and W % Pw == 0 # Image size must be divisible by patch size.


        self.grid = (H // Ph, W // Pw)                 # (n_patches_H, n_patches_W)
        self.n_patches = self.grid[0] * self.grid[1]
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=(Ph, Pw), stride=(Ph, Pw))

    def forward(self, x):                     # x: (B, C, H, W)
        x = self.proj(x)                      # (B, dim, H/Ph, W/Pw)
        x = x.flatten(2)                      # (B, dim, n_patches)
        x = x.transpose(1, 2)                 # (B, n_patches, dim)
        return x