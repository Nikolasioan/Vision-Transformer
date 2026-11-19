import torch
import torch.nn as nn

from patch_embedding import PatchEmbed  


class ViT(nn.Module):
    def __init__(self, img_size=(32, 48), patch_size=4, in_chans=3, dim=192):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) # Add of class embedding.
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                         # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)                   #     (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)    #     (B, 1, dim)
        x = torch.cat([cls, x], dim=1)            #     (B, N+1, dim)
        return x