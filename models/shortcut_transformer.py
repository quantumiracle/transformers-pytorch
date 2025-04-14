"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Shortcut Models: https://github.com/kvfrans/shortcut-models
    - Shortcut Pytorch: https://github.com/smileyenot983/shortcut_pytorch

For shortcut model, it has additional dt_embedder.
"""

import torch
from torch import nn
from models.utils.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from models.utils.blocks import (
    PatchEmbed, 
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint

from models.diffusion_transformer import DiTBlock

class ShortcutTransformer(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.dt_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        # number of denoising steps can be applied
        self.denoise_timesteps = [1, 2, 4, 8, 16, 32, 128]

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, dt, external_cond=None, context_mask=None):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        """

        B, C, H, W = x.shape

        # add spatial embeddings
        x = self.x_embedder(x)  # (B, C, H, W) -> (B, H/2, W/2, D) , C = 16, D = d_model
        # embed noise steps
        c = self.t_embedder(t)  # (B, D)
        dt = self.dt_embedder(dt)  # (B, D)
        c += dt
        if torch.is_tensor(external_cond):
            # to float
            external_cond = external_cond.to(self.dtype)
            if context_mask is not None:
                # context_mask is a tensor of 1s and 0s
                # we want to zero out the external_cond for the sample where context_mask is 0
                # external_cond is a tensor of shape (B, D)
                external_cond = external_cond * context_mask
            out = self.external_cond(external_cond)
            c += out
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)  # (N, T, H, W, D)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

def shortcut_transformer_small():
    return ShortcutTransformer(
        input_h=28,
        input_w=28,
        in_channels=3,
        patch_size=2,
        hidden_size=256,
        depth=16,
        num_heads=16,
        external_cond_dim=1,
    )


def shortcut_transformer_mnist(input_h=28, input_w=28, in_channels=1, patch_size=1, external_cond_dim=1):
    return ShortcutTransformer(
        input_h=input_h,
        input_w=input_w,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=256,
        depth=6,
        num_heads=16,
        external_cond_dim=external_cond_dim,
    )
