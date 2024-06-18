import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from models.blocks import CrossAttention, TimestepEmbedder, TextEmbedder, FinalLayer
from models.blocks import modulate, get_2d_sincos_pos_embed


class EDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **kwargs)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, t, y, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + gate_msa * self.attn1(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.attn2(x, y, mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class EDiT(nn.Module):
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            embedding_dim=768,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            uncond_prob=0.1,
            learn_sigma=True,
            **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(embedding_dim, hidden_size, uncond_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.blocks = nn.ModuleList([
            EDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.adaLN_modulation[1].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.attn2.proj.weight, 0)
            nn.init.constant_(block.attn2.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        return imgs

    def forward(self, x, timestep, y, mask=None, **kwargs):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(timestep)
        t0 = self.adaLN_modulation(t)
        y = self.y_embedder(y, self.training)

        for block in self.blocks:
            x = block(x, t0, y, mask, **kwargs)
        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def EDiT_XL_2(**kwargs):
    return EDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


EDiT_models = {
    "EDiT-XL/2": EDiT_XL_2,
}
