import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Permutation(nn.Module):
    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(nn.Module):
    USE_SPDA = True

    def __init__(self, in_channels: int, head_channels: int):
        super().__init__()
        if in_channels % head_channels != 0:
            raise ValueError("in_channels must be divisible by head_channels")
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self.v_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}

    def forward_spda(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.qkv(x).reshape(batch, seq_len, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale ** 2 / temp
        if mask is not None:
            mask = mask.bool()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(batch, seq_len, channels)
        return self.proj(x)

    def forward_base(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.qkv(x).reshape(batch, seq_len, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum("bmhd,bnhd->bmnh", q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum("bmnh,bnhd->bmhd", attn, v).reshape(batch, seq_len, channels)
        return self.proj(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(channels, head_channels)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(channels, expansion)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 6 * channels, bias=True),
        )
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(y).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_mask,
            attn_temp,
            which_cache,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(channels, out_channels, bias=True)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True),
        )
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class MetaBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_tokens: int,
        permutation: Permutation,
        num_layers: int,
        head_dim: int,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
        global_cond_dim: int = 0,
        token_cond_dim: int = 0,
        label_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, channels) * 1e-2)
        self.label_drop_prob = label_drop_prob
        self.nvp = nvp
        self.permutation = permutation
        self.num_tokens = num_tokens
        self.channels = channels

        if num_classes > 0:
            self.class_embed = nn.Parameter(torch.randn(num_classes, channels) * 1e-2)
            self.fake_latent = nn.Parameter(torch.randn(1, channels) * 1e-2)
        else:
            self.class_embed = None
            self.fake_latent = None

        self.global_context_proj = nn.Linear(global_cond_dim, channels) if global_cond_dim > 0 else None
        self.token_context_proj = nn.Linear(token_cond_dim, channels) if token_cond_dim > 0 else None

        self.attn_blocks = nn.ModuleList([AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)])
        out_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = FinalLayer(channels, out_dim)
        self.register_buffer("attn_mask", torch.tril(torch.ones(num_tokens, num_tokens)))

    def _global_condition(self, x: torch.Tensor, context_global: torch.Tensor | None) -> torch.Tensor:
        if context_global is not None:
            if context_global.dtype in (torch.int32, torch.int64):
                if self.class_embed is None:
                    raise ValueError("Class indices were provided but this flow has no class embeddings.")
                class_embed = self.class_embed[context_global]
                if self.training and self.label_drop_prob > 0:
                    drop_mask = (torch.rand(x.shape[0], device=x.device) < self.label_drop_prob).unsqueeze(1).to(x.dtype)
                    class_embed = drop_mask * self.fake_latent + (1 - drop_mask) * class_embed
                return class_embed.to(dtype=x.dtype)

            if context_global.ndim != 2:
                raise ValueError(f"Global context must have shape [B, C], got {tuple(context_global.shape)}")
            if self.global_context_proj is None:
                if context_global.shape[1] != self.channels:
                    raise ValueError(
                        f"Global context must have shape [B, {self.channels}] when no projector is configured, "
                        f"got {tuple(context_global.shape)}"
                    )
                return context_global.to(device=x.device, dtype=x.dtype)
            return self.global_context_proj(context_global.to(device=x.device, dtype=x.dtype))

        if self.class_embed is not None:
            return self.fake_latent.repeat(x.shape[0], 1)
        return torch.zeros(x.shape[0], self.channels, device=x.device, dtype=x.dtype)

    def _token_condition(self, x: torch.Tensor, context_tokens: torch.Tensor | None) -> torch.Tensor | None:
        if context_tokens is None:
            return None
        if context_tokens.ndim != 3:
            raise ValueError(f"Token context must have shape [B, L, C], got {tuple(context_tokens.shape)}")
        if context_tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Token context length must match input tokens. Expected {self.num_tokens}, got {context_tokens.shape[1]}"
            )
        if self.token_context_proj is None:
            if context_tokens.shape[2] != self.channels:
                raise ValueError(
                    f"Token context must have feature size {self.channels} when no projector is configured, "
                    f"got {context_tokens.shape[2]}"
                )
            return context_tokens.to(device=x.device, dtype=x.dtype)
        return self.token_context_proj(context_tokens.to(device=x.device, dtype=x.dtype))

    def forward(
        self,
        x: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        x_in = x
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x = self.proj_in(x) + pos_embed
        token_context = self._token_condition(x, context_tokens)
        if token_context is not None:
            token_context = self.permutation(token_context)
            x = x + token_context
        global_context = self._global_condition(x, context_global)

        for block in self.attn_blocks:
            x = block(x, global_context, self.attn_mask)
        x = self.proj_out(x, global_context)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        z = self.permutation((x_in - xb) * scale, inverse=True)
        logdet = -xa.mean(dim=(1, 2))
        return z, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        global_context: torch.Tensor,
        token_context: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]
        x = self.proj_in(x_in) + pos_embed[i : i + 1]
        if token_context is not None:
            x = x + token_context[:, i : i + 1]
        for block in self.attn_blocks:
            x = block(x, global_context, attn_temp=attn_temp, which_cache=which_cache)
        x = self.proj_out(x, global_context)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True) -> None:
        for module in self.modules():
            if isinstance(module, Attention):
                module.sample = flag
                module.k_cache = {"cond": [], "uncond": []}
                module.v_cache = {"cond": [], "uncond": []}

    def reverse(
        self,
        x: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
        attn_temp: float = 1.0,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        token_context = self._token_condition(x, context_tokens)
        if token_context is not None:
            token_context = self.permutation(token_context)
        global_context = self._global_condition(x, context_global)
        self.set_sample_mode(True)
        for index in range(x.size(1) - 1):
            za, zb = self.reverse_step(
                x,
                pos_embed,
                index,
                global_context=global_context,
                token_context=token_context,
                which_cache="cond",
                attn_temp=attn_temp,
            )
            scale = za[:, 0].float().exp().type(za.dtype)
            x[:, index + 1] = x[:, index + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class TokenFlow(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int | list[int],
        num_heads: int,
        nvp: bool = True,
        num_classes: int = 0,
        global_cond_dim: int = 0,
        token_cond_dim: int = 0,
        label_drop_prob: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.channels = channels
        self.global_cond_dim = global_cond_dim
        self.token_cond_dim = token_cond_dim

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * num_blocks
        if len(layers_per_block) != num_blocks:
            raise ValueError("layers_per_block must have one entry per block")

        permutations = [PermutationIdentity(num_tokens), PermutationFlip(num_tokens)]
        self.blocks = nn.ModuleList(
            [
                MetaBlock(
                    in_channels=token_dim,
                    channels=channels,
                    num_tokens=num_tokens,
                    permutation=permutations[index % 2],
                    num_layers=layers_per_block[index],
                    head_dim=channels // num_heads,
                    nvp=nvp,
                    num_classes=num_classes,
                    global_cond_dim=global_cond_dim,
                    token_cond_dim=token_cond_dim,
                    label_drop_prob=label_drop_prob,
                )
                for index in range(num_blocks)
            ]
        )
        self.register_buffer("var", torch.ones(num_tokens, token_dim))

    def _validate_tokens(self, x_tokens: torch.Tensor) -> None:
        if x_tokens.ndim != 3:
            raise ValueError(f"Expected token input [B, L, D], got {tuple(x_tokens.shape)}")
        if x_tokens.shape[1] != self.num_tokens or x_tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"Expected tokens [B, {self.num_tokens}, {self.token_dim}], got {tuple(x_tokens.shape)}"
            )

    def forward_tokens(
        self,
        x_tokens: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_tokens(x_tokens)
        logdets = torch.zeros(x_tokens.shape[0], device=x_tokens.device, dtype=x_tokens.dtype)
        tokens = x_tokens
        for block in self.blocks:
            tokens, logdet = block(tokens, context_global=context_global, context_tokens=context_tokens)
            logdets = logdets + logdet
        return tokens, logdets

    def get_loss(self, z_tokens: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        return 0.5 * z_tokens.pow(2).mean() - logdets.mean()

    def reverse_tokens(
        self,
        z_tokens: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_tokens(z_tokens)
        x = z_tokens * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, context_global=context_global, context_tokens=context_tokens)
        return x


class Model(nn.Module):
    """
    Backward-compatible map adapter over the token-first flow core.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int | tuple[int, int],
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int | list[int],
        num_heads: int,
        nvp: bool = True,
        num_classes: int = 0,
        global_cond_dim: int = 0,
        token_cond_dim: int = 0,
        label_drop_prob: float = 0.0,
    ):
        super().__init__()
        if isinstance(img_size, int):
            self.img_hw = (img_size, img_size)
        else:
            self.img_hw = img_size
        if self.img_hw[0] % patch_size != 0 or self.img_hw[1] % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size in both dimensions.")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.token_dim = in_channels * patch_size ** 2
        self.num_tokens = (self.img_hw[0] // patch_size) * (self.img_hw[1] // patch_size)
        self.token_flow = TokenFlow(
            token_dim=self.token_dim,
            num_tokens=self.num_tokens,
            channels=channels,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            num_heads=num_heads,
            nvp=nvp,
            num_classes=num_classes,
            global_cond_dim=global_cond_dim,
            token_cond_dim=token_cond_dim,
            label_drop_prob=label_drop_prob,
        )

    def patchify(self, x_map: torch.Tensor) -> torch.Tensor:
        if x_map.ndim != 4:
            raise ValueError(f"Expected map input [B, C, H, W], got {tuple(x_map.shape)}")
        patches = F.unfold(x_map, kernel_size=self.patch_size, stride=self.patch_size)
        return patches.transpose(1, 2)

    def unpatchify(self, x_tokens: torch.Tensor, output_shape: tuple[int, int] | None = None) -> torch.Tensor:
        out_hw = self.img_hw if output_shape is None else output_shape
        patches = x_tokens.transpose(1, 2)
        return F.fold(patches, out_hw, self.patch_size, stride=self.patch_size)

    def forward_tokens(
        self,
        x_tokens: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.token_flow.forward_tokens(
            x_tokens,
            context_global=context_global,
            context_tokens=context_tokens,
        )

    def reverse_tokens(
        self,
        z_tokens: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.token_flow.reverse_tokens(
            z_tokens,
            context_global=context_global,
            context_tokens=context_tokens,
        )

    def forward_map(
        self,
        x_map: torch.Tensor,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_context = None if context_tokens is None else self.patchify(context_tokens)
        return self.forward_tokens(
            self.patchify(x_map),
            context_global=context_global,
            context_tokens=token_context,
        )

    def reverse_map(
        self,
        z_tokens: torch.Tensor,
        output_shape: tuple[int, int] | None = None,
        context_global: torch.Tensor | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_shape is None:
            output_shape = self.img_hw
        token_context = None if context_tokens is None else self.patchify(context_tokens)
        x_tokens = self.reverse_tokens(
            z_tokens,
            context_global=context_global,
            context_tokens=token_context,
        )
        return self.unpatchify(x_tokens, output_shape=output_shape)

    def get_loss(self, z_tokens: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        return self.token_flow.get_loss(z_tokens, logdets)

    def forward(self, x_map: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        z_tokens, logdet = self.forward_map(x_map, context_global=y)
        return z_tokens, [], logdet

    def reverse(self, z_tokens: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.reverse_map(z_tokens, context_global=y)
