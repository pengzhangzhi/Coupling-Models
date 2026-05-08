from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_flow import TokenFlow
from utils.enhancer import EnhancerCompressionSpec, EnhancerDatasetConfig


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[...]


class FisherFlowEnhancerClassifier(nn.Module):
    """
    Checkpoint-compatible classifier model for Fisher-Flow enhancer FBD assets.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        hidden: int,
        num_cls: int,
        depth: int,
        dropout: float = 0.0,
        mode: str = "",
        prior_pseudocount: float = 2.0,
        cls_expanded_simplex: bool = False,
        clean_data: bool = True,
        classifier: bool = True,
        classifier_free_guidance: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.hidden = hidden
        self.mode = mode
        self.depth = depth
        self.dropout = dropout
        self.prior_pseudocount = prior_pseudocount
        self.cls_expanded_simplex = cls_expanded_simplex
        self.classifier = classifier
        self.cls_free_guidance = classifier_free_guidance
        self.clean_data = clean_data
        self.num_cls = num_cls

        if self.clean_data:
            self.linear = nn.Embedding(self.dim, embedding_dim=hidden)
        else:
            expanded_simplex_input = self.cls_expanded_simplex or (
                not classifier and (self.mode == "dirichlet" or self.mode == "riemannian")
            )
            inp_size = self.dim * (2 if expanded_simplex_input else 1)
            if (self.mode == "ardm" or self.mode == "lrar") and not classifier:
                inp_size += 1
            self.linear = nn.Conv1d(inp_size, self.hidden, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(
                GaussianFourierProjection(embed_dim=self.hidden),
                nn.Linear(self.hidden, self.hidden),
            )

        self.num_layers = 5 * self.depth
        self.convs = [
            nn.Conv1d(self.hidden, self.hidden, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=64, padding=256),
        ]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for _ in range(self.depth)])
        self.time_layers = nn.ModuleList([Dense(self.hidden, self.hidden) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.hidden, self.hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden, self.hidden if classifier else self.dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(self.dropout)
        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.num_cls),
            )
        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=self.hidden)
            self.cls_layers = nn.ModuleList([Dense(self.hidden, self.hidden) for _ in range(self.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cls: torch.Tensor | None = None,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if t is not None:
            seq = x.view(-1, self.k, self.dim)
        else:
            seq = x
        if self.clean_data and seq.ndim == 3:
            seq = seq.argmax(dim=-1)
        if t is not None and len(t.shape) == 0:
            t = t[None].expand(seq.size(0))
        if self.clean_data:
            feat = self.linear(seq.long())
            feat = feat.permute(0, 2, 1)
        else:
            if t is None:
                raise ValueError("Time input is required when clean_data is False.")
            if len(t.shape) > 1:
                t = t.squeeze()
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.cls_free_guidance and not self.classifier:
            if cls is None:
                raise ValueError("Class conditioning is required when classifier_free_guidance is enabled.")
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i](h.permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            feat = h + feat if h.shape == feat.shape else h

        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            return self.cls_head(feat)
        return feat


class EnhancerConditionedConvStack(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        dilations = [1, 1, 4, 16, 64] * depth
        self.num_layers = len(dilations)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4 * dilation, dilation=dilation)
                for dilation in dilations
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in dilations])
        self.cond_layers = nn.ModuleList([Dense(cond_dim, hidden_dim) for _ in dilations])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for conv, norm, cond_layer in zip(self.convs, self.norms, self.cond_layers):
            h = self.dropout(x.clone())
            h = h + cond_layer(cond)[:, :, None]
            h = norm(h.permute(0, 2, 1))
            h = F.relu(conv(h.permute(0, 2, 1)))
            x = x + h if h.shape == x.shape else h
        return x

    def forward_with_capture(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        capture_layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if capture_layer_idx < 0 or capture_layer_idx >= self.num_layers:
            raise ValueError(f"capture_layer_idx={capture_layer_idx} is out of range for {self.num_layers} layers.")

        captured: torch.Tensor | None = None
        for layer_idx, (conv, norm, cond_layer) in enumerate(zip(self.convs, self.norms, self.cond_layers)):
            h = self.dropout(x.clone())
            h = h + cond_layer(cond)[:, :, None]
            h = norm(h.permute(0, 2, 1))
            h = F.relu(conv(h.permute(0, 2, 1)))
            x = x + h if h.shape == x.shape else h
            if layer_idx == capture_layer_idx:
                captured = x

        if captured is None:
            raise RuntimeError("Failed to capture an intermediate decoder feature.")
        return x, captured


class EnhancerEncoder(nn.Module):
    def __init__(
        self,
        spec: EnhancerCompressionSpec,
        dataset: EnhancerDatasetConfig,
        latent_channels: int,
        hidden_dim: int,
        depth: int,
        external_embed_mode: str = "off",
        external_embed_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spec = spec
        self.seq_len = dataset.seq_len
        self.hidden_dim = hidden_dim
        self.external_embed_mode = external_embed_mode
        self.stem = nn.Conv1d(dataset.vocab_size, hidden_dim, kernel_size=9, padding=4)
        self.class_embed = nn.Embedding(dataset.num_classes, hidden_dim)
        cond_dim = hidden_dim
        if external_embed_mode == "encoder_cond":
            if external_embed_dim <= 0:
                raise ValueError("external_embed_dim must be positive when external_embed_mode='encoder_cond'.")
            self.external_proj = nn.Sequential(
                nn.Linear(external_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            cond_dim = hidden_dim * 2
        else:
            self.external_proj = None
        self.stack = EnhancerConditionedConvStack(hidden_dim, depth=depth, cond_dim=cond_dim, dropout=dropout)
        self.out = nn.Conv1d(hidden_dim, latent_channels, kernel_size=3, padding=1)

    def forward(
        self,
        seq_onehot: torch.Tensor,
        labels: torch.Tensor,
        external_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.silu(self.stem(seq_onehot.transpose(1, 2)))
        cond = self.class_embed(labels)
        if self.external_embed_mode == "encoder_cond":
            if external_embedding is None:
                external_cond = cond.new_zeros(cond.shape[0], self.hidden_dim)
            else:
                external_cond = self.external_proj(external_embedding)
            cond = torch.cat([cond, external_cond], dim=-1)
        x = self.stack(x, cond)
        x = self.out(F.silu(x))
        x = F.adaptive_avg_pool1d(x, self.spec.latent_length)
        return x.transpose(1, 2)


class EnhancerSequenceDecoder(nn.Module):
    def __init__(
        self,
        spec: EnhancerCompressionSpec,
        dataset: EnhancerDatasetConfig,
        latent_channels: int,
        hidden_dim: int,
        depth: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = dataset.seq_len
        self.vocab_size = dataset.vocab_size
        self.latent_proj = nn.Conv1d(latent_channels, hidden_dim, kernel_size=3, padding=1)
        self.in_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4)
        self.class_embed = nn.Embedding(dataset.num_classes, hidden_dim)
        self.stack = EnhancerConditionedConvStack(hidden_dim, depth=depth, cond_dim=hidden_dim, dropout=dropout)
        self.out = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.vocab_size, kernel_size=1),
        )
        self.feature_capture_idx = max(0, math.ceil(self.stack.num_layers * 0.25) - 1)

    def _decode_inputs(self, latent_tokens: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = latent_tokens.transpose(1, 2)
        x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        x = F.silu(self.latent_proj(x))
        x = F.silu(self.in_proj(x))
        cond = self.class_embed(labels)
        return x, cond

    def forward(self, latent_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x, cond = self._decode_inputs(latent_tokens, labels)
        x = self.stack(x, cond)
        return self.out(x).transpose(1, 2)

    def forward_with_feature(self, latent_tokens: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, cond = self._decode_inputs(latent_tokens, labels)
        x, feature = self.stack.forward_with_capture(x, cond, capture_layer_idx=self.feature_capture_idx)
        logits = self.out(x).transpose(1, 2)
        return logits, feature.transpose(1, 2)


class EnhancerEncoderDecoder(nn.Module):
    def __init__(
        self,
        spec: EnhancerCompressionSpec,
        dataset: EnhancerDatasetConfig,
        latent_channels: int,
        encoder_width: int,
        decoder_width: int,
        encoder_depth: int,
        decoder_depth: int,
        fixed_std: float,
        external_embed_mode: str = "off",
        external_embed_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.external_embed_mode = external_embed_mode
        self.encoder = EnhancerEncoder(
            spec=spec,
            dataset=dataset,
            latent_channels=latent_channels,
            hidden_dim=encoder_width,
            depth=encoder_depth,
            external_embed_mode=external_embed_mode,
            external_embed_dim=external_embed_dim,
            dropout=dropout,
        )
        self.decoder = EnhancerSequenceDecoder(
            spec=spec,
            dataset=dataset,
            latent_channels=latent_channels,
            hidden_dim=decoder_width,
            depth=decoder_depth,
            dropout=dropout,
        )
        if external_embed_mode == "align":
            if external_embed_dim <= 0:
                raise ValueError("external_embed_dim must be positive when external_embed_mode='align'.")
            self.latent_align_proj = nn.Sequential(
                nn.Linear(latent_channels, encoder_width),
                nn.SiLU(),
                nn.Linear(encoder_width, external_embed_dim),
            )
        else:
            self.latent_align_proj = None
        self.fixed_std = fixed_std

    def encode_mean(
        self,
        seq_onehot: torch.Tensor,
        labels: torch.Tensor,
        external_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(seq_onehot, labels, external_embedding=external_embedding)

    def sample_latent(self, latent_mean: torch.Tensor) -> torch.Tensor:
        return latent_mean + self.fixed_std * torch.randn_like(latent_mean)

    def decode(self, latent_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_tokens, labels)

    def alignment_loss(self, latent_mean: torch.Tensor, external_embedding: torch.Tensor) -> torch.Tensor:
        if self.latent_align_proj is None:
            raise RuntimeError("Alignment loss requested when external_embed_mode is not 'align'.")
        pooled_latent = latent_mean.mean(dim=1)
        projected_latent = self.latent_align_proj(pooled_latent)
        return (1.0 - F.cosine_similarity(projected_latent, external_embedding.detach(), dim=-1)).mean()

    def forward(
        self,
        seq_onehot: torch.Tensor,
        labels: torch.Tensor,
        deterministic: bool = False,
        external_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_mean = self.encode_mean(seq_onehot, labels, external_embedding=external_embedding)
        latent = latent_mean if deterministic else self.sample_latent(latent_mean)
        logits = self.decode(latent, labels)
        return latent_mean, latent, logits


class ConditionalEnhancerLatentFlow(nn.Module):
    def __init__(
        self,
        spec: EnhancerCompressionSpec,
        dataset: EnhancerDatasetConfig,
        token_dim: int,
        flow_width: int,
        num_blocks: int,
        layers_per_block: int,
        num_heads: int,
    ):
        super().__init__()
        self.spec = spec
        self.flow = TokenFlow(
            token_dim=token_dim,
            num_tokens=spec.latent_length,
            channels=flow_width,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            num_heads=num_heads,
            num_classes=dataset.num_classes,
            global_cond_dim=0,
            token_cond_dim=0,
            label_drop_prob=0.0,
        )

    def forward(self, x_tokens: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_tokens.shape[1] != self.spec.latent_length:
            raise ValueError(f"Expected latent length {self.spec.latent_length}, got {x_tokens.shape[1]}")
        return self.flow.forward_tokens(x_tokens, context_global=labels)

    def reverse(self, z_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.flow.reverse_tokens(z_tokens, context_global=labels)

    def get_loss(self, z_tokens: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        return self.flow.get_loss(z_tokens, logdet)


class EnhancerOneStepGenerator(nn.Module):
    def __init__(
        self,
        spec: EnhancerCompressionSpec,
        dataset: EnhancerDatasetConfig,
        latent_channels: int,
        hidden_dim: int,
        depth: int,
        stage2_align_mode: str = "off",
        external_embed_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stage2_align_mode = stage2_align_mode
        self.decoder = EnhancerSequenceDecoder(
            spec=spec,
            dataset=dataset,
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        if stage2_align_mode == "shallow_cosine":
            if external_embed_dim <= 0:
                raise ValueError("external_embed_dim must be positive when stage2_align_mode='shallow_cosine'.")
            self.stage2_align_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, external_embed_dim),
            )
        else:
            self.stage2_align_proj = None

    def forward(self, z_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_tokens, labels)

    def forward_with_alignment_feature(
        self,
        z_tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.forward_with_feature(z_tokens, labels)

    def alignment_loss(self, feature_tokens: torch.Tensor, external_embedding: torch.Tensor) -> torch.Tensor:
        if self.stage2_align_proj is None:
            raise RuntimeError("Alignment loss requested when stage2_align_mode is not 'shallow_cosine'.")
        pooled_feature = feature_tokens.mean(dim=1)
        projected_feature = self.stage2_align_proj(pooled_feature)
        return (1.0 - F.cosine_similarity(projected_feature, external_embedding.detach(), dim=-1)).mean()
