import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.enhancer import (
    EnhancerCompressionSpec,
    EnhancerDatasetConfig,
    decode_tokens,
    default_classifier_checkpoint,
    get_enhancer_dataset_config,
)
from utils.enhancer_dataset import EnhancerTransportDatasetView
from utils.enhancer_external import FROZEN_CLASSIFIER_EMBED_DIM, FrozenEnhancerClassifierEmbedder
from utils.enhancer_fbd import EnhancerFBDEvaluator, get_wasserstein_dist
from utils.enhancer_models import (
    ConditionalEnhancerLatentFlow,
    EnhancerEncoderDecoder,
    EnhancerOneStepGenerator,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


_WANDB_LOGGING_BROKEN = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def logits_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction="mean")


def fixed_variance_kl_loss(latent_mean: torch.Tensor) -> torch.Tensor:
    return 0.5 * latent_mean.pow(2).mean()


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == targets).float().mean()


def log_wandb(data: dict[str, Any]) -> None:
    global _WANDB_LOGGING_BROKEN
    if wandb is not None and wandb.run is not None:
        try:
            wandb.log(data)
        except Exception as exc:  # pragma: no cover - best effort logging
            if not _WANDB_LOGGING_BROKEN:
                print(f"[warn] wandb.log failed; continuing without W&B logging: {exc}")
                _WANDB_LOGGING_BROKEN = True


def init_wandb(args: argparse.Namespace) -> None:
    if wandb is None:
        return
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        dir=args.output_dir,
    )


def compression_spec_from_args(args: argparse.Namespace, dataset: EnhancerDatasetConfig) -> EnhancerCompressionSpec:
    return EnhancerCompressionSpec.from_ratio(dataset.seq_len, args.compression_ratio)


def make_dataloaders(args: argparse.Namespace, dataset: EnhancerDatasetConfig) -> tuple[DataLoader, DataLoader]:
    train_set = EnhancerTransportDatasetView(args.data_dir, dataset, split="train")
    valid_split = "test" if args.val_on_test else "valid"
    valid_set = EnhancerTransportDatasetView(args.data_dir, dataset, split=valid_split)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, valid_loader


def make_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def default_checkpoint_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "checkpoints" / "last.pt"


def best_fbd_checkpoint_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "checkpoints" / "best_fbd.pt"


def checkpoint_path_for_args(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path)
    return default_checkpoint_path(args)


def checkpoint_external_embedding_config(args: argparse.Namespace) -> dict[str, str | float]:
    return {
        "mode": args.external_embed_mode,
        "align_weight": args.external_align_weight,
    }


def checkpoint_stage2_alignment_config(args: argparse.Namespace) -> dict[str, str | float]:
    return {
        "mode": args.stage2_align_mode,
        "weight": args.stage2_align_weight,
    }


def save_checkpoint(
    path: Path,
    autoencoder: EnhancerEncoderDecoder,
    flow: ConditionalEnhancerLatentFlow,
    generator: EnhancerOneStepGenerator,
    args: argparse.Namespace,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
    epoch: int,
    global_step: int,
    best_val_fbd: float | None = None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "autoencoder": autoencoder.state_dict(),
            "flow": flow.state_dict(),
            "generator": generator.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "dataset": dataset.name,
            "compression_spec": {
                "seq_len": spec.seq_len,
                "compression_ratio": spec.compression_ratio,
                "latent_length": spec.latent_length,
            },
            "external_embedding": checkpoint_external_embedding_config(args),
            "stage2_alignment": checkpoint_stage2_alignment_config(args),
            "best_val_fbd": best_val_fbd,
        },
        path,
    )
    print(f"[checkpoint] saved {path}")


def load_checkpoint(
    path: Path,
    autoencoder: EnhancerEncoderDecoder,
    flow: ConditionalEnhancerLatentFlow,
    generator: EnhancerOneStepGenerator,
    args: argparse.Namespace,
    device: torch.device,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("dataset") != dataset.name:
        raise ValueError(f"Checkpoint dataset={payload.get('dataset')} does not match requested dataset={dataset.name}.")
    saved_spec = payload.get("compression_spec")
    if saved_spec is None:
        raise ValueError("Checkpoint is missing compression metadata.")
    if saved_spec["latent_length"] != spec.latent_length or saved_spec["compression_ratio"] != spec.compression_ratio:
        raise ValueError(
            f"Checkpoint compression spec {saved_spec} does not match current spec "
            f"(ratio={spec.compression_ratio}, latent_length={spec.latent_length})."
        )
    saved_external = payload.get("external_embedding")
    if saved_external is None:
        payload_args = payload.get("args", {})
        saved_external = {
            "mode": payload_args.get("external_embed_mode", "off"),
            "align_weight": payload_args.get("external_align_weight", 0.1),
        }
    saved_external_mode = str(saved_external.get("mode", "off"))
    if saved_external_mode != args.external_embed_mode:
        raise ValueError(
            f"Checkpoint external_embed_mode={saved_external_mode} does not match requested "
            f"external_embed_mode={args.external_embed_mode}."
        )
    saved_stage2 = payload.get("stage2_alignment")
    if saved_stage2 is None:
        payload_args = payload.get("args", {})
        saved_stage2 = {
            "mode": payload_args.get("stage2_align_mode", "off"),
            "weight": payload_args.get("stage2_align_weight", 0.0),
        }
    saved_stage2_mode = str(saved_stage2.get("mode", "off"))
    if saved_stage2_mode != args.stage2_align_mode:
        raise ValueError(
            f"Checkpoint stage2_align_mode={saved_stage2_mode} does not match requested "
            f"stage2_align_mode={args.stage2_align_mode}."
        )
    autoencoder.load_state_dict(payload["autoencoder"])
    flow.load_state_dict(payload["flow"])
    generator.load_state_dict(payload["generator"])
    print(f"[checkpoint] loaded {path}")
    return payload


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_path is not None:
        return Path(args.metrics_path)
    return Path(args.output_dir) / "metrics.jsonl"


def append_metrics_record(
    args: argparse.Namespace,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
    event: str,
    epoch: int | None,
    **metrics: float | int | str | None,
) -> None:
    path = metrics_path(args)
    ensure_dir(path.parent)
    classifier_ckpt = resolve_classifier_ckpt(args, dataset) if classifier_ckpt_required(args) else None
    record = {
        "event": event,
        "epoch": epoch,
        "seed": args.seed,
        "dataset": dataset.name,
        "seq_len": dataset.seq_len,
        "num_classes": dataset.num_classes,
        "compression_ratio": spec.compression_ratio,
        "latent_length": spec.latent_length,
        "latent_channels": args.latent_channels,
        "fixed_std": args.fixed_std,
        "beta_kl": args.beta_kl,
        "external_embed_mode": args.external_embed_mode,
        "external_align_weight": args.external_align_weight,
        "stage2_align_mode": args.stage2_align_mode,
        "stage2_align_weight": args.stage2_align_weight,
        "encoder_width": args.encoder_width,
        "decoder_width": args.decoder_width,
        "generator_width": args.generator_width,
        "encoder_depth": args.encoder_depth,
        "decoder_depth": args.decoder_depth,
        "generator_depth": args.generator_depth,
        "flow_width": args.flow_width,
        "flow_blocks": args.flow_blocks,
        "flow_layers": args.flow_layers,
        "flow_heads": args.flow_heads,
        "output_dir": str(Path(args.output_dir).resolve()),
        "checkpoint_path": str(checkpoint_path_for_args(args).resolve()),
        "classifier_ckpt": None if classifier_ckpt is None else str(classifier_ckpt.resolve()),
        "external_embed_ckpt": (
            None
            if not (external_embeddings_enabled(args) or stage2_alignment_enabled(args))
            else str(resolve_external_embed_ckpt(args, dataset).resolve())
        ),
        "skip_fbd": args.skip_fbd,
    }
    record.update(metrics)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def resolve_classifier_ckpt(args: argparse.Namespace, dataset: EnhancerDatasetConfig) -> Path:
    if args.classifier_ckpt is not None:
        return Path(args.classifier_ckpt)
    return default_classifier_checkpoint(args.data_dir, dataset)


def external_embeddings_enabled(args: argparse.Namespace) -> bool:
    return args.external_embed_mode != "off"


def stage2_alignment_enabled(args: argparse.Namespace) -> bool:
    return args.stage2_align_mode != "off"


def classifier_ckpt_required(args: argparse.Namespace) -> bool:
    return not args.skip_fbd or external_embeddings_enabled(args) or stage2_alignment_enabled(args)


def resolve_external_embed_ckpt(args: argparse.Namespace, dataset: EnhancerDatasetConfig) -> Path:
    return resolve_classifier_ckpt(args, dataset)


def sample_generator_tokens(
    generator: EnhancerOneStepGenerator,
    labels: torch.Tensor,
    spec: EnhancerCompressionSpec,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    was_training = generator.training
    generator.eval()
    for start in range(0, labels.shape[0], batch_size):
        batch_labels = labels[start : start + batch_size].to(device)
        z_tokens = torch.randn(batch_labels.shape[0], spec.latent_length, args.latent_channels, device=device)
        logits = generator(z_tokens, batch_labels)
        outputs.append(logits.argmax(dim=-1).cpu())
    generator.train(was_training)
    return torch.cat(outputs, dim=0)


def sample_flow_reverse_tokens(
    autoencoder: EnhancerEncoderDecoder,
    flow: ConditionalEnhancerLatentFlow,
    labels: torch.Tensor,
    spec: EnhancerCompressionSpec,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    auto_was_training = autoencoder.training
    flow_was_training = flow.training
    autoencoder.eval()
    flow.eval()
    for start in range(0, labels.shape[0], batch_size):
        batch_labels = labels[start : start + batch_size].to(device)
        z_tokens = torch.randn(batch_labels.shape[0], spec.latent_length, args.latent_channels, device=device)
        latent_tokens = flow.reverse(z_tokens, batch_labels)
        logits = autoencoder.decode(latent_tokens, batch_labels)
        outputs.append(logits.argmax(dim=-1).cpu())
    autoencoder.train(auto_was_training)
    flow.train(flow_was_training)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_batch_to_z_tokens(
    autoencoder: EnhancerEncoderDecoder,
    flow: ConditionalEnhancerLatentFlow,
    seq_onehot: torch.Tensor,
    seq_tokens: torch.Tensor,
    labels: torch.Tensor,
    external_embedder: FrozenEnhancerClassifierEmbedder | None = None,
) -> torch.Tensor:
    external_embedding = None
    if external_embedder is not None and autoencoder.external_embed_mode == "encoder_cond":
        external_embedding = external_embedder(seq_tokens)
    _, latent, _ = autoencoder(
        seq_onehot,
        labels,
        deterministic=False,
        external_embedding=external_embedding,
    )
    z_tokens, _ = flow(latent, labels)
    return z_tokens


@torch.no_grad()
def evaluate_generator_ce(
    autoencoder: EnhancerEncoderDecoder,
    flow: ConditionalEnhancerLatentFlow,
    generator: EnhancerOneStepGenerator,
    loader: DataLoader,
    device: torch.device,
    external_embedder: FrozenEnhancerClassifierEmbedder | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    auto_was_training = autoencoder.training
    flow_was_training = flow.training
    gen_was_training = generator.training
    autoencoder.eval()
    flow.eval()
    generator.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch_idx, (seq_onehot, seq_tokens, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        seq_onehot = seq_onehot.to(device)
        seq_tokens = seq_tokens.to(device)
        labels = labels.to(device)
        z_tokens = encode_batch_to_z_tokens(
            autoencoder,
            flow,
            seq_onehot,
            seq_tokens,
            labels,
            external_embedder=external_embedder,
        )
        logits = generator(z_tokens, labels)
        total_loss += logits_ce_loss(logits, seq_tokens).item()
        total_acc += token_accuracy(logits, seq_tokens).item()
        count += 1

    autoencoder.train(auto_was_training)
    flow.train(flow_was_training)
    generator.train(gen_was_training)
    return total_loss / max(1, count), total_acc / max(1, count)


@torch.no_grad()
def evaluate_slm_style_fbd(
    generator: EnhancerOneStepGenerator,
    evaluator: EnhancerFBDEvaluator,
    loader: DataLoader,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    real_embeddings: list[np.ndarray] = []
    generated_embeddings: list[np.ndarray] = []
    split_name = "test" if args.val_on_test else "valid"

    # Match SLM's protocol: accumulate embeddings across the full validation epoch,
    # independent of the CE/accuracy debug cap.
    for batch_idx, (_, seq_tokens, labels) in enumerate(loader):
        seq_tokens = seq_tokens.cpu()
        labels = labels.cpu()
        gen_tokens = sample_generator_tokens(generator, labels, spec, args, device, batch_size=args.eval_batch_size)
        generated_embeddings.append(evaluator.embeddings(gen_tokens))
        real_embeddings.append(evaluator.embeddings(seq_tokens, cache_key=f"{dataset.name}:{split_name}:{batch_idx}"))

    if not real_embeddings:
        raise RuntimeError("Validation loader did not produce any sequences.")

    return get_wasserstein_dist(
        np.concatenate(generated_embeddings, axis=0),
        np.concatenate(real_embeddings, axis=0),
    )


@torch.no_grad()
def evaluate_flow_reverse_fbd(
    autoencoder: EnhancerEncoderDecoder,
    flow: ConditionalEnhancerLatentFlow,
    evaluator: EnhancerFBDEvaluator,
    loader: DataLoader,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    real_embeddings: list[np.ndarray] = []
    generated_embeddings: list[np.ndarray] = []
    split_name = "test" if args.val_on_test else "valid"

    for batch_idx, (_, seq_tokens, labels) in enumerate(loader):
        seq_tokens = seq_tokens.cpu()
        labels = labels.cpu()
        gen_tokens = sample_flow_reverse_tokens(
            autoencoder,
            flow,
            labels,
            spec,
            args,
            device,
            batch_size=args.eval_batch_size,
        )
        generated_embeddings.append(evaluator.embeddings(gen_tokens))
        real_embeddings.append(evaluator.embeddings(seq_tokens, cache_key=f"{dataset.name}:{split_name}:{batch_idx}"))

    if not real_embeddings:
        raise RuntimeError("Validation loader did not produce any sequences.")

    return get_wasserstein_dist(
        np.concatenate(generated_embeddings, axis=0),
        np.concatenate(real_embeddings, axis=0),
    )


@torch.no_grad()
def save_generated_examples(
    generator: EnhancerOneStepGenerator,
    loader: DataLoader,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> None:
    was_training = generator.training
    generator.eval()
    _, seq_tokens, labels = next(iter(loader))
    seq_tokens = seq_tokens[: args.sample_batch_size].cpu()
    labels = labels[: args.sample_batch_size].cpu()
    matched_gen = sample_generator_tokens(generator, labels, spec, args, device, batch_size=args.eval_batch_size)

    repeats = math.ceil(args.sample_batch_size / dataset.num_classes)
    free_labels = torch.arange(dataset.num_classes).repeat(repeats)[: args.sample_batch_size]
    free_gen = sample_generator_tokens(generator, free_labels, spec, args, device, batch_size=args.eval_batch_size)

    output_path = Path(args.output_dir) / f"samples_epoch_{epoch:03d}.jsonl"
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for label, sequence in zip(labels.tolist(), decode_tokens(seq_tokens)):
            handle.write(
                json.dumps(
                    {
                        "source": "validation_reference",
                        "dataset": dataset.name,
                        "target_class": label,
                        "sequence": sequence,
                        "epoch": epoch,
                        "checkpoint_path": str(default_checkpoint_path(args).resolve()),
                    }
                )
                + "\n"
            )
        for label, sequence in zip(labels.tolist(), decode_tokens(matched_gen)):
            handle.write(
                json.dumps(
                    {
                        "source": "validation_eval",
                        "dataset": dataset.name,
                        "target_class": label,
                        "sequence": sequence,
                        "epoch": epoch,
                        "checkpoint_path": str(default_checkpoint_path(args).resolve()),
                    }
                )
                + "\n"
            )
        for label, sequence in zip(free_labels.tolist(), decode_tokens(free_gen)):
            handle.write(
                json.dumps(
                    {
                        "source": "free_sampling",
                        "dataset": dataset.name,
                        "target_class": label,
                        "sequence": sequence,
                        "epoch": epoch,
                        "checkpoint_path": str(default_checkpoint_path(args).resolve()),
                    }
                )
                + "\n"
            )
    print(f"[samples] saved {output_path}")
    generator.train(was_training)


def make_models(
    args: argparse.Namespace,
    dataset: EnhancerDatasetConfig,
    spec: EnhancerCompressionSpec,
    device: torch.device,
) -> tuple[EnhancerEncoderDecoder, ConditionalEnhancerLatentFlow, EnhancerOneStepGenerator]:
    autoencoder = EnhancerEncoderDecoder(
        spec=spec,
        dataset=dataset,
        latent_channels=args.latent_channels,
        encoder_width=args.encoder_width,
        decoder_width=args.decoder_width,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        fixed_std=args.fixed_std,
        external_embed_mode=args.external_embed_mode,
        external_embed_dim=FROZEN_CLASSIFIER_EMBED_DIM,
        dropout=args.dropout,
    ).to(device)
    flow = ConditionalEnhancerLatentFlow(
        spec=spec,
        dataset=dataset,
        token_dim=args.latent_channels,
        flow_width=args.flow_width,
        num_blocks=args.flow_blocks,
        layers_per_block=args.flow_layers,
        num_heads=args.flow_heads,
    ).to(device)
    generator = EnhancerOneStepGenerator(
        spec=spec,
        dataset=dataset,
        latent_channels=args.latent_channels,
        hidden_dim=args.generator_width,
        depth=args.generator_depth,
        stage2_align_mode=args.stage2_align_mode,
        external_embed_dim=FROZEN_CLASSIFIER_EMBED_DIM,
        dropout=args.dropout,
    ).to(device)
    return autoencoder, flow, generator


def warmstart_generator_from_autoencoder(
    autoencoder: EnhancerEncoderDecoder,
    generator: EnhancerOneStepGenerator,
) -> bool:
    ae_state = autoencoder.decoder.state_dict()
    gen_state = generator.decoder.state_dict()
    if ae_state.keys() != gen_state.keys():
        return False
    if any(ae_state[name].shape != gen_state[name].shape for name in ae_state):
        return False
    generator.decoder.load_state_dict(ae_state)
    return True


def train(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output_dir))
    device = get_device()
    set_seed(args.seed)
    dataset = get_enhancer_dataset_config(args.dataset)
    spec = compression_spec_from_args(args, dataset)
    train_loader, valid_loader = make_dataloaders(args, dataset)
    autoencoder, flow, generator = make_models(args, dataset, spec, device)

    params = list(autoencoder.parameters()) + list(flow.parameters()) + list(generator.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, len(train_loader), args.epochs)
    global_step = 0
    best_val_fbd = float("inf")
    evaluator: EnhancerFBDEvaluator | None = None
    external_embedder: FrozenEnhancerClassifierEmbedder | None = None
    generator_warm_started = False
    classifier_ckpt = resolve_classifier_ckpt(args, dataset) if classifier_ckpt_required(args) else None
    if classifier_ckpt is not None and not classifier_ckpt.exists():
        raise FileNotFoundError(
            f"Missing enhancer classifier checkpoint: {classifier_ckpt}. "
            "Run scripts/download_fisher_workdir.sh <data_root> or pass --classifier-ckpt."
        )
    if external_embeddings_enabled(args) or stage2_alignment_enabled(args):
        external_embedder = FrozenEnhancerClassifierEmbedder(dataset, resolve_external_embed_ckpt(args, dataset), device)

    for epoch in range(1, args.epochs + 1):
        stage1 = epoch <= args.stage1_epochs
        phase = "stage1_ae_flow" if stage1 else "stage2_generator"

        if not stage1 and not generator_warm_started:
            warm_started = warmstart_generator_from_autoencoder(autoencoder, generator)
            if warm_started:
                print("[stage2] warm-started generator decoder from autoencoder decoder")
            else:
                print("[stage2] skipped generator warm-start because decoder shapes did not match")
            generator_warm_started = True

        for param in autoencoder.parameters():
            param.requires_grad = stage1
        for param in flow.parameters():
            param.requires_grad = stage1
        for param in generator.parameters():
            param.requires_grad = not stage1

        if stage1:
            autoencoder.train()
            flow.train()
            generator.eval()
        else:
            autoencoder.eval()
            flow.eval()
            generator.train()

        trainable_params = [param for param in params if param.requires_grad]
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_flow = 0.0
        epoch_kl = 0.0
        epoch_align = 0.0
        epoch_stage2_align = 0.0
        epoch_gen = 0.0
        epoch_acc = 0.0
        completed_steps = 0

        for step, (seq_onehot, seq_tokens, labels) in enumerate(train_loader, start=1):
            if args.max_train_steps_per_epoch is not None and step > args.max_train_steps_per_epoch:
                break
            seq_onehot = seq_onehot.to(device)
            seq_tokens = seq_tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            if stage1:
                external_embedding = None
                if external_embedder is not None and external_embeddings_enabled(args):
                    with torch.no_grad():
                        external_embedding = external_embedder(seq_tokens)
                latent_mean, latent, recon_logits = autoencoder(
                    seq_onehot,
                    labels,
                    deterministic=False,
                    external_embedding=external_embedding if args.external_embed_mode == "encoder_cond" else None,
                )
                z_tokens, logdet = flow(latent, labels)
                recon_loss = logits_ce_loss(recon_logits, seq_tokens)
                flow_loss = flow.get_loss(z_tokens, logdet)
                kl_loss = fixed_variance_kl_loss(latent_mean)
                if args.external_embed_mode == "align":
                    if external_embedding is None:
                        raise RuntimeError("Alignment mode requires external embeddings during stage 1.")
                    align_loss = autoencoder.alignment_loss(latent_mean, external_embedding)
                else:
                    align_loss = torch.zeros((), device=device)
                gen_loss = torch.zeros((), device=device)
                stage2_align_loss = torch.zeros((), device=device)
                total_loss = (
                    recon_loss
                    + args.lambda_flow * flow_loss
                    + args.beta_kl * kl_loss
                    + args.external_align_weight * align_loss
                )
                train_logits = recon_logits
            else:
                with torch.no_grad():
                    z_tokens = encode_batch_to_z_tokens(
                        autoencoder,
                        flow,
                        seq_onehot,
                        seq_tokens,
                        labels,
                        external_embedder=external_embedder,
                    )
                if stage2_alignment_enabled(args):
                    if external_embedder is None:
                        raise RuntimeError("Stage-2 alignment requires an external embedder.")
                    with torch.no_grad():
                        external_embedding = external_embedder(seq_tokens)
                    gen_logits, generator_feature = generator.forward_with_alignment_feature(z_tokens, labels)
                    stage2_align_loss = generator.alignment_loss(generator_feature, external_embedding)
                else:
                    gen_logits = generator(z_tokens, labels)
                    stage2_align_loss = torch.zeros((), device=device)
                gen_loss = logits_ce_loss(gen_logits, seq_tokens)
                recon_loss = torch.zeros((), device=device)
                flow_loss = torch.zeros((), device=device)
                kl_loss = torch.zeros((), device=device)
                align_loss = torch.zeros((), device=device)
                total_loss = gen_loss + args.stage2_align_weight * stage2_align_loss
                train_logits = gen_logits

            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            acc = token_accuracy(train_logits, seq_tokens)
            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_flow += flow_loss.item()
            epoch_kl += kl_loss.item()
            epoch_align += align_loss.item()
            epoch_stage2_align += stage2_align_loss.item()
            epoch_gen += gen_loss.item()
            epoch_acc += acc.item()

            if global_step % args.log_every == 0:
                print(
                    f"[train] phase={phase} epoch={epoch:03d} step={step:04d} "
                    f"total={total_loss.item():.4f} recon={recon_loss.item():.4f} "
                    f"flow={flow_loss.item():.4f} kl={kl_loss.item():.4f} "
                    f"align={align_loss.item():.4f} "
                    f"s2align={stage2_align_loss.item():.4f} "
                    f"gen={gen_loss.item():.4f} acc={acc.item():.4f}"
                )
                log_wandb(
                    {
                        "train/phase": 1 if stage1 else 2,
                        "train/total_loss": total_loss.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/flow_loss": flow_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/align_loss": align_loss.item(),
                        "train/stage2_align_loss": stage2_align_loss.item(),
                        "train/gen_loss": gen_loss.item(),
                        "train/acc": acc.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                        "train/epoch": epoch,
                    }
                )
            global_step += 1
            completed_steps = step

        steps = completed_steps if completed_steps > 0 else 1
        print(
            f"[epoch] phase={phase} epoch={epoch:03d} avg_total={epoch_total / steps:.4f} "
            f"avg_recon={epoch_recon / steps:.4f} avg_flow={epoch_flow / steps:.4f} "
            f"avg_kl={epoch_kl / steps:.4f} "
            f"avg_align={epoch_align / steps:.4f} avg_s2align={epoch_stage2_align / steps:.4f} "
            f"avg_gen={epoch_gen / steps:.4f} avg_acc={epoch_acc / steps:.4f}"
        )

        val_ce = None
        val_acc = None
        val_fbd = None
        flow_reverse_fbd = None
        if not stage1:
            val_ce, val_acc = evaluate_generator_ce(
                autoencoder,
                flow,
                generator,
                valid_loader,
                device,
                external_embedder=external_embedder,
                max_batches=args.max_eval_batches,
            )
            print(f"[eval] epoch={epoch:03d} val_ce={val_ce:.4f} val_acc={val_acc:.4f}")
            log_wandb({"eval/val_ce": val_ce, "eval/val_acc": val_acc, "eval/epoch": epoch})

        if not args.skip_fbd and epoch % args.fbd_every == 0:
            if evaluator is None:
                evaluator = EnhancerFBDEvaluator(dataset, classifier_ckpt, device)
            flow_reverse_fbd = evaluate_flow_reverse_fbd(autoencoder, flow, evaluator, valid_loader, dataset, spec, args, device)
            print(f"[eval] epoch={epoch:03d} flow_reverse_fbd={flow_reverse_fbd:.4f}")

            eval_log = {
                "eval/flow_reverse_fbd": flow_reverse_fbd,
                "eval/epoch": epoch,
            }
            if not stage1:
                val_fbd = evaluate_slm_style_fbd(generator, evaluator, valid_loader, dataset, spec, args, device)
                print(f"[eval] epoch={epoch:03d} val_fbd={val_fbd:.4f}")
                eval_log["eval/val_fbd"] = val_fbd
                if val_fbd < best_val_fbd:
                    best_val_fbd = val_fbd
                    save_checkpoint(
                        best_fbd_checkpoint_path(args),
                        autoencoder,
                        flow,
                        generator,
                        args,
                        dataset,
                        spec,
                        epoch,
                        global_step,
                        best_val_fbd=best_val_fbd,
                    )
            log_wandb(eval_log)

        if not stage1 or flow_reverse_fbd is not None:
            append_metrics_record(
                args,
                dataset,
                spec,
                "eval",
                epoch,
                val_ce=val_ce,
                val_acc=val_acc,
                val_fbd=val_fbd,
                flow_reverse_fbd=flow_reverse_fbd,
            )

        append_metrics_record(
            args,
            dataset,
            spec,
            "train_epoch",
            epoch,
            phase=phase,
            total=epoch_total / steps,
            recon=epoch_recon / steps,
            flow=epoch_flow / steps,
            kl=epoch_kl / steps,
            align=epoch_align / steps,
            stage2_align=epoch_stage2_align / steps,
            gen=epoch_gen / steps,
            acc=epoch_acc / steps,
            val_ce=val_ce,
            val_acc=val_acc,
            val_fbd=val_fbd,
            flow_reverse_fbd=flow_reverse_fbd,
        )
        log_wandb(
            {
                "epoch/phase": 1 if stage1 else 2,
                "epoch/total_loss": epoch_total / steps,
                "epoch/recon_loss": epoch_recon / steps,
                "epoch/flow_loss": epoch_flow / steps,
                "epoch/kl_loss": epoch_kl / steps,
                "epoch/align_loss": epoch_align / steps,
                "epoch/stage2_align_loss": epoch_stage2_align / steps,
                "epoch/gen_loss": epoch_gen / steps,
                "epoch/acc": epoch_acc / steps,
                "epoch/epoch": epoch,
            }
        )

        if not stage1 and (epoch % args.sample_every == 0 or epoch == args.epochs):
            save_generated_examples(generator, valid_loader, dataset, spec, args, device, epoch)

    save_checkpoint(
        default_checkpoint_path(args),
        autoencoder,
        flow,
        generator,
        args,
        dataset,
        spec,
        args.epochs,
        global_step,
        best_val_fbd=None if best_val_fbd == float("inf") else best_val_fbd,
    )


def eval_only(args: argparse.Namespace) -> None:
    device = get_device()
    set_seed(args.seed)
    dataset = get_enhancer_dataset_config(args.dataset)
    spec = compression_spec_from_args(args, dataset)
    _, valid_loader = make_dataloaders(args, dataset)
    autoencoder, flow, generator = make_models(args, dataset, spec, device)
    external_embedder: FrozenEnhancerClassifierEmbedder | None = None
    if external_embeddings_enabled(args):
        external_embedder = FrozenEnhancerClassifierEmbedder(dataset, resolve_external_embed_ckpt(args, dataset), device)
    payload = load_checkpoint(checkpoint_path_for_args(args), autoencoder, flow, generator, args, device, dataset, spec)

    val_ce, val_acc = evaluate_generator_ce(
        autoencoder,
        flow,
        generator,
        valid_loader,
        device,
        external_embedder=external_embedder,
        max_batches=args.max_eval_batches,
    )
    print(f"[eval] val_ce={val_ce:.4f} val_acc={val_acc:.4f}")
    if args.skip_fbd:
        return
    classifier_ckpt = resolve_classifier_ckpt(args, dataset)
    evaluator = EnhancerFBDEvaluator(dataset, classifier_ckpt, device)
    val_fbd = evaluate_slm_style_fbd(generator, evaluator, valid_loader, dataset, spec, args, device)
    flow_reverse_fbd = evaluate_flow_reverse_fbd(autoencoder, flow, evaluator, valid_loader, dataset, spec, args, device)
    print(f"[eval] val_fbd={val_fbd:.4f}")
    print(f"[eval] flow_reverse_fbd={flow_reverse_fbd:.4f}")
    log_wandb(
        {
            "eval/val_ce": val_ce,
            "eval/val_acc": val_acc,
            "eval/val_fbd": val_fbd,
            "eval/flow_reverse_fbd": flow_reverse_fbd,
            "eval/epoch": int(payload.get("epoch", -1)),
        }
    )
    append_metrics_record(
        args,
        dataset,
        spec,
        "eval_only",
        int(payload.get("epoch", -1)),
        val_ce=val_ce,
        val_acc=val_acc,
        val_fbd=val_fbd,
        flow_reverse_fbd=flow_reverse_fbd,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a one-step enhancer transport model.")
    parser.add_argument("--dataset", type=str, choices=("fb", "mel"), default="fb")
    parser.add_argument("--data-dir", type=str, default="./data/dna_data")
    parser.add_argument("--output-dir", type=str, default="./outputs/one_step_enhancer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--compression-ratio", type=str, default="1/4")
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--fixed-std", type=float, default=0.5)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--decoder-width", type=int, default=256)
    parser.add_argument("--generator-width", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--generator-depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--flow-width", type=int, default=128)
    parser.add_argument("--flow-blocks", type=int, default=4)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--flow-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--beta-kl", type=float, default=0.0)
    parser.add_argument("--external-embed-mode", type=str, choices=("off", "align", "encoder_cond"), default="off")
    parser.add_argument("--external-align-weight", type=float, default=0.1)
    parser.add_argument("--stage2-align-mode", type=str, choices=("off", "shallow_cosine"), default="off")
    parser.add_argument("--stage2-align-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--fbd-every", type=int, default=1)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--val-on-test", action="store_true")
    parser.add_argument("--skip-fbd", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="one-step-enhancer")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    dataset = get_enhancer_dataset_config(args.dataset)
    if args.stage1_epochs < 1 or args.stage1_epochs > args.epochs:
        raise ValueError("--stage1-epochs must be in [1, --epochs].")
    if args.flow_width % args.flow_heads != 0:
        raise ValueError("--flow-width must be divisible by --flow-heads.")
    if args.max_train_steps_per_epoch is not None and args.max_train_steps_per_epoch < 1:
        raise ValueError("--max-train-steps-per-epoch must be positive when provided.")
    if args.max_eval_batches is not None and args.max_eval_batches < 1:
        raise ValueError("--max-eval-batches must be positive when provided.")
    if args.fbd_every < 1:
        raise ValueError("--fbd-every must be positive.")
    if args.beta_kl < 0:
        raise ValueError("--beta-kl must be non-negative.")
    if args.external_align_weight < 0:
        raise ValueError("--external-align-weight must be non-negative.")
    if args.stage2_align_weight < 0:
        raise ValueError("--stage2-align-weight must be non-negative.")
    compression_spec_from_args(args, dataset)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    ensure_dir(Path(args.output_dir))
    init_wandb(args)
    try:
        if args.eval_only:
            eval_only(args)
        else:
            train(args)
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
