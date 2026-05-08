from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy


def _default_precision(cfg: dict) -> str:
    lightning_cfg = cfg.get("lightning", {})
    if "precision" in lightning_cfg:
        return lightning_cfg["precision"]
    mixed_precision = bool(cfg.get("training", {}).get("mixed_precision", False))
    if mixed_precision:
        return "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"
    return "32-true"


def _default_accelerator(lightning_cfg: dict) -> str:
    configured = lightning_cfg.get("accelerator")
    if configured is not None:
        return configured
    return "gpu" if torch.cuda.is_available() else "cpu"


def _default_devices(default_devices: int | None = None) -> int:
    if default_devices is not None:
        return default_devices
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def _build_strategy(
    strategy_name: str | None,
    *,
    devices: int,
) -> str | DDPStrategy:
    if strategy_name not in (None, "ddp"):
        return strategy_name
    if devices > 1:
        return DDPStrategy(find_unused_parameters=False)
    return strategy_name


def build_trainer_kwargs(
    cfg: dict,
    *,
    max_steps: int | None = None,
    default_devices: int | None = None,
    callbacks: list[Callback] | None = None,
    logger: Any = None,
    use_distributed_sampler: bool = True,
) -> dict[str, Any]:
    lightning_cfg = cfg.get("lightning", {})
    accelerator = _default_accelerator(lightning_cfg)
    devices = int(lightning_cfg.get("devices", _default_devices(default_devices)))
    kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": devices,
        "num_nodes": int(lightning_cfg.get("num_nodes", 1)),
        "strategy": _build_strategy(
            lightning_cfg.get("strategy"),
            devices=devices,
        ),
        "precision": _default_precision(cfg),
        "logger": logger,
        "callbacks": callbacks or [],
        "enable_checkpointing": False,
        "log_every_n_steps": int(lightning_cfg.get("log_every_n_steps", 1)),
        "gradient_clip_val": float(cfg.get("training", {}).get("grad_clip", 0.0)),
        "num_sanity_val_steps": 0,
        "use_distributed_sampler": use_distributed_sampler,
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return kwargs


def build_wandb_logger(
    cfg: dict,
    save_dir: str | os.PathLike[str],
    *,
    run_name: str | None = None,
) -> WandbLogger:
    wb = cfg.get("wandb", {})
    return WandbLogger(
        project=wb.get("project", "latent-transport-lm"),
        name=run_name if run_name is not None else wb.get("run_name", None),
        entity=wb.get("entity", None),
        tags=wb.get("tags", []),
        save_dir=str(save_dir),
        log_model=False,
    )


def save_legacy_checkpoint(path: str | Path, payload: dict) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)


def load_legacy_checkpoint_state(path: str | Path) -> dict | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def remaining_steps(max_steps: int, start_step: int) -> int:
    return max(0, max_steps - start_step)


def resolve_fit_checkpoint_path(
    path: str | Path,
    *,
    required_datamodule_key: str | None = None,
) -> str | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if required_datamodule_key is not None and required_datamodule_key not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing {required_datamodule_key} and is not safe to auto-resume"
        )
    return str(checkpoint_path)


class TimeBudgetCallback(Callback):
    def __init__(self, budget_seconds: int):
        super().__init__()
        self.budget_seconds = int(budget_seconds)
        self.started_at: float | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.started_at = time.time()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.started_at is None:
            return
        if time.time() - self.started_at >= self.budget_seconds:
            trainer.should_stop = True


class LegacyCheckpointCallback(Callback):
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        every_n_train_steps: int,
        payload_fn: Callable[[pl.Trainer, pl.LightningModule], dict],
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.every_n_train_steps = int(every_n_train_steps)
        self.payload_fn = payload_fn

    def _save(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        save_legacy_checkpoint(self.checkpoint_path, self.payload_fn(trainer, pl_module))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(getattr(pl_module, "legacy_global_step", trainer.global_step))
        if step > 0 and step % self.every_n_train_steps == 0:
            self._save(trainer, pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._save(trainer, pl_module)


class FixedPathCheckpointCallback(Callback):
    def __init__(self, checkpoint_path: str | Path, *, every_n_train_steps: int) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.every_n_train_steps = int(every_n_train_steps)

    def _save(self, trainer: pl.Trainer) -> None:
        if not trainer.is_global_zero:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path))

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step > 0 and trainer.global_step % self.every_n_train_steps == 0:
            self._save(trainer)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._save(trainer)
