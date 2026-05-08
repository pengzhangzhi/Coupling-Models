"""Data preparation — single entry point for all dataset/tokenizer combinations.

Canonical data layout:
  $LTLM_DATA_ROOT/lm1b/
      train_qwen.npy, test_qwen.npy   (Qwen tokenizer, int32, shape (N, 128))
      train_bert.dat/, test_bert.dat/ (BERT tokenizer, Arrow, for MDLM)
  $LTLM_DATA_ROOT/owt/
      train_qwen.npy, test_qwen.npy
      train_bert.dat/, test_bert.dat/

Usage:
  scripts/run.sh python prepare.py --dataset lm1b --tokenizer qwen   # -> $LTLM_DATA_ROOT/lm1b/{train,test}_qwen.npy
  scripts/run.sh python prepare.py --dataset owt  --tokenizer qwen   # -> $LTLM_DATA_ROOT/owt/{train,test}_qwen.npy
  scripts/run.sh python prepare.py --dataset lm1b --tokenizer bert   # -> $LTLM_DATA_ROOT/lm1b/{train,test}_bert.dat
  scripts/run.sh python prepare.py --dataset owt  --tokenizer bert   # -> $LTLM_DATA_ROOT/owt/{train,test}_bert.dat

Constants exported (used by training scripts):
  QWEN_MODEL, QWEN_VOCAB_SIZE, QWEN_HIDDEN, MAX_SEQ_LEN, REPR_DIM,
  PAD_TOKEN_ID, TIME_BUDGET

DataLoaders (Qwen tokenizer only — used by Stage A, Stage B, AR teacher):
  get_dataloaders(batch_size, dataset="lm1b") -> (train_loader, val_loader)
"""
import argparse
import os
import numpy as np
import torch
from pathlib import Path
from lightning import pytorch as pl
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from runtime_paths import configure_process_environment, dataset_dir, tmp_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QWEN_MODEL      = "Qwen/Qwen2.5-0.5B"
QWEN_VOCAB_SIZE = 151_936
QWEN_HIDDEN     = 896
MAX_SEQ_LEN     = 128
# Compatibility alias for the current full-dimensional latent experiment.
REPR_DIM        = QWEN_HIDDEN
TIME_BUDGET     = int(os.getenv("TIME_BUDGET", "1800"))
PAD_TOKEN_ID    = 151_643  # Qwen2.5-0.5B eos_token_id (used as pad)

configure_process_environment()

# Canonical data directories
_LM1B_DIR = dataset_dir("lm1b")
_OWT_DIR  = dataset_dir("owt")


# ---------------------------------------------------------------------------
# Dataset class (Qwen tokenizer — npy format)
# ---------------------------------------------------------------------------
class TokenizedDataset(Dataset):
    """Memory-mapped numpy array → (token_ids, padding_mask)."""

    def __init__(self, token_ids: np.ndarray):
        self.token_ids = token_ids  # (N, T) int32

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        ids = torch.from_numpy(self.token_ids[idx].astype(np.int64))
        return ids, (ids != PAD_TOKEN_ID)


# ---------------------------------------------------------------------------
# Qwen tokenizer — LM1B
# ---------------------------------------------------------------------------
def _qwen_lm1b(split: str) -> np.ndarray:
    cache = _LM1B_DIR / f"{split}_qwen.npy"
    if cache.exists():
        return np.load(str(cache), mmap_mode="r")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[prepare] Tokenizing LM1B {split} with Qwen tokenizer...")
    _LM1B_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("dvruette/lm1b", split=split, revision="~parquet",
                      cache_dir=str(_LM1B_DIR / "raw"))
    n = len(ds)
    tmp = str(cache) + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int32, shape=(n, MAX_SEQ_LEN))
    for i, ex in enumerate(ds):
        arr[i] = tokenizer.encode(ex["text"], max_length=MAX_SEQ_LEN,
                                  padding="max_length", truncation=True)
        if (i + 1) % 500_000 == 0:
            print(f"  {i+1:,}/{n:,}")
    del arr
    os.rename(tmp, str(cache))
    print(f"[prepare] Saved {cache}")
    return np.load(str(cache), mmap_mode="r")


# ---------------------------------------------------------------------------
# Qwen tokenizer — OWT
# ---------------------------------------------------------------------------
def _qwen_owt() -> None:
    train_path = _OWT_DIR / "train_qwen.npy"
    test_path  = _OWT_DIR / "test_qwen.npy"

    if train_path.exists() and test_path.exists():
        t = np.load(str(train_path), mmap_mode="r")
        v = np.load(str(test_path),  mmap_mode="r")
        print(f"[prepare] OWT already cached — train {t.shape}, test {v.shape}")
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    _OWT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("[prepare] Loading OpenWebText...")
    ds = load_dataset("openwebtext", split="train")
    n = len(ds)
    test_size  = 100_000
    train_size = n - test_size
    print(f"[prepare] OWT: {n:,} total, train {train_size:,}, test {test_size:,}")

    for split_name, start, end, out_path in [
        ("train", 0,          train_size, train_path),
        ("test",  train_size, n,          test_path),
    ]:
        if out_path.exists():
            continue
        count = end - start
        print(f"[prepare] Tokenizing OWT {split_name} ({count:,})...")
        tmp = str(out_path) + ".tmp"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int32, shape=(count, MAX_SEQ_LEN))
        for i in range(count):
            arr[i] = tokenizer.encode(ds[start + i]["text"], max_length=MAX_SEQ_LEN,
                                      padding="max_length", truncation=True)
            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,}/{count:,}")
        del arr
        os.rename(tmp, str(out_path))
        print(f"[prepare] Saved {out_path}")


# ---------------------------------------------------------------------------
# BERT tokenizer — LM1B or OWT (Arrow format for MDLM)
# Requires: mdlm conda env (baselines/mdlm/dataloader.py + omegaconf)
# Output: {lm1b,owt}/train_bert.dat and {lm1b,owt}/test_bert.dat
# ---------------------------------------------------------------------------
def _bert(dataset: str, seq_len: int = 1024, num_proc: int = 8) -> None:
    import shutil, sys, tempfile
    sys.path.insert(0, str(Path(__file__).parent / "baselines" / "mdlm"))
    import omegaconf
    import dataloader as mdlm_dl

    data_dir = _LM1B_DIR if dataset == "lm1b" else _OWT_DIR
    train_out = data_dir / "train_bert.dat"
    test_out  = data_dir / "test_bert.dat"

    if train_out.exists() and test_out.exists():
        print(f"[prepare] BERT {dataset}: already cached at {data_dir}")
        return

    # Use a temp dir for MDLM's intermediate HF download cache so it doesn't
    # pollute the canonical data dir.
    tmp_cache = Path(tempfile.mkdtemp(prefix="mdlm_prep_", dir=str(tmp_root())))
    print(f"[prepare] BERT {dataset}: temp cache → {tmp_cache}")

    try:
        cfg = omegaconf.OmegaConf.create({
            "data": {
                "train": "lm1b" if dataset == "lm1b" else "openwebtext-train",
                "valid": "lm1b" if dataset == "lm1b" else "openwebtext-valid",
                "tokenizer_name_or_path": "bert-base-uncased",
                "cache_dir": str(tmp_cache),
                "wrap": False,
                "streaming": False,
            },
            "model": {"length": seq_len},
            "loader": {
                "global_batch_size": 512, "batch_size": 512,
                "eval_global_batch_size": 512, "eval_batch_size": 512,
                "num_workers": num_proc, "pin_memory": False,
            },
            "trainer": {"devices": 1, "num_nodes": 1, "accumulate_grad_batches": 1},
        })

        tokenizer = mdlm_dl.get_tokenizer(cfg)
        print(f"[prepare] BERT tokenizer vocab={tokenizer.vocab_size}")

        if dataset == "lm1b":
            splits = [("lm1b", "train", train_out), ("lm1b", "test", test_out)]
        else:
            splits = [
                ("openwebtext-train", "train", train_out),
                ("openwebtext-valid", "train", test_out),
            ]

        for ds_name, mode, out_path in splits:
            if out_path.exists():
                print(f"[prepare] {out_path.name} already exists, skipping")
                continue
            print(f"[prepare] BERT {dataset}/{ds_name} split='{mode}'...", flush=True)
            # MDLM generates e.g. lm1b_train_bs1024_unwrapped.dat inside tmp_cache
            mdlm_dl.get_dataset(
                ds_name, tokenizer, mode=mode, wrap=False,
                cache_dir=str(tmp_cache), block_size=seq_len, num_proc=num_proc,
            )
            # Find the generated .dat dir and move it to the canonical name
            generated = next(tmp_cache.glob("*.dat"))
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated), str(out_path))
            print(f"[prepare] Saved {out_path}")
    finally:
        shutil.rmtree(tmp_cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# DataLoaders (Qwen only — used by all training scripts)
# ---------------------------------------------------------------------------
def _resolve_qwen_datasets(dataset: str = "lm1b") -> tuple[TokenizedDataset, TokenizedDataset]:
    if dataset == "lm1b":
        train_ids = _qwen_lm1b("train")
        val_ids   = _qwen_lm1b("test")
    elif dataset == "owt":
        train_path = _OWT_DIR / "train_qwen.npy"
        test_path  = _OWT_DIR / "test_qwen.npy"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"OWT data not found at {_OWT_DIR}. "
                "Run: scripts/run.sh python prepare.py --dataset owt --tokenizer qwen"
            )
        train_ids = np.load(str(train_path), mmap_mode="r")
        val_ids   = np.load(str(test_path),  mmap_mode="r")
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'lm1b' or 'owt'.")

    return TokenizedDataset(train_ids), TokenizedDataset(val_ids)


def build_dataloaders_for_training(
    batch_size: int,
    dataset: str = "lm1b",
    num_workers: int = 4,
    *,
    distributed: bool = False,
    world_size: int | None = None,
    rank: int | None = None,
    train_subset_fraction: float = 1.0,
    train_subset_seed: int = 0,
):
    train_dataset, val_dataset = _resolve_qwen_datasets(dataset)
    if not 0 < train_subset_fraction <= 1:
        raise ValueError("train_subset_fraction must be in the interval (0, 1]")
    if train_subset_fraction < 1.0:
        rng = np.random.default_rng(train_subset_seed)
        subset_size = max(1, int(round(len(train_dataset) * train_subset_fraction)))
        subset_indices = np.sort(
            rng.choice(len(train_dataset), size=subset_size, replace=False)
        ).tolist()
        train_dataset = Subset(train_dataset, subset_indices)
    train_sampler = None
    val_sampler = None
    train_shuffle = True
    if distributed:
        if world_size is None or rank is None:
            raise ValueError("world_size and rank must be provided when distributed=True")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=train_shuffle, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


class LTLMTrainingDataModule(pl.LightningDataModule):
    """Shared training datamodule with resumable train-loader state."""

    def __init__(
        self,
        *,
        batch_size: int,
        dataset: str = "lm1b",
        num_workers: int = 4,
        train_subset_fraction: float = 1.0,
        train_subset_seed: int = 0,
        train_shuffle_seed: int = 0,
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.dataset = dataset
        self.num_workers = int(num_workers)
        self.train_subset_fraction = float(train_subset_fraction)
        self.train_subset_seed = int(train_subset_seed)
        self.train_shuffle_seed = int(train_shuffle_seed)
        self._configured_world_size = None if world_size is None else int(world_size)
        self._configured_rank = None if rank is None else int(rank)
        self._train_dataset = None
        self._val_dataset = None
        self._train_loader: StatefulDataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._pending_train_loader_state: dict | None = None
        self._resume_metadata: dict | None = None

    def _distributed_context(self) -> tuple[int, int]:
        if self._configured_world_size is not None and self._configured_rank is not None:
            return self._configured_world_size, self._configured_rank
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1, 0
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        global_rank = int(getattr(trainer, "global_rank", 0) or 0)
        return world_size, global_rank

    def _resume_metadata_payload(self) -> dict[str, int | float | str]:
        world_size, _ = self._distributed_context()
        return {
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "train_subset_fraction": self.train_subset_fraction,
            "train_subset_seed": self.train_subset_seed,
            "train_shuffle_seed": self.train_shuffle_seed,
            "world_size": world_size,
        }

    def _validate_resume_metadata(self, metadata: dict) -> None:
        current = self._resume_metadata_payload()
        for key, value in current.items():
            if metadata.get(key) != value:
                raise ValueError(
                    "Resumable training state mismatch for "
                    f"{key}: checkpoint={metadata.get(key)!r}, current={value!r}"
                )

    def prepare_data(self) -> None:
        _resolve_qwen_datasets(self.dataset)

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        train_dataset, val_dataset = _resolve_qwen_datasets(self.dataset)
        if not 0 < self.train_subset_fraction <= 1:
            raise ValueError("train_subset_fraction must be in the interval (0, 1]")
        if self.train_subset_fraction < 1.0:
            rng = np.random.default_rng(self.train_subset_seed)
            subset_size = max(1, int(round(len(train_dataset) * self.train_subset_fraction)))
            subset_indices = np.sort(
                rng.choice(len(train_dataset), size=subset_size, replace=False)
            ).tolist()
            train_dataset = Subset(train_dataset, subset_indices)
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

    def train_dataloader(self) -> StatefulDataLoader:
        if self._train_dataset is None:
            self.setup("fit")
        world_size, rank = self._distributed_context()
        train_sampler = None
        train_shuffle = True
        generator = torch.Generator()
        generator.manual_seed(self.train_shuffle_seed)
        if world_size > 1:
            train_sampler = DistributedSampler(
                self._train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=self.train_shuffle_seed,
                drop_last=True,
            )
            train_shuffle = False
            generator = None
        self._train_loader = StatefulDataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            generator=generator,
        )
        if self._pending_train_loader_state is not None:
            self._train_loader.load_state_dict(self._pending_train_loader_state)
            self._pending_train_loader_state = None
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup("validate")
        world_size, rank = self._distributed_context()
        val_sampler = None
        if world_size > 1:
            val_sampler = DistributedSampler(
                self._val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        self._val_loader = DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return self._val_loader

    def state_dict(self) -> dict:
        if self._train_loader is None:
            raise RuntimeError("train_dataloader() must be created before saving datamodule state")
        return {
            "resume_metadata": self._resume_metadata_payload(),
            "train_loader_state": self._train_loader.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        metadata = state_dict.get("resume_metadata")
        train_loader_state = state_dict.get("train_loader_state")
        if metadata is None or train_loader_state is None:
            raise ValueError("Checkpoint is missing resumable datamodule state")
        self._validate_resume_metadata(metadata)
        self._resume_metadata = dict(metadata)
        self._pending_train_loader_state = train_loader_state


def get_dataloaders(batch_size: int, dataset: str = "lm1b", num_workers: int = 4):
    """Return (train_loader, val_loader) for Qwen-tokenized data.

    Args:
        dataset: "lm1b" (default) or "owt".
    """
    return build_dataloaders_for_training(
        batch_size=batch_size,
        dataset=dataset,
        num_workers=num_workers,
        distributed=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["lm1b", "owt"],  default="lm1b")
    parser.add_argument("--tokenizer", choices=["qwen", "bert"], default="qwen")
    parser.add_argument("--seq_len",   type=int, default=1024,
                        help="Sequence length for BERT/MDLM (default 1024)")
    parser.add_argument("--num_proc",  type=int,
                        default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    args = parser.parse_args()

    print(f"[prepare] dataset={args.dataset}  tokenizer={args.tokenizer}")

    if args.tokenizer == "qwen":
        if args.dataset == "lm1b":
            for split in ("train", "test"):
                arr = _qwen_lm1b(split)
                print(f"[prepare] lm1b/{split}: {arr.shape}")
        else:
            _qwen_owt()
    else:
        _bert(args.dataset, seq_len=args.seq_len, num_proc=args.num_proc)

    print("[prepare] Done.")
