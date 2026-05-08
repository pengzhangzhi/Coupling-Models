"""Data preparation — single entry point for all dataset/tokenizer combinations.

Canonical data layout:
  $LTLM_DATA_ROOT/lm1b/
      train_qwen.npy, test_qwen.npy   (Qwen tokenizer, int32, shape (N, 128))
      train_bert.dat/, test_bert.dat/ (BERT tokenizer, Arrow, for MDLM)
  $LTLM_DATA_ROOT/owt/
      train_qwen.npy, test_qwen.npy
      train_bert.dat/, test_bert.dat/

Usage:
  uv run python prepare.py --dataset lm1b --tokenizer qwen   # → data/lm1b/{train,test}_qwen.npy
  uv run python prepare.py --dataset owt  --tokenizer qwen   # → data/owt/{train,test}_qwen.npy
  uv run python prepare.py --dataset lm1b --tokenizer bert   # → data/lm1b/{train,test}_bert.dat
  uv run python prepare.py --dataset owt  --tokenizer bert   # → data/owt/{train,test}_bert.dat

Constants exported (used by training scripts):
  QWEN_MODEL, QWEN_VOCAB_SIZE, QWEN_HIDDEN, MAX_SEQ_LEN, REPR_DIM,
  PAD_TOKEN_ID

DataLoaders (Qwen tokenizer only — used by Stage A, Stage B, AR teacher):
  get_dataloaders(batch_size, dataset="lm1b") -> (train_loader, val_loader)
"""
import argparse
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from runtime_paths import configure_process_environment, dataset_dir, tmp_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QWEN_MODEL      = "Qwen/Qwen2.5-0.5B"
QWEN_VOCAB_SIZE = 151_936
QWEN_HIDDEN     = 896
MAX_SEQ_LEN     = 128
REPR_DIM        = 256
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
def get_dataloaders(batch_size: int, dataset: str = "lm1b", num_workers: int = 4):
    """Return (train_loader, val_loader) for Qwen-tokenized data.

    Args:
        dataset: "lm1b" (default) or "owt".
    """
    if dataset == "lm1b":
        train_ids = _qwen_lm1b("train")
        val_ids   = _qwen_lm1b("test")
    elif dataset == "owt":
        train_path = _OWT_DIR / "train_qwen.npy"
        test_path  = _OWT_DIR / "test_qwen.npy"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"OWT data not found at {_OWT_DIR}. "
                "Run: uv run python prepare.py --dataset owt --tokenizer qwen"
            )
        train_ids = np.load(str(train_path), mmap_mode="r")
        val_ids   = np.load(str(test_path),  mmap_mode="r")
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'lm1b' or 'owt'.")

    train_loader = DataLoader(
        TokenizedDataset(train_ids), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        TokenizedDataset(val_ids), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


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
