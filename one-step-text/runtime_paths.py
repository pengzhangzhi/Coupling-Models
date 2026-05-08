import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent


def repo_root() -> Path:
    return _REPO_ROOT


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _require_runtime_root() -> Path:
    explicit_root = os.getenv("LTLM_RUNTIME_ROOT")
    if explicit_root:
        return Path(explicit_root).expanduser()
    return Path.home() / ".local" / "share" / "latent-transport-lm"


def runtime_root() -> Path:
    return _require_runtime_root()


def data_root() -> Path:
    return _path_from_env("LTLM_DATA_ROOT", runtime_root() / "data")


def checkpoint_root() -> Path:
    return _path_from_env("LTLM_CHECKPOINT_ROOT", runtime_root() / "checkpoints")


def wandb_root() -> Path:
    return _path_from_env("LTLM_WANDB_DIR", runtime_root() / "wandb")


def tmp_root() -> Path:
    return _path_from_env("LTLM_TMPDIR", runtime_root() / "tmp")


def uv_project_environment() -> Path:
    return _path_from_env("UV_PROJECT_ENVIRONMENT", runtime_root() / ".venv")


def dataset_dir(dataset: str) -> Path:
    return data_root() / dataset


def ensure_runtime_dirs() -> None:
    for path in (
        runtime_root(),
        data_root(),
        checkpoint_root(),
        wandb_root(),
        tmp_root(),
        uv_project_environment().parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_process_environment() -> None:
    os.environ.setdefault("TMPDIR", str(tmp_root()))
    os.environ.setdefault("TEMP", str(tmp_root()))
    os.environ.setdefault("TMP", str(tmp_root()))


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "checkpoints":
        return checkpoint_root().joinpath(*candidate.parts[1:])
    return checkpoint_root() / candidate
