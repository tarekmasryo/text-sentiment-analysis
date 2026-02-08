from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def project_root(start: Optional[Path] = None) -> Path:
    """Best-effort repo root detection (works in notebooks + scripts)."""
    p = (start or Path.cwd()).resolve()
    markers = {"pyproject.toml", "requirements.txt", "README.md", ".git"}
    for _ in range(10):
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    return (start or Path.cwd()).resolve()


def resolve_data_path(
    filename: str,
    *,
    local_subdir: str = "data/raw",
    kaggle_subdir_hint: Optional[str] = None,
    extra_candidates: Optional[Iterable[str]] = None,
) -> Path:
    """Resolve a dataset file path with a simple priority:

    1) Explicit env var `DATA_PATH` (full file path)
    2) Local repo path: <root>/<local_subdir>/<filename>
    3) Kaggle input path fallback: /kaggle/input/**/<filename>
       - If kaggle_subdir_hint is provided, try /kaggle/input/<hint>/<filename> first.
    4) Any extra candidate paths
    """
    env_path = os.environ.get("DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    root = project_root()
    local = (root / local_subdir / filename).resolve()
    if local.exists():
        return local

    if extra_candidates:
        for c in extra_candidates:
            p = Path(c).expanduser()
            p = (root / p).resolve() if not p.is_absolute() else p.resolve()
            if p.exists():
                return p

    if _is_kaggle():
        base = Path("/kaggle/input")
        if kaggle_subdir_hint:
            hinted = base / kaggle_subdir_hint / filename
            if hinted.exists():
                return hinted

        # Light search (avoid heavy rglob on some envs)
        for pat in ("*/*", "*/*/*"):
            for p in base.glob(pat):
                if p.is_file() and p.name == filename:
                    return p.resolve()

    raise FileNotFoundError(
        f"Could not find '{filename}'.\n"
        f"Tried: DATA_PATH env var, {local_subdir}/, Kaggle /kaggle/input/.\n"
        f"Tip: put the file under 'data/raw/' or set DATA_PATH."
    )
