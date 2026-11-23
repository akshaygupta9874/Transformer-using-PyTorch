"""Run metadata, deterministic seeding, and optional W&B logging."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunMetadata:
    name: str
    seed: int
    created_at: str
    git_commit: str | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def write_run_manifest(directory: str | Path, name: str, config: dict, seed: int) -> Path:
    """Persist the full resolved configuration beside checkpoints and metrics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = RunMetadata(name=name, seed=seed, created_at=datetime.now(timezone.utc).isoformat())
    path = directory / "run_manifest.json"
    path.write_text(json.dumps({"metadata": asdict(metadata), "config": config}, indent=2), encoding="utf-8")
    return path


class OptionalWandb:
    """No-op tracker unless W&B is installed and explicitly enabled."""

    def __init__(self, project: str, config: dict, enabled: bool = False):
        self.run = None
        if enabled:
            try:
                import wandb

                self.run = wandb.init(project=project, config=config)
            except ImportError as error:
                raise RuntimeError("Install wandb to enable remote experiment tracking.") from error

    def log(self, values: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
