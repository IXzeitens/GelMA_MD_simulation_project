"""Project directory layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def packmol_dir(self) -> Path:
        return self.root / "packmol"

    @property
    def script_dir(self) -> Path:
        return self.root / "script"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def output_dir(self) -> Path:
        return self.root / "Output"

    def ensure_working_dirs(self) -> None:
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
