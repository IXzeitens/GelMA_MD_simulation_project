"""Entry point for the gelatin MD pipeline (Workflow 1: build + simulate).

Run from a per-subsystem working directory that contains `config.json`,
`input/<name>.pdb`, and `packmol/`. The `pipeline` package is imported from
the gelma_md package root (added to sys.path below), so this script can live
under `workflow/` while the core code lives under `pipeline/`.

    cd <subsystem_dir>          # has config.json, input/, packmol/, Output/
    python /path/to/gelma_md/workflow/main.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the gelma_md package root importable regardless of CWD, so that
# `pipeline/` (sibling of this workflow/ dir) resolves.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from pipeline.paths import ProjectPaths   # noqa: E402
from pipeline.pipeline import run          # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # The subsystem working directory is the current directory at run time.
    run(ProjectPaths(Path.cwd()))


if __name__ == "__main__":
    main()
