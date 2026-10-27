from __future__ import annotations

import os
from pathlib import Path
import sys


def app_home() -> Path:
    override = os.environ.get("HANDWRITER_HOME")
    if override:
        root = Path(override).resolve()
    elif getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent.parent
    return root


def _directory(name: str) -> Path:
    target = app_home() / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def models_dir() -> Path:
    return _directory("models")


def projects_dir() -> Path:
    return _directory("projects")


def logs_dir() -> Path:
    return _directory("logs")
