"""Headless training process entry point; deliberately imports no Qt modules."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(spec_path: str | Path | None = None) -> int:
    if spec_path is None:
        arguments = sys.argv[1:]
        if len(arguments) != 1:
            if sys.stderr:
                print("usage: python -m handwriter.training_worker SPEC.json", file=sys.stderr)
            return 2
        spec_path = arguments[0]
    from .training import run_training

    def emit(event):
        if sys.stdout:
            print(json.dumps(event, ensure_ascii=False), flush=True)

    result = run_training(Path(spec_path), emit=emit)
    return 0 if result.get("status") == "completed" else 3 if result.get("status") == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
