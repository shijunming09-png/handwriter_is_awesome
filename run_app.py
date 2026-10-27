"""The same entry point is used by source launches and the portable executable."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    if "--training-worker" in sys.argv:
        parser = argparse.ArgumentParser(description="Handwriter isolated local training worker")
        parser.add_argument("--training-worker", type=Path, required=True)
        args = parser.parse_args()
        from handwriter.training_worker import main as worker_main
        return worker_main(args.training_worker)
    if "--smoke-test" in sys.argv:
        parser = argparse.ArgumentParser(description="Offline end-to-end application verification")
        parser.add_argument("--smoke-test", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--models-dir", type=Path)
        parser.add_argument("--recognizer-dir", type=Path)
        args = parser.parse_args()
        from handwriter.smoketest import run_smoke_test
        return run_smoke_test(args.smoke_test, args.output_dir, args.models_dir, args.recognizer_dir)
    from handwriter.__main__ import main as gui_main
    return gui_main()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
