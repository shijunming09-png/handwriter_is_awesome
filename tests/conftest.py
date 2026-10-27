from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("qa_fixture_generator", SOURCE_ROOT / "tools" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path_factory.mktemp("pdf_fixtures")
    module.generate_fixtures(output)
    return output
