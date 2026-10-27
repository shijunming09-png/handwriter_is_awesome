"""Test task recovery/control flow with a deterministic local engine substitute.

These tests validate saving and cancellation, not OCR recognition quality.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from handwriter.data import LineResult
from handwriter.gui import TaskWorker
from handwriter.storage import create_project, load_project
import handwriter.engine as engine_module
import handwriter.model_manager as manager_module


class LocalEngine:
    def __init__(self, *args, progress=None, cancelled=None, **kwargs):
        self.progress = progress or (lambda _: None)
        self.cancelled = cancelled or (lambda: False)
        self.detected_count = 0
        self.recognized_count = 0

    def load(self):
        return "test-local-engine"

    def detect(self, image):
        self.detected_count += 1
        return [LineResult(id=f"new-{self.detected_count}", polygon=[[10, 10], [500, 10], [500, 80], [10, 80]])]

    def recognize(self, image, lines):
        self.recognized_count += len(lines)
        for line in lines:
            line.raw_text = "New locally recognized text"
            line.status = "recognized"


def _setup_engine(monkeypatch, tmp_path, factory):
    monkeypatch.setenv("HANDWRITER_HOME", str(tmp_path / "app_data"))
    monkeypatch.setattr(manager_module, "models_ready", lambda _: True)
    monkeypatch.setattr(engine_module, "OcrEngine", factory)


def test_continue_skips_completed_pages_and_preserves_corrections(fixture_dir, tmp_path, monkeypatch):
    project = create_project(str(fixture_dir / "two_pages.pdf"), tmp_path / "projects")
    project.pages[0].status = "recognized"
    project.pages[0].lines = [LineResult(id="corrected", polygon=[[10, 10], [500, 10], [500, 80], [10, 80]], raw_text="Old OCR", edited_text="Human correction", status="recognized")]
    instances = []

    def make_engine(*args, **kwargs):
        instance = LocalEngine(*args, **kwargs)
        instances.append(instance)
        return instance

    _setup_engine(monkeypatch, tmp_path, make_engine)
    worker = TaskWorker(project)
    worker.run()
    restored = load_project(Path(project.work_dir))
    assert instances[0].detected_count == 1
    assert instances[0].recognized_count == 1
    assert restored.pages[0].lines[0].text == "Human correction"
    assert restored.pages[0].lines[0].raw_text == "Old OCR"
    assert restored.pages[1].lines[0].raw_text == "New locally recognized text"
    assert restored.pages[1].status == "recognized"


def test_cancel_after_one_page_saves_it_and_leaves_next_page_pending(fixture_dir, tmp_path, monkeypatch):
    project = create_project(str(fixture_dir / "two_pages.pdf"), tmp_path / "projects")
    worker = TaskWorker(project)

    class CancellingEngine(LocalEngine):
        def recognize(self, image, lines):
            super().recognize(image, lines)
            worker.cancel()

    _setup_engine(monkeypatch, tmp_path, CancellingEngine)
    completed = []
    worker.finished.connect(completed.append)
    worker.run()
    restored = load_project(Path(project.work_dir))
    assert restored.pages[0].status == "recognized"
    assert restored.pages[0].lines[0].raw_text == "New locally recognized text"
    assert restored.pages[1].status == "pending"
    assert restored.pages[1].lines == []
    assert completed and "取消" in completed[-1]


def test_bad_page_does_not_drop_other_completed_results(fixture_dir, tmp_path, monkeypatch):
    project = create_project(str(fixture_dir / "two_pages.pdf"), tmp_path / "projects")

    class OneBadPageEngine(LocalEngine):
        def detect(self, image):
            self.detected_count += 1
            if self.detected_count == 1:
                raise RuntimeError("simulated page failure")
            return [LineResult(id="surviving", polygon=[[10, 10], [500, 10], [500, 80], [10, 80]])]

    _setup_engine(monkeypatch, tmp_path, OneBadPageEngine)
    worker = TaskWorker(project)
    worker.run()
    restored = load_project(Path(project.work_dir))
    assert restored.pages[0].status == "error"
    assert "simulated page failure" in restored.pages[0].error
    assert restored.pages[1].status == "recognized"
    assert restored.pages[1].lines[0].raw_text == "New locally recognized text"
