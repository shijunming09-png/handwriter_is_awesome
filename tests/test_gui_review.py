"""Offscreen GUI regression tests; never operates another desktop application."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from handwriter.data import LineResult
from handwriter.gui import MainWindow
from handwriter.storage import create_project, load_project


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def review_window(qt_app, fixture_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("HANDWRITER_HOME", str(tmp_path / "app_data"))
    project = create_project(str(fixture_dir / "single_page.pdf"), tmp_path / "projects")
    project.pages[0].status = "recognized"
    project.pages[0].lines = [
        LineResult(id="top", polygon=[[100, 100], [900, 100], [900, 180], [100, 180]], raw_text="First original line", status="recognized"),
        LineResult(id="bottom", polygon=[[100, 220], [900, 220], [900, 300], [100, 300]], raw_text="Second original line", status="recognized"),
    ]
    window = MainWindow()
    window._set_project(project)
    monkeypatch.setattr(window, "_confirm_discard", lambda *args, **kwargs: True)
    yield window
    window._save_timer.stop()
    window.close()
    qt_app.processEvents()


def test_box_change_clears_old_text_and_manual_transcription_persists(review_window):
    window = review_window
    page = window.project.pages[0]
    line = page.lines[0]
    line.edited_text = "Previous correction"
    updated = [[105, 105], [920, 105], [920, 185], [105, 185]]
    window._change_line_box(line.id, updated)
    assert line.polygon == updated
    assert line.status == "stale"
    assert line.raw_text == "" and line.edited_text is None
    assert page.status == "detected"
    window.line_editors[line.id].editor.setPlainText("New manual transcription")
    assert line.text == "New manual transcription"
    assert line.status == "recognized"
    window._save_project(quiet=True)
    restored = load_project(Path(window.project.work_dir))
    assert restored.pages[0].lines[0].text == "New manual transcription"


def test_deletion_preserves_completed_state_and_allows_intentionally_empty_page(review_window):
    window = review_window
    page = window.project.pages[0]
    window._selected_ids = ["top"]
    window._delete_lines()
    assert [line.id for line in page.lines] == ["bottom"]
    assert page.status == "recognized"
    window._selected_ids = ["bottom"]
    window._delete_lines()
    assert page.lines == []
    assert page.status == "recognized"


def test_split_and_merge_create_stale_rows_without_reusing_old_ocr(review_window):
    window = review_window
    page = window.project.pages[0]
    window._selected_ids = ["top"]
    window._split_line()
    assert len(page.lines) == 3
    newly_split = [line for line in page.lines if line.id != "bottom"]
    assert all(line.status == "stale" and not line.text for line in newly_split)
    assert len({line.id for line in page.lines}) == 3
    window._selected_ids = [line.id for line in newly_split]
    window._merge_lines()
    assert len(page.lines) == 2
    merged = next(line for line in page.lines if line.id != "bottom")
    assert merged.status == "stale" and merged.text == ""
    assert page.lines[0] is merged


def test_ui_busy_mode_locks_text_and_box_edits(review_window):
    window = review_window
    window._set_busy(True)
    assert not window.canvas.editable
    assert all(card.editor.isReadOnly() for card in window.line_editors.values())
    assert not window.import_button.isEnabled()
    assert not window.word_button.isEnabled()
    assert window.cancel_button.isEnabled()
    window._set_busy(False)
    assert window.canvas.editable
    assert all(not card.editor.isReadOnly() for card in window.line_editors.values())
