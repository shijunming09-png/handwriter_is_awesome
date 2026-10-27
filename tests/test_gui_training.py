"""White-box Qt tests for human confirmation and process-isolated training."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QProcess, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from handwriter import learning, paths
from handwriter.data import LineResult
from handwriter.gui import MainWindow
from handwriter.storage import create_project, load_project
from handwriter.training_ui import TrainingDialog
import handwriter.training_ui as training_ui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def reviewed_window(app, fixture_dir, tmp_path, monkeypatch):
    home = tmp_path / "app"
    monkeypatch.setenv("HANDWRITER_HOME", str(home))
    writer = learning.save_writer(home, "同一位作者")
    project = create_project(fixture_dir / "single_page.pdf", home / "projects")
    page = project.pages[0]
    page.status = "recognized"
    page.lines = [
        LineResult(id="first", polygon=[[100, 100], [900, 100], [900, 200], [100, 200]], raw_text="Correct unchanged transcription.", status="recognized"),
        LineResult(id="second", polygon=[[100, 250], [900, 250], [900, 350], [100, 350]], raw_text="Another correct transcription.", status="recognized"),
    ]
    learning.register_project(home, project)
    window = MainWindow()
    window._set_project(project)
    monkeypatch.setattr(window, "_confirm_discard", lambda *args: True)
    yield window
    window._save_timer.stop()
    window.close()
    app.processEvents()


def test_unedited_line_requires_explicit_confirmation_and_edit_invalidates(reviewed_window):
    window = reviewed_window
    page, line = window.project.pages[0], window.project.pages[0].lines[0]
    assert not learning.is_line_reviewed(window.project, page, line)
    assert line.edited_text is None
    window.line_editors[line.id].review_checkbox.setChecked(True)
    assert learning.is_line_reviewed(window.project, page, line)
    assert line.edited_text is None
    window.line_editors[line.id].editor.setPlainText("A changed human correction.")
    assert line.edited_text == "A changed human correction."
    assert not line.reviewed and not line.reviewed_hash
    assert not window.line_editors[line.id].review_checkbox.isChecked()
    window._save_project(quiet=True)
    restored = load_project(window.project.work_dir)
    assert not restored.pages[0].lines[0].reviewed
    assert restored.pages[0].lines[0].text == line.text


def test_exclusion_and_page_confirmation_are_explicit(reviewed_window):
    window = reviewed_window
    page = window.project.pages[0]
    window.line_editors["second"].exclude_checkbox.setChecked(True)
    window._confirm_page()
    assert learning.is_line_reviewed(window.project, page, page.lines[0])
    assert page.lines[1].training_excluded and not page.lines[1].reviewed
    window.line_editors["first"].exclude_checkbox.setChecked(True)
    assert not page.lines[0].reviewed
    window.line_editors["first"].exclude_checkbox.setChecked(False)
    assert not page.lines[0].reviewed


def test_geometry_split_and_merge_remove_confirmation(reviewed_window):
    window = reviewed_window
    page = window.project.pages[0]
    window._confirm_page()
    first, second = page.lines
    window._change_line_box(first.id, [[110, 110], [910, 110], [910, 210], [110, 210]])
    assert not first.reviewed and first.status == "stale"
    window._selected_ids = [second.id]
    window._split_line()
    assert not second.reviewed
    assert all(not line.reviewed for line in page.lines)
    window._selected_ids = [line.id for line in page.lines if line.id != first.id]
    window._merge_lines()
    assert all(not line.reviewed for line in page.lines)


def test_model_switch_keeps_text_and_confirmation(reviewed_window, monkeypatch):
    window = reviewed_window
    page = window.project.pages[0]
    window.line_editors["first"].editor.setPlainText("Preserved human text.")
    window._confirm_page()
    old_hash = page.lines[0].reviewed_hash
    selected = []
    monkeypatch.setattr(learning, "set_active_model", lambda home, model: selected.append(model))
    window.recognizer_combo.addItem("个人模型", "personal-v1")
    window.recognizer_combo.setCurrentIndex(window.recognizer_combo.findData("personal-v1"))
    assert selected == ["personal-v1"]
    assert window.project.recognizer_id == "personal-v1"
    assert page.lines[0].text == "Preserved human text."
    assert page.lines[0].reviewed_hash == old_hash
    assert learning.is_line_reviewed(window.project, page, page.lines[0])
    window.recognizer_combo.setCurrentIndex(0)
    assert page.lines[0].text == "Preserved human text."


def test_training_busy_locks_ocr_and_review_controls(reviewed_window):
    window = reviewed_window
    window._training_state_changed(True)
    assert not window.start_button.isEnabled()
    assert not window.recognizer_combo.isEnabled()
    assert not window.canvas.editable
    assert all(card.editor.isReadOnly() and not card.review_checkbox.isEnabled() for card in window.line_editors.values())
    window._start_job("recognize")
    assert window._thread is None
    window._training_state_changed(False)
    assert window.start_button.isEnabled()


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class FakeProcess:
    ProcessState = QProcess.ProcessState
    ProcessChannelMode = QProcess.ProcessChannelMode
    ProcessError = QProcess.ProcessError

    def __init__(self, parent):
        self.started, self.finished, self.errorOccurred = _Signal(), _Signal(), _Signal()
        self._state = self.ProcessState.NotRunning

    def state(self):
        return self._state

    def setProcessChannelMode(self, value):
        pass

    def setStandardOutputFile(self, path, mode):
        self.log_path = path

    def setProcessEnvironment(self, value):
        pass

    def setWorkingDirectory(self, value):
        self.working_directory = value

    def start(self, executable, arguments):
        self.executable, self.arguments = executable, arguments
        self._state = self.ProcessState.Running
        self.started.emit()

    def processId(self):
        return 123456789

    def deleteLater(self):
        pass


@pytest.fixture
def dialog(app, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    writer = learning.save_writer(home, "测试作者")
    monkeypatch.setattr(learning, "summarize_dataset", lambda *args: {"writer": writer, "confirmed_lines": 0, "eligible_lines": 0, "pages": 0, "counts": {}, "exclusions": [], "projects": [], "ready": False, "reason": "样本不足"})
    widget = TrainingDialog(home, home / "models")
    for _ in range(200):
        QTest.qWait(10)
        if not widget.busy:
            break
    assert not widget.busy
    yield widget
    widget._poll_timer.stop()
    if widget._busy:
        widget._busy = False
    widget.close()
    app.processEvents()


def _spec(dialog):
    writer = learning.load_writer(dialog.home)
    directory = dialog.home / "writers" / writer["id"] / "runs" / "test-run"
    directory.mkdir(parents=True)
    snapshot = dialog.home / "dataset.json"
    snapshot.write_text("{}", encoding="utf-8")
    spec = {"home": str(dialog.home), "models_dir": str(dialog.models_dir), "snapshot_path": str(snapshot), "run_dir": str(directory), "resume": False, "cancel_file": str(directory / "cancel.request")}
    (directory / "run.json").write_text(json.dumps({"run_id": "test-run", "writer_id": writer["id"], "status": "queued"}), encoding="utf-8")
    return spec


def test_training_uses_subprocess_jsonl_cancel_and_same_snapshot_resume(dialog, monkeypatch):
    monkeypatch.setattr(training_ui, "QProcess", FakeProcess)
    spec = _spec(dialog)
    directory = Path(spec["run_dir"])
    dialog.launch_spec(spec)
    assert dialog.busy
    assert dialog._process.arguments[-2:] == ["--training-worker", str(directory / "run-spec.json")]
    assert dialog._process.arguments[0].endswith("run_app.py")
    assert (directory / "run-spec.json").is_file()
    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "epoch", "stage": "train", "epoch": 4, "max_epochs": 20, "cer": 0.12, "wer": 0.25, "peak_gpu_mb": 1500, "message": "第4轮"}) + "\n")
    dialog._poll_events()
    assert "12.00%" in dialog.metrics_label.text()
    assert dialog.progress_bar.maximum() == 20 and dialog.progress_bar.value() == 4
    dialog.cancel_training()
    assert Path(spec["cancel_file"]).is_file()
    checkpoint = directory / "checkpoints" / "last"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"checkpoint")
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (directory / "result.json").write_text(json.dumps({"status": "cancelled", "run_id": "test-run"}), encoding="utf-8")
    dialog._process_finished(0, QProcess.ExitStatus.NormalExit)
    assert not dialog.busy
    assert dialog.resume_button.isEnabled()
    dialog.resume_selected()
    assert dialog.busy and dialog._spec["resume"] is True
    assert dialog._spec["snapshot_path"] == spec["snapshot_path"]
    assert not Path(spec["cancel_file"]).exists()
    assert (directory / "result.previous.json").is_file()
    (directory / "result.json").write_text(json.dumps({"status": "cancelled", "run_id": "test-run"}), encoding="utf-8")
    dialog._process_finished(0, QProcess.ExitStatus.NormalExit)
    assert not dialog.busy


def test_process_failure_unlocks_and_shows_actionable_message(dialog, monkeypatch):
    monkeypatch.setattr(training_ui, "QProcess", FakeProcess)
    spec = _spec(dialog)
    dialog.launch_spec(spec)
    dialog._process_finished(13, QProcess.ExitStatus.CrashExit)
    assert not dialog.busy
    assert "未产生完整结果" in dialog.notice.text()
    assert dialog._process is None


def test_frozen_process_dispatch_uses_current_executable(dialog, monkeypatch):
    monkeypatch.setattr(training_ui, "QProcess", FakeProcess)
    monkeypatch.setattr(training_ui.sys, "frozen", True, raising=False)
    spec = _spec(dialog)
    dialog.launch_spec(spec)
    assert dialog._process.arguments == ["--training-worker", str(Path(spec["run_dir"]) / "run-spec.json")]
    dialog._process_finished(13, QProcess.ExitStatus.CrashExit)


def test_missing_project_exclusion_is_displayed_without_line_id(dialog):
    dialog._display_summary({"exclusions": [{"project_id": "missing", "page_index": None, "line_id": None, "reason": "任务文件已丢失"}], "reason": "样本不足"})
    assert "任务文件已丢失" in dialog.exclusions_text.toPlainText()


def test_cancelling_snapshot_preparation_does_not_launch_training(dialog, monkeypatch):
    snapshot = dialog.home / "prepared-dataset.json"
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(learning, "create_snapshot", lambda *args, **kwargs: snapshot)
    launched = []
    monkeypatch.setattr(dialog, "launch_spec", lambda spec: launched.append(spec))
    dialog._pending_launch = True
    dialog._start_data_job("snapshot")
    dialog.cancel_training()
    for _ in range(200):
        QTest.qWait(5)
        if not dialog.busy:
            break
    assert not dialog.busy
    assert not launched
    assert dialog.snapshot_path == snapshot


def test_model_comparison_uses_baseline_and_candidate_rows(dialog):
    metrics = {"baseline": {"test": {"cer": .3, "wer": .5, "lines": [{"id": "line1", "ground_truth": "a correct line", "prediction": "wrong line"}]}}, "candidate": {"test": {"cer": .1, "wer": .2, "lines": [{"id": "line1", "ground_truth": "a correct line", "prediction": "a correct line"}]}}}
    dialog._show_evaluation({"metrics": metrics}, dialog.home)
    assert dialog.diff_table.rowCount() == 1
    assert dialog.diff_table.item(0, 2).text() == "wrong line"
    assert dialog.diff_table.item(0, 3).text() == "a correct line"
    assert "30.00%" in dialog.evaluation_label.text() and "10.00%" in dialog.evaluation_label.text()


def test_repeated_dataset_refresh_cleans_finished_threads(dialog):
    for _ in range(15):
        dialog.refresh_data()
        for _ in range(200):
            QTest.qWait(5)
            if not dialog.busy:
                break
        assert not dialog.busy
        assert dialog._data_thread is None and dialog._data_worker is None


def test_personal_model_resolution_failure_never_runs_base(reviewed_window, monkeypatch):
    from handwriter.gui import TaskWorker
    errors = []
    monkeypatch.setattr(learning, "resolve_recognizer", lambda *args: None)
    worker = TaskWorker(reviewed_window.project, recognizer_id="missing-personal")
    worker.failed.connect(errors.append)
    worker.run()
    assert errors and "个人模型缺失或校验失败" in errors[0]
