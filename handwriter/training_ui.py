"""Writer review, immutable datasets and isolated training process controls.

Qt worker threads only prepare CPU data.  Actual optimization always runs in a
separate instance of the application, communicating through local JSON files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QIODevice, QObject, QProcess, QProcessEnvironment, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import learning
from .style import APP_STYLE

logger = logging.getLogger(__name__)


def _read_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else ({} if default is None else default)
    except (OSError, ValueError):
        return {} if default is None else default


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".gui.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _metric(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _release_gpu_cache():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("Unable to release inference GPU cache")


class DatasetWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, home: Path, models_dir: Path, operation: str, parent=None):
        super().__init__(parent)
        self.home, self.models_dir, self.operation = home, models_dir, operation

    @Slot()
    def run(self):
        try:
            if self.operation == "summary":
                value = learning.summarize_dataset(self.home, self.models_dir)
            else:
                value = learning.create_snapshot(self.home, self.models_dir, enforce_minimum=True)
            self.result.emit(value)
        except Exception as exc:
            logger.exception("Dataset %s failed", self.operation)
            self.failed.emit(str(exc))


class TrainingDialog(QDialog):
    busy_changed = Signal(bool)
    models_changed = Signal()
    active_model_changed = Signal(str)
    project_changed = Signal()

    def __init__(self, home: Path, models_dir: Path, project=None, parent=None):
        super().__init__(parent)
        self.home = Path(home).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.project = project
        self.summary = {}
        self.snapshot_path = None
        self._busy = False
        self._data_thread = None
        self._data_worker = None
        self._data_operation = None
        self._pending_launch = False
        self._cancel_requested = False
        self._close_after_stop = False
        self._process = None
        self._spec = None
        self._event_offset = 0
        self._terminal_handled = False
        self._last_error = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(600)
        self._poll_timer.timeout.connect(self._poll_events)
        self.setWindowTitle("个人笔迹训练 · 本地校对与模型版本")
        self.setObjectName("trainingDialog")
        self.resize(1120, 850)
        self.setMinimumSize(930, 720)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._load_writer()
        self.refresh_models()
        self.refresh_runs()
        self.refresh_data()

    @property
    def busy(self):
        return self._busy

    def _button(self, text, callback, name):
        button = QPushButton(text)
        button.setObjectName(name)
        button.clicked.connect(callback)
        return button

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        heading = QLabel("让模型逐渐适应同一位作者的笔迹")
        heading.setObjectName("projectTitle")
        layout.addWidget(heading)
        description = QLabel("只使用你在本应用中明确确认的行图与转写；训练手动启动，完成后由你决定是否启用。")
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("trainingTabs")
        layout.addWidget(self.tabs, 1)
        self._build_data_tab()
        self._build_training_tab()
        self._build_models_tab()
        self.notice = QLabel("就绪")
        self.notice.setObjectName("trainingNotice")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        footer = QHBoxLayout()
        hint = QLabel("通用模型始终保留 · 本地训练 · 按整页划分训练、验证与测试")
        hint.setObjectName("muted")
        footer.addWidget(hint)
        footer.addStretch()
        self.close_button = self._button("关闭", self.close, "trainingCloseButton")
        footer.addWidget(self.close_button)
        layout.addLayout(footer)

    def _build_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        author_row = QHBoxLayout()
        author_row.addWidget(QLabel("作者姓名"))
        self.writer_edit = QLineEdit()
        self.writer_edit.setObjectName("writerNameEdit")
        self.writer_edit.setPlaceholderText("例如：祖父的英文手稿")
        author_row.addWidget(self.writer_edit, 1)
        self.save_writer_button = self._button("保存作者", self.save_writer, "saveWriterButton")
        author_row.addWidget(self.save_writer_button)
        self.register_button = self._button("将当前任务归入该作者", self.register_current_project, "registerProjectButton")
        author_row.addWidget(self.register_button)
        layout.addLayout(author_row)
        self.writer_description = QLabel()
        self.writer_description.setWordWrap(True)
        self.writer_description.setObjectName("muted")
        layout.addWidget(self.writer_description)
        stats_row = QHBoxLayout()
        self.dataset_label = QLabel("正在检查已确认数据……")
        self.dataset_label.setObjectName("sectionTitle")
        stats_row.addWidget(self.dataset_label, 1)
        self.refresh_button = self._button("刷新数据", self.refresh_data, "refreshDatasetButton")
        stats_row.addWidget(self.refresh_button)
        layout.addLayout(stats_row)
        self.split_label = QLabel("最低门槛：150 行、5 页；训练 / 验证 / 测试至少 90 / 20 / 20 行。")
        self.split_label.setWordWrap(True)
        layout.addWidget(self.split_label)
        self.readiness_label = QLabel()
        self.readiness_label.setObjectName("warning")
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label)
        label = QLabel("该作者关联的任务")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        self.project_list = QListWidget()
        self.project_list.setObjectName("writerProjectsList")
        layout.addWidget(self.project_list, 1)
        exclusion_title = QLabel("未纳入训练的行与原因")
        exclusion_title.setObjectName("sectionTitle")
        layout.addWidget(exclusion_title)
        self.exclusions_text = QPlainTextEdit()
        self.exclusions_text.setObjectName("trainingExclusions")
        self.exclusions_text.setReadOnly(True)
        self.exclusions_text.setMaximumHeight(160)
        layout.addWidget(self.exclusions_text)
        note = QLabel("先在此页关联任务，再回到主窗口逐行勾选「已核对，用于训练」，或使用「确认本页」。修改文字或行框后需重新确认；单纯编辑不会自动进入训练集。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        self.tabs.addTab(tab, "1  作者与数据")

    def _build_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.snapshot_button = self._button("建立数据快照", self.create_snapshot, "createSnapshotButton")
        self.start_button = self._button("开始训练", self.start_training, "startTrainingButton")
        self.start_button.setObjectName("primary")
        self.cancel_button = self._button("取消训练", self.cancel_training, "cancelTrainingButton")
        self.resume_button = self._button("恢复所选训练", self.resume_selected, "resumeTrainingButton")
        for button in [self.snapshot_button, self.start_button, self.cancel_button, self.resume_button]:
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        self.snapshot_label = QLabel("尚未建立快照。开始训练会先建立固定快照；后续校对不会改动本次训练数据。")
        self.snapshot_label.setObjectName("muted")
        self.snapshot_label.setWordWrap(True)
        layout.addWidget(self.snapshot_label)
        self.stage_label = QLabel("阶段：等待")
        self.stage_label.setObjectName("sectionTitle")
        self.metrics_label = QLabel("轮次：—    CER：—    WER：—    GPU 峰值：—")
        layout.addWidget(self.stage_label)
        layout.addWidget(self.metrics_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        self.event_log = QPlainTextEdit()
        self.event_log.setObjectName("trainingEventLog")
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(700)
        self.event_log.setMaximumHeight(120)
        layout.addWidget(self.event_log)
        label = QLabel("训练历史（取消后可从最近的完整检查点恢复）")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        self.run_list = QListWidget()
        self.run_list.setObjectName("trainingRunsList")
        self.run_list.setMaximumHeight(125)
        self.run_list.currentItemChanged.connect(self._run_selected)
        layout.addWidget(self.run_list)
        self.evaluation_label = QLabel("选择已评估的训练记录，查看相同保留集上的新旧模型差异。")
        self.evaluation_label.setWordWrap(True)
        layout.addWidget(self.evaluation_label)
        self.diff_table = QTableWidget(0, 4)
        self.diff_table.setObjectName("modelDifferenceTable")
        self.diff_table.setHorizontalHeaderLabels(["保留集 / 行", "人工转写", "通用模型", "个人模型"])
        self.diff_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diff_table.setWordWrap(True)
        self.diff_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.diff_table, 1)
        self.tabs.addTab(tab, "2  训练与评估")

    def _build_models_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        text = QLabel("训练完成不会自动切换模型。启用仅影响后续识别，不修改任何已有文字或人工校对。")
        text.setWordWrap(True)
        layout.addWidget(text)
        self.model_list = QListWidget()
        self.model_list.setObjectName("personalModelsList")
        self.model_list.currentItemChanged.connect(self._model_selected)
        layout.addWidget(self.model_list, 1)
        self.model_details = QLabel("通用 TrOCR 模型始终保留。")
        self.model_details.setWordWrap(True)
        layout.addWidget(self.model_details)
        row = QHBoxLayout()
        self.activate_button = self._button("启用所选模型", self.activate_selected, "activateModelButton")
        self.base_button = self._button("回到通用模型", self.activate_base, "useBaseModelButton")
        row.addWidget(self.activate_button)
        row.addWidget(self.base_button)
        row.addStretch()
        layout.addLayout(row)
        self.tabs.addTab(tab, "3  模型版本")

    def _set_busy(self, busy):
        changed = bool(busy) != self._busy
        self._busy = bool(busy)
        if changed:
            self.busy_changed.emit(self._busy)
        self._refresh_controls()

    def _refresh_controls(self):
        writer = learning.load_writer(self.home)
        writable = not self._busy
        for widget in [self.writer_edit, self.save_writer_button, self.refresh_button, self.activate_button, self.base_button]:
            widget.setEnabled(writable)
        self.register_button.setEnabled(writable and self.project is not None and bool(writer))
        self.snapshot_button.setEnabled(writable and bool(self.summary.get("ready")))
        self.start_button.setEnabled(writable and (bool(self.summary.get("ready")) or self.snapshot_path is not None))
        self.cancel_button.setEnabled(self._busy and not self._cancel_requested)
        selected = self.run_list.currentItem()
        resumable = self._resumable(selected.data(Qt.ItemDataRole.UserRole)) if selected else False
        self.resume_button.setEnabled(writable and resumable)

    def _load_writer(self):
        writer = learning.load_writer(self.home)
        self.writer_edit.setText(writer["name"] if writer else "")
        assigned = self.project is not None and writer and getattr(self.project, "writer_id", "") == writer["id"]
        self.writer_description.setText((f"当前作者：{writer['name']}。" if writer else "请先保存一位作者。") + ("当前任务已归入该作者。" if assigned else "任务需要明确关联，旧任务不会自动纳入。"))

    def _flush_project(self):
        parent = self.parent()
        if parent is not None and hasattr(parent, "_save_project"):
            parent._save_project(quiet=True)

    def save_writer(self):
        if self._busy:
            return
        try:
            learning.save_writer(self.home, self.writer_edit.text().strip())
            self._load_writer()
            self.project_changed.emit()
            self.refresh_data()
        except Exception as exc:
            self._show_error(str(exc))

    def register_current_project(self):
        if self._busy or self.project is None:
            return
        try:
            self._flush_project()
            learning.register_project(self.home, self.project)
            self._load_writer()
            self.project_changed.emit()
            self.refresh_data()
        except Exception as exc:
            self._show_error(str(exc))

    def refresh_data(self):
        if not self._busy:
            self._start_data_job("summary")

    def create_snapshot(self):
        if self._busy:
            return
        self._pending_launch = False
        self._start_data_job("snapshot")

    def start_training(self):
        if self._busy:
            return
        self.tabs.setCurrentIndex(1)
        if self.snapshot_path is not None and Path(self.snapshot_path).is_file():
            self._launch_new_run()
        else:
            self._pending_launch = True
            self._start_data_job("snapshot")

    def _start_data_job(self, operation):
        self._flush_project()
        self._cancel_requested = False
        self._last_error = None
        self._data_operation = operation
        self._set_busy(True)
        self.notice.setText("正在检查数据……" if operation == "summary" else "正在裁切、校验并建立固定数据快照……")
        self._data_worker = DatasetWorker(self.home, self.models_dir, operation, self)
        self._data_thread = self._data_worker
        self._data_worker.result.connect(self._data_result, Qt.ConnectionType.QueuedConnection)
        self._data_worker.failed.connect(self._data_failed, Qt.ConnectionType.QueuedConnection)
        self._data_thread.finished.connect(self._data_finished, Qt.ConnectionType.QueuedConnection)
        self._data_thread.start()

    @Slot(object)
    def _data_result(self, value):
        if self._data_operation == "summary":
            self.summary = value
            self._display_summary(value)
        else:
            self.snapshot_path = Path(value)
            self.snapshot_label.setText("固定训练快照：" + str(self.snapshot_path))
            self.notice.setText("快照已建立。后续校对不会修改这个快照。")

    @Slot(str)
    def _data_failed(self, message):
        self._last_error = message
        self._show_error(message)

    @Slot()
    def _data_finished(self):
        thread = self._data_thread
        if thread is not None:
            thread.wait()
        self._data_thread, self._data_worker = None, None
        if thread is not None:
            thread.deleteLater()
        launch = self._pending_launch and not self._cancel_requested and not self._last_error and self.snapshot_path is not None
        self._pending_launch = False
        if launch:
            self._launch_new_run()
        else:
            self._set_busy(False)
            if self._cancel_requested:
                self.notice.setText("已取消启动训练，已完成的数据准备保留。")
            if self._close_after_stop:
                self.close()

    def _display_summary(self, summary):
        self.dataset_label.setText(f"已确认 {summary.get('confirmed_lines', 0)} 行  ·  合格 {summary.get('eligible_lines', 0)} 行  ·  {summary.get('pages', 0)} 页")
        counts = summary.get("counts", {})
        self.split_label.setText(f"训练 / 验证 / 测试：{counts.get('train', 0)} / {counts.get('validation', 0)} / {counts.get('test', 0)} 行\n最低门槛：150 行、5 页，三个集合至少 90 / 20 / 20 行；按整页划分。")
        self.readiness_label.setText(summary.get("reason", "请关联任务并确认正确转写。"))
        self.project_list.clear()
        for project in summary.get("projects", []):
            suffix = "" if project.get("exists", True) else " · 文件缺失"
            item = QListWidgetItem(project.get("title", project.get("id", "任务")) + suffix)
            item.setToolTip(str(project.get("path", "")))
            self.project_list.addItem(item)
        exclusions = []
        for item in summary.get("exclusions", []):
            if isinstance(item, str):
                exclusions.append(item)
            else:
                page = item.get("page_index")
                prefix = f"第 {page + 1} 页" if isinstance(page, int) else "任务"
                exclusions.append(f"{prefix} · {str(item.get('line_id') or '')[:10]}：{item.get('reason', '未纳入')}")
        self.exclusions_text.setPlainText("\n".join(exclusions) if exclusions else "没有额外排除项。只有明确确认且通过数据校验的文字行才会进入快照。")
        self.notice.setText("数据检查完成。" + summary.get("reason", ""))

    def _new_spec(self):
        writer = learning.load_writer(self.home)
        if not writer:
            raise ValueError("请先保存作者。")
        if self.snapshot_path is None or not Path(self.snapshot_path).is_file():
            raise ValueError("找不到完整数据快照，请重新建立快照。")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        run_dir = self.home / "writers" / writer["id"] / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        spec = {"home": str(self.home), "models_dir": str(self.models_dir), "snapshot_path": str(Path(self.snapshot_path).resolve()), "run_dir": str(run_dir), "resume": False, "cancel_file": str(run_dir / "cancel.request")}
        _write_json(run_dir / "run-spec.json", spec)
        _write_json(run_dir / "run.json", {"id": run_id, "run_id": run_id, "writer_id": writer["id"], "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(), "snapshot_id": Path(spec["snapshot_path"]).parent.name, "run_dir": str(run_dir)})
        return spec

    def _launch_new_run(self):
        try:
            self.launch_spec(self._new_spec())
        except Exception as exc:
            self._show_error(str(exc))
            self._set_busy(False)

    def launch_spec(self, spec):
        """Launch a real optimizer subprocess; kept public for integration tests."""
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("已有训练进程正在运行。")
        self._spec = dict(spec)
        run_dir = Path(self._spec["run_dir"]).resolve()
        if not run_dir.is_relative_to((self.home / "writers").resolve()):
            raise ValueError("训练目录不属于当前作者工作区。")
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "run-spec.json"
        _write_json(spec_path, self._spec)
        cancel_path = Path(self._spec["cancel_file"])
        if not cancel_path.resolve().is_relative_to(run_dir):
            raise ValueError("取消请求文件不属于当前训练目录。")
        if cancel_path.exists():
            cancel_path.unlink()
        if self._spec.get("resume") and (run_dir / "result.json").is_file():
            (run_dir / "result.json").replace(run_dir / "result.previous.json")
        self._cancel_requested = False
        self._terminal_handled = False
        self._event_offset = 0
        self._last_error = None
        self._set_busy(True)
        self.tabs.setCurrentIndex(1)
        self.stage_label.setText("阶段：启动独立训练进程")
        self.progress_bar.setRange(0, 0)
        self.event_log.clear()
        self.notice.setText("训练独立运行；识别暂时锁定。完成后不会自动启用新模型。")
        _release_gpu_cache()
        process = QProcess(self)
        self._process = process
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setStandardOutputFile(str(run_dir / "worker-console.log"), QIODevice.OpenModeFlag.Append)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("HANDWRITER_HOME", str(self.home))
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(self.home if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent))
        arguments = ["--training-worker", str(spec_path)] if getattr(sys, "frozen", False) else [str(Path(__file__).resolve().parent.parent / "run_app.py"), "--training-worker", str(spec_path)]
        process.started.connect(self._process_started)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        process.start(sys.executable, arguments)
        self._poll_timer.start()

    def _update_run_state(self, **changes):
        if self._spec:
            path = Path(self._spec["run_dir"]) / "run.json"
            value = _read_json(path)
            value.update(changes)
            _write_json(path, value)

    def _process_started(self):
        self._update_run_state(status="running", process_pid=int(self._process.processId()))
        self.refresh_runs()

    def _poll_events(self):
        if not self._spec:
            return
        path = Path(self._spec["run_dir"]) / "events.jsonl"
        try:
            with path.open("rb") as handle:
                if path.stat().st_size < self._event_offset:
                    self._event_offset = 0
                handle.seek(self._event_offset)
                for raw in handle:
                    if not raw.endswith(b"\n"):
                        break
                    self._event_offset += len(raw)
                    try:
                        self._apply_event(json.loads(raw.decode("utf-8")))
                    except (ValueError, UnicodeError):
                        logger.warning("Malformed training event ignored")
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Unable to read training events")

    def _apply_event(self, event):
        if not isinstance(event, dict):
            raise ValueError("Invalid training event")
        stage_names = {"validate": "校验快照", "baseline": "评估通用模型", "load": "加载训练模型", "train": "训练", "evaluate": "评估个人模型", "publish": "保存个人模型"}
        stage = event.get("stage", "")
        self.stage_label.setText("阶段：" + stage_names.get(stage, stage or "准备"))
        epoch = event.get("epoch", "—")
        gpu = event.get("peak_gpu_mb")
        gpu_text = f"{float(gpu):.0f} MB" if isinstance(gpu, (float, int)) else "—"
        self.metrics_label.setText(f"轮次：{epoch}    CER：{_metric(event.get('cer'))}    WER：{_metric(event.get('wer'))}    GPU 峰值：{gpu_text}")
        message = str(event.get("message") or event.get("event") or "")
        if event.get("loss") is not None:
            message += f" · loss {event['loss']}"
        self.event_log.appendPlainText(message)
        self.notice.setText(message)
        if isinstance(epoch, (int, float)) and stage == "train":
            max_epochs = max(1, int(event.get("max_epochs", 20)))
            self.progress_bar.setRange(0, max_epochs)
            self.progress_bar.setValue(min(max_epochs, int(epoch)))

    def cancel_training(self):
        if not self._busy:
            return
        self._cancel_requested = True
        self._pending_launch = False
        if self._spec and self._process is not None:
            Path(self._spec["cancel_file"]).write_text("cancel\n", encoding="utf-8")
            self._update_run_state(status="cancel_requested")
        self.notice.setText("已请求取消，等待当前安全检查点；已完成的训练状态会保留。")
        self._refresh_controls()

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._complete_process(-1, "训练进程无法启动，请查看 worker-console.log。")

    def _process_finished(self, exit_code, exit_status):
        self._complete_process(int(exit_code))

    def _complete_process(self, exit_code, startup_error=None):
        if self._terminal_handled:
            return
        self._terminal_handled = True
        self._poll_events()
        self._poll_timer.stop()
        run_dir = Path(self._spec["run_dir"])
        result = _read_json(run_dir / "result.json")
        status = result.get("status", "failed")
        if startup_error or not result:
            result = {"status": "failed", "error": startup_error or f"训练进程退出（代码 {exit_code}），未产生完整结果。请查看日志；有完整检查点时可恢复。"}
            status = "failed"
        try:
            if status == "completed" and result.get("published_dir"):
                learning.register_model(self.home, result)
                self.models_changed.emit()
            self._update_run_state(status=status, exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            logger.exception("Unable to register training result")
            self._show_error("训练结果登记失败：" + str(exc))
            status = "failed"
        if status == "completed":
            self.notice.setText("训练完成。新版本尚未启用，请比较保留集差异后在「模型版本」中选择。")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
        elif status == "cancelled":
            self.notice.setText("训练已取消，完整检查点已保留；选择历史记录可恢复。")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        else:
            self._show_error(result.get("error", "训练失败，请查看日志。"))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        self._show_evaluation(result, run_dir)
        process = self._process
        self._process = None
        if process is not None:
            process.deleteLater()
        self._set_busy(False)
        self.refresh_runs()
        self.refresh_models()
        if self._close_after_stop:
            self.close()

    def refresh_runs(self):
        previous = self.run_list.currentItem()
        selected_id = (previous.data(Qt.ItemDataRole.UserRole) or {}).get("run_id") if previous else None
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for run in learning.list_runs(self.home):
            run_id = run.get("run_id", run.get("id", ""))
            run_dir = self._run_directory(run)
            run = dict(run, run_id=run_id, run_dir=str(run_dir))
            item = QListWidgetItem(f"{run_id}  ·  {run.get('status', '未知')}")
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.run_list.addItem(item)
            if run_id == selected_id:
                self.run_list.setCurrentItem(item)
        if self.run_list.currentRow() < 0 and self.run_list.count():
            self.run_list.setCurrentRow(0)
        self.run_list.blockSignals(False)
        self._refresh_controls()

    def _run_directory(self, run):
        if run.get("run_dir"):
            return Path(run["run_dir"])
        writer = learning.load_writer(self.home) or {}
        return self.home / "writers" / str(writer.get("id", "")) / "runs" / str(run.get("run_id", run.get("id", "")))

    def _resumable(self, run):
        if not run or run.get("status") == "completed":
            return False
        directory = self._run_directory(run)
        checkpoint = directory / "checkpoints" / "last"
        return (directory / "run-spec.json").is_file() and (checkpoint / "state.pt").is_file() and (checkpoint / "adapter" / "adapter_model.safetensors").is_file()

    def resume_selected(self):
        if self._busy:
            return
        item = self.run_list.currentItem()
        run = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not self._resumable(run):
            self._show_error("这条训练记录没有可恢复的完整检查点。")
            return
        try:
            directory = self._run_directory(run)
            spec = _read_json(directory / "run-spec.json")
            spec["resume"] = True
            self.snapshot_path = Path(spec["snapshot_path"])
            self.snapshot_label.setText("恢复训练，沿用原快照：" + str(self.snapshot_path))
            self.launch_spec(spec)
        except Exception as exc:
            self._show_error(str(exc))
            self._set_busy(False)

    def _run_selected(self, current, previous=None):
        if current:
            run = current.data(Qt.ItemDataRole.UserRole)
            self._show_evaluation(run, self._run_directory(run))
        self._refresh_controls()

    def _show_evaluation(self, result, run_dir):
        evaluation = _read_json(Path(run_dir) / "evaluation.json")
        metrics = evaluation.get("metrics", evaluation) if evaluation else result.get("metrics", {})
        baseline, candidate = metrics.get("baseline", {}), metrics.get("candidate", {})
        before, after = baseline.get("test", {}), candidate.get("test", {})
        self.evaluation_label.setText(f"保留测试集  CER：{_metric(before.get('cer'))} → {_metric(after.get('cer'))}    WER：{_metric(before.get('wer'))} → {_metric(after.get('wer'))}。仅对本次固定保留集有效。")
        rows = []
        for split in ["validation", "test"]:
            original = {str(line.get("id")): line for line in baseline.get(split, {}).get("lines", [])}
            changed = {str(line.get("id")): line for line in candidate.get(split, {}).get("lines", [])}
            for line_id in sorted(original.keys() | changed.keys()):
                old, new = original.get(line_id, {}), changed.get(line_id, {})
                rows.append([f"{'验证' if split == 'validation' else '测试'} / {line_id[:10]}", str(new.get("ground_truth", old.get("ground_truth", ""))), str(old.get("prediction", "")), str(new.get("prediction", ""))])
        self.diff_table.setRowCount(len(rows))
        for row_number, values in enumerate(rows):
            for column, text in enumerate(values):
                self.diff_table.setItem(row_number, column, QTableWidgetItem(text))
        self.diff_table.resizeRowsToContents()

    def refresh_models(self):
        selected = self.model_list.currentItem()
        selected_id = (selected.data(Qt.ItemDataRole.UserRole) or {}).get("id") if selected else None
        self.model_list.clear()
        for model in learning.list_models(self.home):
            title = model.get("title", model["id"])
            if model.get("active"):
                title += "  ·  当前启用"
            if not model.get("available", True):
                title += "  ·  文件缺失"
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, model)
            self.model_list.addItem(item)
            if model["id"] == selected_id or (selected_id is None and model.get("active")):
                self.model_list.setCurrentItem(item)
        self._refresh_controls()

    def _model_selected(self, current, previous=None):
        if not current:
            return
        model = current.data(Qt.ItemDataRole.UserRole)
        metrics = model.get("metrics", {})
        baseline = metrics.get("baseline", {}).get("test", {})
        candidate = metrics.get("candidate", {}).get("test", {})
        self.model_details.setText("通用 TrOCR：原始模型保留。" if model["id"] == "base" else f"测试集 CER {_metric(baseline.get('cer'))} → {_metric(candidate.get('cer'))}；WER {_metric(baseline.get('wer'))} → {_metric(candidate.get('wer'))}。\n" + ("验证集 CER 有改善；测试集表现请以上方数值为准。" if model.get("improved") else "验证集 CER 未改善，启用前请仔细比较测试集数值。"))

    def activate_selected(self):
        if self._busy:
            return
        item = self.model_list.currentItem()
        if item:
            self._activate(item.data(Qt.ItemDataRole.UserRole)["id"])

    def activate_base(self):
        if not self._busy:
            self._activate("base")

    def _activate(self, model_id):
        try:
            learning.set_active_model(self.home, model_id)
            self.refresh_models()
            self.active_model_changed.emit(model_id)
            self.notice.setText("已切换后续识别模型；已有转写与人工校对完整保留。")
        except Exception as exc:
            self._show_error(str(exc))

    def _show_error(self, message):
        self.notice.setText("提示：" + str(message))
        self.event_log.appendPlainText("提示：" + str(message))

    def reject(self):
        # Escape follows the same safe cancellation path as the Close button.
        if self._busy:
            self.close()
        else:
            super().reject()

    def closeEvent(self, event):
        if self._busy:
            answer = QMessageBox.question(self, "训练仍在进行", "关闭前需要请求取消，并等待当前安全检查点完成。是否取消并关闭训练页？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Yes:
                self._close_after_stop = True
                self.cancel_training()
            event.ignore()
            return
        event.accept()
