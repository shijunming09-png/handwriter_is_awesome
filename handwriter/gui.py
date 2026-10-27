"""Chinese desktop application and its cancellable OCR worker."""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path

from PIL import Image, ImageStat
from PySide6.QtCore import QEvent, QObject, QRectF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QFont, QFontDatabase, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from . import paths
from .canvas import PageCanvas, bounds, pil_pixmap, rectangle_polygon
from .data import LineResult
from .exporter import export_docx, export_txt
from .pdfio import load_page_image
from .storage import create_project, load_project, save_project
from .style import APP_STYLE

logger = logging.getLogger(__name__)

STATUS = {"pending": "待处理", "detected": "已切行", "recognized": "已识别", "review": "待复核", "stale": "需重新识别", "error": "处理失败"}


def _ensure_fonts():
    # The offscreen Qt platform does not enumerate system fonts on Windows.
    # Loading installed fonts also makes fallback behavior predictable in a bundle.
    if not QFontDatabase.families():
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ["msyh.ttc", "segoeui.ttf", "seguisb.ttf"]:
            if (fonts / filename).is_file():
                QFontDatabase.addApplicationFont(str(fonts / filename))
    app = QApplication.instance()
    if app is not None:
        app.setFont(QFont("Microsoft YaHei UI", 10))


def _sort_lines(lines):
    lines.sort(key=lambda line: (sum(point[1] for point in line.polygon) / len(line.polygon), min(point[0] for point in line.polygon)))


def _update_page_status(page):
    if any(line.status in {"pending", "stale"} for line in page.lines):
        page.status = "detected"
    elif any(line.status in {"review", "error"} for line in page.lines):
        page.status = "review"
    else:
        page.status = "recognized"


class TaskWorker(QObject):
    progress = Signal(str)
    page_done = Signal(int)
    counts = Signal(int, int)
    failed = Signal(str)
    finished = Signal(str)

    def __init__(self, project, mode="recognize", page_index=None, line_ids=None, recognizer_id="base"):
        super().__init__()
        self.project = project
        self.mode = mode
        self.page_index = page_index
        self.line_ids = set(line_ids or [])
        self.recognizer_id = recognizer_id
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        result = "处理完成"
        engine = None
        gpu_lock = None
        try:
            from filelock import FileLock, Timeout
            gpu_lock = FileLock(str(paths.app_home() / "training.lock"))
            try:
                gpu_lock.acquire(timeout=0)
            except Timeout as exc:
                raise RuntimeError("训练或另一识别任务仍在运行，请等待完成后重试。") from exc
            from .engine import OcrEngine
            from .learning import invalidate_line_review, is_line_reviewed, resolve_recognizer
            from .model_manager import models_ready
            recognizer_dir = resolve_recognizer(paths.app_home(), self.recognizer_id)
            if self.recognizer_id != "base" and recognizer_dir is None:
                raise ValueError("所选个人模型缺失或校验失败，请明确切换到通用模型后重试。")
            engine = OcrEngine(paths.models_dir(), progress=self.progress.emit, cancelled=self.cancel_event.is_set, recognizer_dir=recognizer_dir)
            if self.mode == "prepare" or not models_ready(paths.models_dir()):
                self.progress.emit("准备本地模型，文稿不会上传……")
                engine.prepare_models()
            if self.cancel_event.is_set():
                result = "已取消，已完成的结果已保留"
                return
            if self.mode == "prepare":
                result = "模型准备完成，之后可离线使用"
                return
            self.progress.emit("正在加载本地模型并检查计算设备……")
            device = engine.load()
            self.progress.emit(f"运行设备：{device}")
            pages = self.project.pages if self.page_index is None else [self.project.pages[self.page_index]]
            errors = 0
            for number, page in enumerate(pages, 1):
                if self.cancel_event.is_set():
                    break
                self.counts.emit(number - 1, len(pages))
                self.progress.emit(f"第 {page.index + 1} / {len(self.project.pages)} 页：准备页面")
                if self.mode != "selected" and page.status in {"recognized", "review"} and not any(line.status in {"pending", "stale", "error"} for line in page.lines):
                    self.counts.emit(number, len(pages))
                    continue
                try:
                    image = load_page_image(self.project, page)
                    page.error = ""
                    if not page.lines and self.mode != "selected":
                        self.progress.emit(f"第 {page.index + 1} 页：自动检测与切行")
                        page.lines = engine.detect(image)
                        _sort_lines(page.lines)
                        page.status = "detected"
                        save_project(self.project)
                    if self.cancel_event.is_set():
                        break
                    if not page.lines:
                        small = image.convert("L")
                        small.thumbnail((256, 256))
                        histogram = small.histogram()
                        ink_fraction = sum(histogram[:220]) / max(1, sum(histogram))
                        page.status = "review" if ink_fraction > 0.001 else "recognized"
                        page.warnings = ["页面可能有文字，但未检测到文字行。请检查原图，必要时手动添加行框。"] if page.status == "review" else []
                    else:
                        selected = [line for line in page.lines if line.id in self.line_ids] if self.mode == "selected" else [line for line in page.lines if line.status in {"pending", "stale", "error"}]
                        self.progress.emit(f"第 {page.index + 1} 页：识别 {len(selected)} 行")
                        if selected:
                            engine.recognize(image, selected)
                            for line in selected:
                                if line.status not in {"pending", "stale", "error"}:
                                    line.recognition_model_id = self.recognizer_id
                                if getattr(line, "reviewed", False) and not is_line_reviewed(self.project, page, line):
                                    invalidate_line_review(line)
                        _update_page_status(page)
                except Exception as exc:
                    if self.cancel_event.is_set():
                        break
                    logger.exception("Page %s failed", page.index + 1)
                    page.error = str(exc)
                    page.status = "error"
                    errors += 1
                finally:
                    save_project(self.project)
                    self.page_done.emit(page.index)
                self.counts.emit(number, len(pages))
            if self.cancel_event.is_set():
                result = "已取消，已完成的结果已保留"
            elif errors:
                result = f"处理结束，{errors} 页失败；可修复后继续识别"
            else:
                result = "识别完成，请对照原稿校对后导出"
        except Exception as exc:
            if self.cancel_event.is_set():
                result = "已取消，已完成的结果已保留"
            else:
                logger.exception("OCR worker failed")
                self.failed.emit(str(exc))
                result = "处理未完成，请查看错误提示"
        finally:
            # No recognizer survives an OCR job: a training subprocess needs the GPU.
            if engine is not None and hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception:
                    logger.exception("Recognizer cleanup failed")
            engine = None
            import gc
            gc.collect()
            try:
                import sys
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.exception("GPU cache cleanup failed")
            if gpu_lock is not None and gpu_lock.is_locked:
                gpu_lock.release()
            if self.project is not None:
                try:
                    save_project(self.project)
                except Exception:
                    logger.exception("Failed to save project at task end")
            self.finished.emit(result)


class CropPreview(QLabel):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.original = pil_pixmap(image)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumSize(1, 26)
        self.setMaximumHeight(76)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.width() > 0:
            scaled = self.original.scaled(self.width(), 74, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)
            self.setFixedHeight(max(26, scaled.height()))


class LineEditor(QFrame):
    text_changed = Signal(str, str)
    focused = Signal(str)
    review_changed = Signal(str, bool)
    exclusion_changed = Signal(str, bool)

    def __init__(self, line, number, image=None, parent=None):
        super().__init__(parent)
        self.line_id = line.id
        self.setObjectName("lineCard")
        self.setProperty("selected", False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(6)
        row = QHBoxLayout()
        title = QLabel(f"第 {number} 行")
        title.setStyleSheet("font-weight: 700;")
        row.addWidget(title)
        row.addStretch()
        self.state_label = QLabel(STATUS.get(line.status, line.status))
        self.state_label.setObjectName("statusText")
        row.addWidget(self.state_label)
        layout.addLayout(row)
        if image is not None:
            rect = bounds(line.polygon).intersected(QRectF(0, 0, image.width, image.height))
            if rect.width() > 0 and rect.height() > 0:
                crop = image.crop((int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())))
                crop.thumbnail((690, 75))
                thumb = CropPreview(crop)
                layout.addWidget(thumb)
        self.editor = QPlainTextEdit(line.text)
        self.editor.setObjectName(f"lineText_{line.id}")
        self.editor.setPlaceholderText("识别后在这里校对；也可以直接输入人工转写。")
        self.editor.setMinimumHeight(62)
        self.editor.setMaximumHeight(100)
        self.editor.setTabChangesFocus(True)
        self.editor.installEventFilter(self)
        self.editor.textChanged.connect(lambda: self.text_changed.emit(self.line_id, self.editor.toPlainText()))
        layout.addWidget(self.editor)
        self.warning_label = QLabel("；".join(line.warnings))
        self.warning_label.setObjectName("warning")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(bool(line.warnings))
        layout.addWidget(self.warning_label)
        training_row = QHBoxLayout()
        self.review_checkbox = QCheckBox("已核对，用于训练")
        self.review_checkbox.setObjectName(f"reviewed_{line.id}")
        self.review_checkbox.setChecked(bool(getattr(line, "reviewed", False)))
        self.review_checkbox.setToolTip("请核对完整行图与转写；未修改的正确识别结果也可确认。修改文字或行框后需重新确认。")
        self.review_checkbox.toggled.connect(lambda checked: self.review_changed.emit(self.line_id, checked))
        self.exclude_checkbox = QCheckBox("排除训练")
        self.exclude_checkbox.setObjectName(f"excluded_{line.id}")
        self.exclude_checkbox.setChecked(bool(getattr(line, "training_excluded", False)))
        self.exclude_checkbox.toggled.connect(lambda checked: self.exclusion_changed.emit(self.line_id, checked))
        training_row.addWidget(self.review_checkbox)
        training_row.addStretch()
        training_row.addWidget(self.exclude_checkbox)
        layout.addLayout(training_row)

    def update_status(self, line):
        self.state_label.setText("人工转写" if line.edited_text is not None and not line.raw_text and line.status == "recognized" else STATUS.get(line.status, line.status))
        self.warning_label.setText("；".join(line.warnings))
        self.warning_label.setVisible(bool(line.warnings))
        for checkbox, value in [(self.review_checkbox, bool(getattr(line, "reviewed", False))), (self.exclude_checkbox, bool(getattr(line, "training_excluded", False)))]:
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)

    def eventFilter(self, watched, event):
        if watched is self.editor and event.type() == QEvent.Type.FocusIn:
            self.focused.emit(self.line_id)
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        self.focused.emit(self.line_id)
        super().mousePressEvent(event)

    def set_selected(self, selected):
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        _ensure_fonts()
        self.project = None
        self.current_page_index = -1
        self.current_image = None
        self.line_editors: dict[str, LineEditor] = {}
        self._selected_ids = []
        self._busy = False
        self._training_busy = False
        self._training_dialog = None
        self._worker = None
        self._thread = None
        self._job_error = None
        self._close_after_task = False
        self._dirty = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(650)
        self._save_timer.timeout.connect(self._save_project)
        self.setWindowTitle("墨迹 Handwriter · 本地英文手稿识别")
        self.setObjectName("handwriterWindow")
        self.resize(1450, 940)
        self.setMinimumSize(1100, 740)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLE)
        self._build_menu()
        self._build_ui()
        self._update_model_badge()
        self._refresh_recognizer_choices()
        self._refresh_actions()

    def _action(self, text, slot, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _build_menu(self):
        self.import_action = self._action("导入 PDF…", self._import_pdf, QKeySequence.StandardKey.Open)
        self.open_action = self._action("打开已保存任务…", self._open_project, "Ctrl+Shift+O")
        self.save_action = self._action("保存任务", self._save_project, QKeySequence.StandardKey.Save)
        self.export_word_action = self._action("导出 Word…", lambda: self._export("docx"))
        self.export_txt_action = self._action("导出 TXT…", lambda: self._export("txt"))
        menu = self.menuBar().addMenu("文件")
        for action in [self.import_action, self.open_action, self.save_action]:
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self.export_word_action)
        menu.addAction(self.export_txt_action)
        menu.addSeparator()
        menu.addAction(self._action("退出", self.close, QKeySequence.StandardKey.Quit))
        learning_menu = self.menuBar().addMenu("个人笔迹")
        self.training_action = self._action("训练与模型管理…", self._open_training)
        learning_menu.addAction(self.training_action)
        help_menu = self.menuBar().addMenu("帮助")
        help_menu.addAction(self._action("使用说明", self._show_help))
        help_menu.addAction(self._action("打开日志文件夹", self._open_logs))

    def _button(self, text, callback, name=None, tooltip=None):
        button = QPushButton(text)
        if name:
            button.setObjectName(name)
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(25, 16, 25, 16)
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(2)
        brand = QLabel("墨迹  Handwriter")
        brand.setObjectName("brand")
        brand_layout.addWidget(brand)
        subtitle = QLabel("英文手稿，从纸面到可编辑文字")
        subtitle.setObjectName("brandSub")
        brand_layout.addWidget(subtitle)
        header_layout.addLayout(brand_layout)
        header_layout.addStretch()
        badge = QLabel("本机处理  ·  TrOCR")
        badge.setObjectName("localBadge")
        header_layout.addWidget(badge)
        root_layout.addWidget(header)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 18, 20, 12)
        body_layout.setSpacing(13)
        command_row = QHBoxLayout()
        self.import_button = self._button("＋ 导入 PDF", self._import_pdf, "importButton")
        self.open_button = self._button("打开任务", self._open_project, "openProjectButton")
        self.start_button = self._button("开始 / 继续识别", self._start_recognition, "primary")
        self.start_button.setToolTip("已完成页面会跳过，继续处理待识别或已修改的文字行。")
        self.cancel_button = self._button("取消处理", self._cancel_job, "cancelButton")
        self.word_button = self._button("导出 Word", lambda: self._export("docx"), "exportWordButton")
        self.txt_button = self._button("导出 TXT", lambda: self._export("txt"), "exportTxtButton")
        self.training_button = self._button("个人笔迹训练", self._open_training, "trainingButton")
        for button in [self.import_button, self.open_button, self.start_button, self.cancel_button]:
            command_row.addWidget(button)
        command_row.addStretch()
        command_row.addWidget(self.training_button)
        command_row.addWidget(self.word_button)
        command_row.addWidget(self.txt_button)
        body_layout.addLayout(command_row)
        project_row = QHBoxLayout()
        self.project_title = QLabel("导入一份手写 PDF，开始转写")
        self.project_title.setObjectName("projectTitle")
        project_row.addWidget(self.project_title)
        project_row.addStretch()
        self.summary_label = QLabel("支持单栏英文手稿 · PDF → Word / TXT")
        self.summary_label.setObjectName("muted")
        project_row.addWidget(self.summary_label)
        body_layout.addLayout(project_row)
        recognizer_row = QHBoxLayout()
        recognizer_row.addWidget(QLabel("后续识别使用："))
        self.recognizer_combo = QComboBox()
        self.recognizer_combo.setObjectName("recognizerCombo")
        self.recognizer_combo.setMinimumWidth(230)
        self.recognizer_combo.currentIndexChanged.connect(self._recognizer_changed)
        recognizer_row.addWidget(self.recognizer_combo)
        self.writer_label = QLabel("当前任务：未归入作者")
        self.writer_label.setObjectName("muted")
        recognizer_row.addSpacing(15)
        recognizer_row.addWidget(self.writer_label)
        recognizer_row.addStretch()
        recognizer_hint = QLabel("切换模型保留已有识别文字和人工校对")
        recognizer_hint.setObjectName("muted")
        recognizer_row.addWidget(recognizer_hint)
        body_layout.addLayout(recognizer_row)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(13, 15, 13, 12)
        title = QLabel("文档页面")
        title.setObjectName("sectionTitle")
        sidebar_layout.addWidget(title)
        self.page_list = QListWidget()
        self.page_list.setObjectName("pageList")
        self.page_list.currentRowChanged.connect(self._show_page)
        sidebar_layout.addWidget(self.page_list, 1)
        model_title = QLabel("本地模型")
        model_title.setObjectName("sectionTitle")
        sidebar_layout.addWidget(model_title)
        self.model_label = QLabel()
        self.model_label.setObjectName("muted")
        self.model_label.setWordWrap(True)
        sidebar_layout.addWidget(self.model_label)
        self.prepare_button = self._button("准备 / 检查模型", self._prepare_models, "prepareModelsButton")
        sidebar_layout.addWidget(self.prepare_button)
        privacy = QLabel("文稿始终留在本机。\n仅首次下载模型需要联网。")
        privacy.setObjectName("muted")
        privacy.setWordWrap(True)
        sidebar_layout.addWidget(privacy)
        splitter.addWidget(sidebar)
        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)
        controls = QHBoxLayout()
        controls.setSpacing(5)
        self.add_button = self._button("新增行", self._add_mode, "addLineButton")
        self.delete_button = self._button("删除", self._delete_lines, "deleteLineButton")
        self.split_button = self._button("拆分", self._split_line, "splitLineButton", "将选中行分成上下两行，再拖动边缘调整。")
        split_menu = QMenu(self.split_button)
        split_menu.addAction("拆成上下两行").triggered.connect(lambda: self._split_line("vertical"))
        split_menu.addAction("长行拆成左右两段").triggered.connect(lambda: self._split_line("horizontal"))
        self.split_button.setMenu(split_menu)
        self.merge_button = self._button("合并", self._merge_lines, "mergeLinesButton", "按 Ctrl 多选行框后合并。")
        self.rotate_button = self._button("旋转 90°", self._rotate_page, "rotatePageButton")
        for button in [self.add_button, self.delete_button, self.split_button, self.merge_button, self.rotate_button]:
            controls.addWidget(button)
        controls.addStretch()
        controls.addWidget(self._button("−", lambda: self.canvas.zoom(1 / 1.2), tooltip="缩小；Ctrl + 滚轮也可缩放"))
        controls.addWidget(self._button("＋", lambda: self.canvas.zoom(1.2), tooltip="放大"))
        controls.addWidget(self._button("适应", lambda: self.canvas.fit_page(), tooltip="显示整页"))
        viewer_layout.addLayout(controls)
        self.viewer_stack = QStackedWidget()
        self.empty_page = QWidget()
        empty_layout = QVBoxLayout(self.empty_page)
        empty_layout.addStretch()
        empty_title = QLabel("让手写稿重新可读")
        empty_title.setObjectName("emptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_sub = QLabel("拖入 PDF，或点击左上方「导入 PDF」\n自动切行 · 本地识别 · 对照校对")
        empty_sub.setObjectName("emptySub")
        empty_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_sub.setStyleSheet("line-height: 1.8;")
        empty_layout.addWidget(empty_sub)
        empty_layout.addStretch()
        self.viewer_stack.addWidget(self.empty_page)
        self.canvas = PageCanvas()
        self.canvas.lines_selected.connect(self._canvas_selection)
        self.canvas.line_rect_changed.connect(self._change_line_box)
        self.canvas.line_rects_changed.connect(self._change_line_boxes)
        self.canvas.new_rect_requested.connect(self._new_line)
        self.viewer_stack.addWidget(self.canvas)
        viewer_layout.addWidget(self.viewer_stack, 1)
        self.page_warning = QLabel()
        self.page_warning.setWordWrap(True)
        self.page_warning.setObjectName("warning")
        self.page_warning.hide()
        viewer_layout.addWidget(self.page_warning)
        hint = QLabel("拖动行框移动，选中后拖动边缘调整 · Ctrl 多选 · Ctrl + 滚轮缩放")
        hint.setObjectName("muted")
        viewer_layout.addWidget(hint)
        splitter.addWidget(viewer)
        review = QWidget()
        review.setObjectName("reviewPane")
        review_layout = QVBoxLayout(review)
        review_layout.setContentsMargins(13, 14, 13, 12)
        review_title = QHBoxLayout()
        label = QLabel("对照与校对")
        label.setObjectName("sectionTitle")
        review_title.addWidget(label)
        review_title.addStretch()
        self.confirm_page_button = self._button("确认本页", self._confirm_page, "confirmPageButton", "明确确认本页每一条非排除、已完成的文字行。不会把单纯编辑自动视为确认。")
        review_title.addWidget(self.confirm_page_button)
        self.recognize_selected_button = self._button("重识别选中行", self._recognize_selected, "recognizeSelectedButton")
        review_title.addWidget(self.recognize_selected_button)
        review_layout.addLayout(review_title)
        self.line_count_label = QLabel("识别结果将显示在这里")
        self.line_count_label.setObjectName("muted")
        review_layout.addWidget(self.line_count_label)
        self.review_scroll = QScrollArea()
        self.review_scroll.setWidgetResizable(True)
        self.review_content = QWidget()
        self.review_content.setStyleSheet("background: transparent;")
        self.review_layout = QVBoxLayout(self.review_content)
        self.review_layout.setContentsMargins(0, 0, 3, 0)
        self.review_layout.setSpacing(10)
        self.review_layout.addStretch()
        self.review_scroll.setWidget(self.review_content)
        review_layout.addWidget(self.review_scroll, 1)
        autosave = QLabel("文字修改自动保存；导出采用校对后的内容。")
        autosave.setObjectName("muted")
        review_layout.addWidget(autosave)
        splitter.addWidget(review)
        splitter.setSizes([185, 700, 430])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        body_layout.addWidget(splitter, 1)
        self.progress_label = QLabel("就绪。识别效果取决于笔迹与扫描质量，请对照原稿校对。")
        self.progress_label.setObjectName("muted")
        self.progress_label.setWordWrap(True)
        body_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        body_layout.addWidget(self.progress_bar)
        root_layout.addWidget(body, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("本地工作区就绪")

    def _update_model_badge(self):
        try:
            from .model_manager import models_ready
            ready = models_ready(paths.models_dir())
            self.model_label.setText("模型已就绪\n可完全离线识别" if ready else "首次使用需要下载\nTrOCR + 文字行检测模型")
        except Exception:
            self.model_label.setText("模型将在首次识别时准备")

    def _refresh_writer_label(self):
        from .learning import load_writer
        writer = load_writer(paths.app_home())
        owned = self.project is not None and writer and getattr(self.project, "writer_id", "") == writer["id"]
        self.writer_label.setText(f"当前任务作者：{writer['name']}" if owned else "当前任务：未归入作者（训练页可关联）")

    def _refresh_recognizer_choices(self, selected=None):
        from .learning import get_active_model, list_models
        selected = selected if selected is not None else (self.recognizer_combo.currentData() or get_active_model(paths.app_home()))
        self.recognizer_combo.blockSignals(True)
        self.recognizer_combo.clear()
        for item in list_models(paths.app_home()):
            if item.get("available", True):
                self.recognizer_combo.addItem(item.get("title", item["id"]), item["id"])
        if self.recognizer_combo.count() == 0:
            self.recognizer_combo.addItem("通用 TrOCR", "base")
        index = self.recognizer_combo.findData(selected)
        self.recognizer_combo.setCurrentIndex(max(0, index))
        self.recognizer_combo.blockSignals(False)

    def _recognizer_changed(self, index):
        if self._busy or self._training_busy or index < 0:
            return
        model_id = self.recognizer_combo.itemData(index) or "base"
        from .learning import set_active_model
        try:
            set_active_model(paths.app_home(), model_id)
            if self.project is not None:
                self.project.recognizer_id = model_id
                self._save_project(quiet=True)
            self.statusBar().showMessage("识别模型已切换，仅影响后续识别；已有文字和校对保留。", 10000)
        except Exception as exc:
            self._error("无法切换模型", str(exc))
            self._refresh_recognizer_choices("base")

    def _training_state_changed(self, busy):
        self._training_busy = bool(busy)
        self._refresh_actions()

    def _apply_active_model(self, model_id):
        self._refresh_recognizer_choices(model_id)
        if self.project is not None:
            self.project.recognizer_id = model_id
            self._save_project(quiet=True)

    def _open_training(self):
        if self._busy or self._training_busy:
            return
        self._save_project(quiet=True)
        from .training_ui import TrainingDialog
        dialog = TrainingDialog(paths.app_home(), paths.models_dir(), self.project, self)
        self._training_dialog = dialog
        dialog.busy_changed.connect(self._training_state_changed)
        dialog.models_changed.connect(lambda: self._refresh_recognizer_choices())
        dialog.active_model_changed.connect(self._apply_active_model)
        dialog.project_changed.connect(self._training_project_changed)
        self._training_state_changed(dialog.busy)
        dialog.exec()
        self._training_dialog = None
        self._training_busy = False
        self._refresh_writer_label()
        self._build_line_editors()
        self._refresh_actions()

    def _training_project_changed(self):
        self._refresh_writer_label()
        self._build_line_editors()
        self._refresh_actions()

    def _refresh_review_card(self, page, line):
        from .learning import is_line_reviewed
        card = self.line_editors.get(line.id)
        if card is not None:
            card.update_status(line)
            card.review_checkbox.blockSignals(True)
            card.review_checkbox.setChecked(is_line_reviewed(self.project, page, line))
            card.review_checkbox.blockSignals(False)

    def _line_review_changed(self, line_id, confirmed):
        page = self._page()
        if page is None or self._busy or self._training_busy:
            return
        from .learning import set_line_review
        line = next((line for line in page.lines if line.id == line_id), None)
        if line is None:
            return
        try:
            set_line_review(self.project, page, line, bool(confirmed))
            self._save_project(quiet=True)
            self.statusBar().showMessage("训练确认已保存。" if confirmed else "已撤销此行的训练确认。", 6000)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 10000)
        self._refresh_review_card(page, line)

    def _line_exclusion_changed(self, line_id, excluded):
        page = self._page()
        if page is None or self._busy or self._training_busy:
            return
        from .learning import invalidate_line_review
        line = next((line for line in page.lines if line.id == line_id), None)
        if line is not None:
            invalidate_line_review(line)
            line.training_excluded = bool(excluded)
            self._refresh_review_card(page, line)
            self._save_project(quiet=True)
            self._refresh_actions()

    def _confirm_page(self):
        page = self._page()
        if page is None or self._busy or self._training_busy:
            return
        from .learning import set_line_review
        confirmed, skipped = 0, 0
        for line in page.lines:
            if getattr(line, "training_excluded", False):
                skipped += 1
                continue
            try:
                set_line_review(self.project, page, line, True)
                confirmed += 1
            except ValueError:
                skipped += 1
            self._refresh_review_card(page, line)
        self._save_project(quiet=True)
        self.statusBar().showMessage(f"本页已确认 {confirmed} 行；跳过 {skipped} 行（排除、未完成或无有效文字）。", 12000)

    def _refresh_actions(self):
        available = self.project is not None
        active_page = available and self.current_page_index >= 0
        writable = not (self._busy or self._training_busy)
        self.training_button.setEnabled(writable)
        self.training_action.setEnabled(writable)
        self.recognizer_combo.setEnabled(writable)
        self.confirm_page_button.setEnabled(active_page and writable and bool(getattr(self.project, "writer_id", "")))
        for widget in [self.import_button, self.open_button, self.prepare_button]:
            widget.setEnabled(writable)
        for action in [self.import_action, self.open_action]:
            action.setEnabled(writable)
        self.start_button.setEnabled(available and writable)
        self.cancel_button.setEnabled(self._busy)
        for widget in [self.word_button, self.txt_button]:
            widget.setEnabled(available and writable)
        for action in [self.export_word_action, self.export_txt_action, self.save_action]:
            action.setEnabled(available and writable)
        for widget in [self.add_button, self.rotate_button]:
            widget.setEnabled(active_page and writable and self.current_image is not None)
        self.delete_button.setEnabled(active_page and writable and bool(self._selected_ids))
        self.split_button.setEnabled(active_page and writable and len(self._selected_ids) == 1)
        self.merge_button.setEnabled(active_page and writable and len(self._selected_ids) >= 2)
        self.recognize_selected_button.setEnabled(active_page and writable and bool(self._selected_ids))
        self.canvas.editable = writable
        for editor in self.line_editors.values():
            editor.editor.setReadOnly(not writable)
            editor.review_checkbox.setEnabled(writable and bool(getattr(self.project, "writer_id", "")) and not editor.exclude_checkbox.isChecked())
            editor.exclude_checkbox.setEnabled(writable)

    def _set_busy(self, busy):
        self._busy = busy
        if busy:
            self.canvas.set_add_mode(False)
        self._refresh_actions()
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)

    def _import_pdf(self):
        if self._busy:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "导入英文手写扫描稿", "", "PDF 文件 (*.pdf)")
        if filename:
            self.import_pdf(filename)

    def import_pdf(self, filename):
        """Public import entrypoint, also used by drag-and-drop and smoke tests."""
        if self._busy:
            return
        self._save_project(quiet=True)
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            project = create_project(str(filename), paths.projects_dir())
            from .learning import get_active_model
            project.recognizer_id = get_active_model(paths.app_home())
            self._set_project(project)
            self.statusBar().showMessage("已创建本地任务。点击「开始 / 继续识别」进行自动切行与识别。", 12000)
        except Exception as exc:
            logger.exception("Import failed")
            self._error("无法导入 PDF", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _open_project(self):
        if self._busy:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "打开本地识别任务", str(paths.projects_dir()), "任务文件 (project.json);;JSON 文件 (*.json)")
        if filename:
            self.open_project(filename)

    def open_project(self, filename):
        self._save_project(quiet=True)
        try:
            self._set_project(load_project(filename))
            self.statusBar().showMessage("任务已恢复，校对内容已保留。", 8000)
        except Exception as exc:
            logger.exception("Open project failed")
            self._error("无法打开任务", str(exc))

    def _set_project(self, project):
        self.canvas.set_add_mode(False)
        self.project = project
        self.current_page_index = -1
        self.current_image = None
        self._selected_ids = []
        self._dirty = False
        self.project_title.setText(project.title)
        self._refresh_recognizer_choices(getattr(project, "recognizer_id", "base"))
        self._refresh_writer_label()
        self._refresh_page_list()
        if project.pages:
            self.page_list.blockSignals(True)
            self.page_list.setCurrentRow(0)
            self.page_list.blockSignals(False)
            self._show_page(0)
        self._refresh_actions()

    def _refresh_page_list(self):
        if self.project is None:
            return
        selected = self.page_list.currentRow()
        self.page_list.blockSignals(True)
        self.page_list.clear()
        for page in self.project.pages:
            item = QListWidgetItem(f"第 {page.index + 1:02d} 页\n{STATUS.get(page.status, page.status)}" + (f" · {len(page.lines)} 行" if page.lines else ""))
            item.setData(Qt.ItemDataRole.UserRole, page.index)
            if page.error:
                item.setToolTip(page.error)
            self.page_list.addItem(item)
        if 0 <= selected < self.page_list.count():
            self.page_list.setCurrentRow(selected)
        self.page_list.blockSignals(False)
        finished = sum(page.status in {"recognized", "review"} for page in self.project.pages)
        self.summary_label.setText(f"共 {len(self.project.pages)} 页  ·  已处理 {finished} 页")

    def _show_page(self, index):
        if self.project is None or not 0 <= index < len(self.project.pages):
            return
        self.current_page_index = index
        self._selected_ids = []
        page = self.project.pages[index]
        self.current_image = None
        try:
            image_path = Path(self.project.work_dir) / page.image_path if page.image_path else None
            if self._busy and (image_path is None or not image_path.is_file()):
                self.viewer_stack.setCurrentWidget(self.empty_page)
                self._build_line_editors()
                self._refresh_actions()
                return
            self.current_image = load_page_image(self.project, page)
            self.canvas.set_page(self.current_image, page.lines)
            self.viewer_stack.setCurrentWidget(self.canvas)
            if not self._busy:
                save_project(self.project)
        except Exception as exc:
            logger.exception("Page preview failed")
            self.progress_label.setText(f"第 {index + 1} 页预览失败：{exc}")
            self.viewer_stack.setCurrentWidget(self.empty_page)
        self._build_line_editors()
        warnings = list(page.warnings)
        if page.error:
            warnings.insert(0, "本页处理失败：" + page.error)
        self.page_warning.setText("；".join(warnings))
        self.page_warning.setVisible(bool(warnings))
        self._refresh_actions()

    def _build_line_editors(self):
        while self.review_layout.count():
            item = self.review_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self.line_editors = {}
        if self.project is None or self.current_page_index < 0:
            self.review_layout.addStretch()
            return
        page = self.project.pages[self.current_page_index]
        self.line_count_label.setText(f"第 {page.index + 1} 页 · {len(page.lines)} 行" if page.lines else "尚无文字行。识别后将显示行图与文字。")
        for number, line in enumerate(page.lines, 1):
            card = LineEditor(line, number, self.current_image)
            card.text_changed.connect(self._line_text_changed)
            card.focused.connect(self._focus_line)
            card.review_changed.connect(self._line_review_changed)
            card.exclusion_changed.connect(self._line_exclusion_changed)
            from .learning import is_line_reviewed
            card.review_checkbox.blockSignals(True)
            card.review_checkbox.setChecked(is_line_reviewed(self.project, page, line))
            card.review_checkbox.blockSignals(False)
            card.editor.setReadOnly(self._busy or self._training_busy)
            self.review_layout.addWidget(card)
            self.line_editors[line.id] = card
        self.review_layout.addStretch()

    def _page(self):
        if self.project is not None and 0 <= self.current_page_index < len(self.project.pages):
            return self.project.pages[self.current_page_index]
        return None

    def _line_text_changed(self, line_id, text):
        page = self._page()
        if page is None or self._busy or self._training_busy:
            return
        for line in page.lines:
            if line.id == line_id:
                from .learning import invalidate_line_review
                invalidate_line_review(line)
                line.edited_text = text
                if line.status in {"stale", "pending", "error"}:
                    line.status = "recognized"
                    line.warnings = ["此行为人工转写，请对照原稿检查。"]
                self.line_editors[line_id].update_status(line)
                break
        _update_page_status(page)
        page.error = ""
        self._dirty = True
        self._save_timer.start()
        self.statusBar().showMessage("正在自动保存校对内容……")
        self._refresh_page_list()

    def _focus_line(self, line_id):
        self._selected_ids = [line_id]
        self.canvas.select_lines([line_id])
        for current_id, card in self.line_editors.items():
            card.set_selected(current_id == line_id)
        self._refresh_actions()

    def _canvas_selection(self, selected):
        self._selected_ids = selected
        for line_id, card in self.line_editors.items():
            card.set_selected(line_id in selected)
        if len(selected) == 1 and selected[0] in self.line_editors:
            self.review_scroll.ensureWidgetVisible(self.line_editors[selected[0]], 0, 40)
        self._refresh_actions()

    def _confirm_discard(self, lines, operation="调整行框"):
        if not any(line.edited_text is not None for line in lines):
            return True
        answer = QMessageBox.question(self, "保留已有校对", f"{operation}会使这些行的校对文字与原图不再对应，需要清除这些行的文字并重新识别。\n\n是否继续？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def _stale(self, line):
        from .learning import invalidate_line_review
        invalidate_line_review(line)
        line.raw_text = ""
        line.edited_text = None
        line.status = "stale"
        line.warnings = ["行框已修改，请重新识别或输入人工转写。"]

    def _changed_lines(self, selected=()):
        page = self._page()
        if page is None:
            return
        _sort_lines(page.lines)
        _update_page_status(page)
        page.error = ""
        if page.lines:
            page.warnings = []
        self.canvas.update_lines(page.lines, selected)
        self._selected_ids = list(selected)
        self._build_line_editors()
        for line_id in selected:
            if line_id in self.line_editors:
                self.line_editors[line_id].set_selected(True)
        self._save_project(quiet=True)
        self._refresh_page_list()
        self._refresh_actions()

    def _change_line_box(self, line_id, polygon):
        self._change_line_boxes([(line_id, polygon)])

    def _change_line_boxes(self, changes):
        page = self._page()
        if page is None or self._busy:
            return
        polygons = dict(changes)
        lines = [line for line in page.lines if line.id in polygons]
        if not lines:
            return
        selected = [line.id for line in lines]
        if not self._confirm_discard(lines):
            self.canvas.update_lines(page.lines, selected)
            return
        for line in lines:
            line.polygon = polygons[line.id]
            self._stale(line)
        self._changed_lines(selected)

    def _add_mode(self):
        self.canvas.set_add_mode(True)
        self.progress_label.setText("在原图上拖出一个矩形，覆盖完整的一行文字。")

    def _new_line(self, polygon):
        page = self._page()
        if page is None or self._busy:
            return
        line = LineResult(id=uuid.uuid4().hex, polygon=polygon)
        self._stale(line)
        page.lines.append(line)
        self._changed_lines([line.id])

    def _delete_lines(self):
        page = self._page()
        if page is None or self._busy or not self._selected_ids:
            return
        selected = [line for line in page.lines if line.id in self._selected_ids]
        if not self._confirm_discard(selected, "删除行框"):
            return
        from .learning import invalidate_line_review
        for line in selected:
            invalidate_line_review(line)
        page.lines = [line for line in page.lines if line.id not in self._selected_ids]
        self._changed_lines()

    def _split_line(self, direction="vertical"):
        page = self._page()
        if page is None or self._busy or len(self._selected_ids) != 1:
            return
        line = next(item for item in page.lines if item.id == self._selected_ids[0])
        if not self._confirm_discard([line], "拆分行框"):
            return
        rect = bounds(line.polygon)
        if direction != "horizontal" and rect.height() < 18:
            self.statusBar().showMessage("行框太矮，请先放大行框，再拆成上下两行。", 8000)
            return
        if direction == "horizontal":
            upper = QRectF(rect.x(), rect.y(), rect.width() / 2, rect.height())
            lower = QRectF(rect.x() + rect.width() / 2, rect.y(), rect.width() / 2, rect.height())
        else:
            upper = QRectF(rect.x(), rect.y(), rect.width(), rect.height() / 2)
            lower = QRectF(rect.x(), rect.y() + rect.height() / 2, rect.width(), rect.height() / 2)
        new_lines = [LineResult(id=uuid.uuid4().hex, polygon=rectangle_polygon(item)) for item in [upper, lower]]
        for item in new_lines:
            self._stale(item)
        from .learning import invalidate_line_review
        invalidate_line_review(line)
        page.lines.remove(line)
        page.lines.extend(new_lines)
        self._changed_lines([item.id for item in new_lines])

    def _merge_lines(self):
        page = self._page()
        if page is None or self._busy or len(self._selected_ids) < 2:
            return
        selected = [line for line in page.lines if line.id in self._selected_ids]
        if not self._confirm_discard(selected, "合并行框"):
            return
        rect = bounds(selected[0].polygon)
        for line in selected[1:]:
            rect = rect.united(bounds(line.polygon))
        merged = LineResult(id=uuid.uuid4().hex, polygon=rectangle_polygon(rect))
        self._stale(merged)
        from .learning import invalidate_line_review
        for line in selected:
            invalidate_line_review(line)
        page.lines = [line for line in page.lines if line not in selected] + [merged]
        self._changed_lines([merged.id])

    def _rotate_page(self):
        page = self._page()
        if page is None or self._busy:
            return
        if page.lines:
            answer = QMessageBox.question(self, "旋转当前页面", "旋转会清除本页的行框和识别文字，以便按新方向重新切行。其他页面保留。\n\n是否继续？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        page.rotation = (page.rotation + 90) % 360
        from .learning import invalidate_line_review
        for line in page.lines:
            invalidate_line_review(line)
        page.image_path = ""
        page.skew_angle = 0
        page.lines = []
        page.status = "pending"
        page.error = ""
        page.warnings = []
        self._show_page(page.index)
        self._save_project(quiet=True)
        self._refresh_page_list()

    def _prepare_models(self):
        self._start_job("prepare")

    def _start_recognition(self):
        if self.project:
            self._start_job("recognize")

    def _recognize_selected(self):
        if self.project and self._selected_ids:
            self._start_job("selected", self.current_page_index, list(self._selected_ids))

    def _start_job(self, mode, page_index=None, line_ids=None):
        if self._busy or self._training_busy:
            return
        from filelock import FileLock, Timeout
        try:
            lock = FileLock(str(paths.app_home() / "training.lock"))
            lock.acquire(timeout=0)
            lock.release()
        except Timeout:
            self._error("训练正在运行", "另一训练进程仍占用设备，请等待训练完成或在训练页请求取消后再识别。")
            return
        self._save_project(quiet=True)
        self._job_error = None
        self._set_busy(True)
        self.progress_label.setText("正在启动后台任务……")
        self._thread = QThread(self)
        recognizer_id = self.recognizer_combo.currentData() or "base"
        if self.project is not None:
            self.project.recognizer_id = recognizer_id
            save_project(self.project)
        self._worker = TaskWorker(self.project, mode, page_index, line_ids, recognizer_id)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._job_progress)
        self._worker.page_done.connect(self._page_done)
        self._worker.counts.connect(self._job_counts)
        self._worker.failed.connect(self._job_failed)
        self._worker.finished.connect(self._job_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    def _job_progress(self, text):
        self.progress_label.setText(text)
        self.statusBar().showMessage(text)

    def _job_counts(self, done, total):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)

    def _page_done(self, index):
        self._refresh_page_list()
        if index == self.current_page_index:
            self._show_page(index)

    def _job_failed(self, message):
        self._job_error = message

    def _job_finished(self, message):
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot()
    def _thread_finished(self):
        thread = self._thread
        if thread is not None:
            thread.quit()
            thread.wait()
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self._set_busy(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self._update_model_badge()
        self._refresh_page_list()
        if self.current_page_index >= 0:
            self._show_page(self.current_page_index)
        if self._job_error:
            self._error("任务未完成", self._job_error)
        if self._close_after_task:
            self.close()

    def _cancel_job(self):
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("正在取消，将保存已完成的结果……")

    def _save_project(self, checked=False, quiet=False):
        self._save_timer.stop()
        if self.project is None or self._busy:
            return
        try:
            save_project(self.project)
            self._dirty = False
            if not quiet:
                self.statusBar().showMessage("任务已保存 · " + str(Path(self.project.work_dir) / "project.json"), 7000)
        except Exception as exc:
            logger.exception("Project save failed")
            self._error("保存失败", str(exc))

    def _export(self, kind):
        if self.project is None or self._busy:
            return
        self._save_project(quiet=True)
        try:
            from .exporter import validate_export
            validate_export(self.project)
        except Exception as exc:
            self._error("暂时无法导出", str(exc))
            return
        reviews = sum(line.status == "review" for page in self.project.pages for line in page.lines)
        page_reviews = sum(bool(page.warnings) for page in self.project.pages)
        if reviews or page_reviews:
            answer = QMessageBox.question(self, "导出前复核", f"文档中有 {reviews} 行待复核文字，{page_reviews} 页带提示。\n\n可以先检查原稿，也可以继续导出当前内容。", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        extension = ".docx" if kind == "docx" else ".txt"
        label = "Word 文档 (*.docx)" if kind == "docx" else "文本文件 (*.txt)"
        default = Path(self.project.work_dir) / (self.project.title + extension)
        filename, _ = QFileDialog.getSaveFileName(self, "导出校对后的文字", str(default), label)
        if not filename:
            return
        if not filename.lower().endswith(extension):
            filename += extension
        try:
            (export_docx if kind == "docx" else export_txt)(self.project, filename)
            self.statusBar().showMessage("已导出：" + filename, 15000)
            self.progress_label.setText("导出完成：" + filename)
        except Exception as exc:
            logger.exception("Export failed")
            self._error("无法导出", str(exc))

    def _error(self, title, text):
        QMessageBox.critical(self, title, text + "\n\n详细诊断保存在本地日志中，可通过「帮助 → 打开日志文件夹」查看。")

    def _show_help(self):
        QMessageBox.information(self, "使用说明", "1. 导入单栏英文手写扫描 PDF。\n2. 点击「开始 / 继续识别」，首次使用会下载模型。\n3. 对照原图校对右侧文字，修改会自动保存。\n4. 切错行时可新增、拖动调整、拆分或合并行框，再重识别选中行。\n5. 导出 Word 或 TXT。\n\n模型准备后可断网使用。识别结果需要人工校对；第一版优先支持单栏英文手稿。\n任务可通过「文件 → 打开已保存任务」恢复。\n\nCtrl + 滚轮缩放；Ctrl + 单击可多选行框。")

    def _open_logs(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.logs_dir())))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if not self._busy and event.mimeData().hasUrls() and any(url.toLocalFile().lower().endswith(".pdf") for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if self._busy:
            return
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".pdf"):
                self.import_pdf(path)
                event.acceptProposedAction()
                break

    def closeEvent(self, event: QCloseEvent):
        if self._training_busy:
            event.ignore()
            return
        if self._busy:
            answer = QMessageBox.question(self, "处理仍在进行", "关闭前需要停止当前任务并保存已完成结果。是否取消处理并关闭？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Yes:
                self._close_after_task = True
                self._cancel_job()
            event.ignore()
            return
        self._save_project(quiet=True)
        event.accept()


def create_window():
    return MainWindow()
