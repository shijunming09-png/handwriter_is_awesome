"""Desktop entrypoint, also used by the portable executable."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path


def main():
    from . import paths
    log_dir = paths.logs_dir()
    handler = logging.handlers.RotatingFileHandler(log_dir / "handwriter.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    logger = logging.getLogger("handwriter")
    logger.info("Starting Handwriter, frozen=%s, home=%s", getattr(sys, "frozen", False), paths.app_home())
    from PySide6.QtCore import QObject, QTimer, Signal
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox
    from .gui import MainWindow
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Handwriter")
    app.setApplicationDisplayName("墨迹 Handwriter")
    app.setOrganizationName("Handwriter Local")
    app.setStyle("Fusion")
    resources = Path(getattr(sys, "_MEIPASS", paths.app_home()))
    icon_path = resources / "assets" / "handwriter.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    class ErrorBridge(QObject):
        message = Signal(str)

    error_bridge = ErrorBridge(app)
    error_bridge.message.connect(lambda message: QMessageBox.critical(None, "应用发生错误", f"{message}\n\n已保存到本地日志：{log_dir / 'handwriter.log'}"))

    def exception_hook(exc_type, value, traceback):
        logger.critical("Unhandled exception", exc_info=(exc_type, value, traceback))
        error_bridge.message.emit(str(value))

    def thread_exception_hook(args):
        logger.critical("Unhandled thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        error_bridge.message.emit(str(args.exc_value))

    sys.excepthook = exception_hook
    threading.excepthook = thread_exception_hook
    window = MainWindow()
    window.show()
    for argument in sys.argv[1:]:
        if argument.lower().endswith(".pdf") and Path(argument).is_file():
            QTimer.singleShot(0, lambda filename=argument: window.import_pdf(filename))
            break
        if argument.lower().endswith(".json") and Path(argument).is_file():
            QTimer.singleShot(0, lambda filename=argument: window.open_project(filename))
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
