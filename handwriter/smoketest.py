from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
import traceback


def run_smoke_test(pdf: Path, output_dir: Path, model_path: Path | None = None,
                   recognizer_dir: Path | None = None) -> int:
    # Never download or make API calls during this verification path.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .engine import OcrEngine
    from .exporter import export_docx, export_txt
    from .paths import models_dir
    from .pdfio import load_page_image
    from .storage import create_project, save_project

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"offline": True, "input": str(pdf), "pages": [], "success": False}
    report["network_attempts"] = []
    original_connect = socket.socket.connect

    def deny_network(sock, address):
        report["network_attempts"].append(str(address))
        raise OSError("Network is disabled for the offline acceptance test")

    socket.socket.connect = deny_network
    started = time.monotonic()

    def progress(message: str) -> None:
        with (output_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")

    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from .gui import MainWindow
        from .training_ui import TrainingDialog
        report["qt_version"] = QtCore.qVersion()
        report["gui_libraries_loaded"] = bool(QtGui.QImage and QtWidgets.QApplication)
        report["desktop_classes_imported"] = bool(MainWindow and TrainingDialog)
        project = create_project(pdf, output_dir / "projects")
        engine = OcrEngine(model_path or models_dir(), progress=progress, recognizer_dir=recognizer_dir)
        report["recognizer"] = str(recognizer_dir) if recognizer_dir else "base"
        report["device"] = engine.load()
        for page in project.pages:
            image = load_page_image(project, page)
            page.lines = engine.detect(image)
            page.status = "detected"
            if page.lines:
                engine.recognize(image, page.lines)
            page.status = "review" if any(line.status != "recognized" for line in page.lines) else "recognized"
            save_project(project)
            report["pages"].append({
                "index": page.index,
                "size": [page.width, page.height],
                "skew_angle": page.skew_angle,
                "lines": [{"text": line.text, "polygon": line.polygon,
                           "status": line.status, "warnings": line.warnings}
                          for line in page.lines],
            })
        report["docx"] = str(export_docx(project, output_dir / "recognized.docx"))
        report["txt"] = str(export_txt(project, output_dir / "recognized.txt"))
        report["project"] = str(Path(project.work_dir) / "project.json")
        report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        socket.socket.connect = original_connect
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["success"] else 1
