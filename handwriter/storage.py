from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from .data import DocumentProject, PageResult, utc_now


def project_path(project: DocumentProject, relative_path: str) -> Path:
    root = Path(project.work_dir).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("任务文件路径超出任务文件夹。")
    return path


def create_project(pdf_path: str | Path, projects_dir: Path) -> DocumentProject:
    from .pdfio import inspect_pdf

    source = Path(pdf_path).resolve()
    count = inspect_pdf(source)
    project_id = uuid4().hex
    directory = Path(projects_dir).resolve() / project_id
    directory.mkdir(parents=True)
    (directory / "pages").mkdir()
    shutil.copy2(source, directory / "source.pdf")
    project = DocumentProject(
        id=project_id, title=source.stem, source_pdf="source.pdf",
        work_dir=str(directory), pages=[PageResult(index=i) for i in range(count)],
    )
    save_project(project)
    return project


def save_project(project: DocumentProject) -> Path:
    directory = Path(project.work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    project.updated_at = utc_now()
    destination = directory / "project.json"
    # Preserve the exact original bytes before the first v1 -> v2 save. Opening
    # a v1 task is read-only and does not migrate or back it up prematurely.
    if destination.is_file():
        try:
            previous_version = json.loads(destination.read_text(encoding="utf-8")).get("schema_version", 1)
        except (OSError, ValueError, AttributeError):
            previous_version = None
        backup = directory / "project.v1.backup.json"
        if previous_version == 1 and not backup.exists():
            shutil.copy2(destination, backup)
    project.schema_version = 2
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_project(path: str | Path) -> DocumentProject:
    location = Path(path).resolve()
    if location.is_dir():
        location = location / "project.json"
    if location.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("任务记录过大，无法打开。")
    try:
        project = DocumentProject.from_dict(json.loads(location.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError("任务记录已损坏或不是 Handwriter 任务。") from exc
    project.work_dir = str(location.parent)
    if not project_path(project, project.source_pdf).is_file():
        raise ValueError("任务的 source.pdf 不存在，请保留完整任务文件夹。")
    for page in project.pages:
        if page.image_path:
            project_path(project, page.image_path)
    return project
