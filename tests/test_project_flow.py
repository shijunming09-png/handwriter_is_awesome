from __future__ import annotations

from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops
import pytest

from handwriter.data import LineResult
from handwriter.storage import create_project, save_project, load_project
from handwriter.pdfio import render_page, load_page_image
from handwriter.exporter import export_docx, export_txt


def _line(identifier: str, text: str, top: int = 20, edited: str | None = None):
    return LineResult(
        id=identifier,
        polygon=[[10, top], [500, top], [500, top + 60], [10, top + 60]],
        raw_text=text,
        edited_text=edited,
        status="recognized",
    )


def test_project_copies_source_and_survives_move(fixture_dir, tmp_path):
    source = tmp_path / "原始文件.pdf"
    shutil.copy2(fixture_dir / "two_pages.pdf", source)
    project = create_project(str(source), tmp_path / "项目")
    assert len(project.pages) == 2
    project.pages[0].lines = [_line("line-1", "machine draft", edited="Edited by a person.")]
    save_project(project)
    source.unlink()
    moved = tmp_path / "搬移后的项目"
    shutil.copytree(Path(project.work_dir), moved)
    loaded = load_project(moved)
    assert Path(loaded.work_dir).resolve() == moved.resolve()
    assert (Path(loaded.work_dir) / loaded.source_pdf).is_file()
    assert loaded.pages[0].lines[0].raw_text == "machine draft"
    assert loaded.pages[0].lines[0].edited_text == "Edited by a person."
    assert loaded.pages[0].lines[0].text == "Edited by a person."
    # The portable project must render after both the source and old project are
    # unavailable, not rely on the original absolute PDF path.
    image = render_page(loaded, loaded.pages[1])
    assert image.width > 0 and image.height > 0


def test_render_cache_rotation_and_blank_page(fixture_dir, tmp_path):
    project = create_project(str(fixture_dir / "two_pages.pdf"), tmp_path / "projects")
    page = project.pages[0]
    original = render_page(project, page)
    assert original.width == page.width and original.height == page.height
    assert page.image_path
    assert not Path(page.image_path).is_absolute()
    assert (Path(project.work_dir) / page.image_path).is_file()
    cached = load_page_image(project, page)
    assert cached.size == original.size
    assert ImageChops.difference(original.convert("RGB"), cached.convert("RGB")).getbbox() is None
    page.rotation = 90
    rotated = render_page(project, page)
    assert rotated.size == (original.height, original.width)
    blank = render_page(project, project.pages[1]).convert("L")
    assert blank.getextrema()[0] >= 245


def test_chinese_pdf_path_and_invalid_inputs(fixture_dir, tmp_path):
    project = create_project(str(fixture_dir / "中文路径" / "英文扫描稿.pdf"), tmp_path / "中文任务")
    assert len(project.pages) == 1
    for filename in ("broken.pdf", "encrypted.pdf"):
        path = fixture_dir / filename
        if path.exists():
            with pytest.raises(Exception):
                create_project(str(path), tmp_path / "bad")
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        create_project(str(tmp_path / "missing.pdf"), tmp_path / "missing")


def test_exports_use_edits_and_preserve_page_line_order(fixture_dir, tmp_path):
    project = create_project(str(fixture_dir / "two_pages.pdf"), tmp_path / "projects")
    project.pages[0].lines = [
        _line("a", "OLD OCR VALUE", edited="First corrected line."),
        _line("b", "Second line with café and 中文.", top=100),
    ]
    project.pages[0].status = "recognized"
    project.pages[1].lines = [_line("c", "Third line on page two.")]
    project.pages[1].status = "recognized"
    docx_path = tmp_path / "校对结果.docx"
    txt_path = tmp_path / "校对结果.txt"
    export_docx(project, docx_path)
    export_txt(project, txt_path)
    text = txt_path.read_text(encoding="utf-8-sig")
    assert "OLD OCR VALUE" not in text
    assert text.index("First corrected line.") < text.index("Second line") < text.index("Third line")
    assert "café and 中文" in text
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml")
    document = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    strings = [node.text or "" for node in document.findall(".//w:t", ns)]
    content = "\n".join(strings)
    assert "OLD OCR VALUE" not in content
    assert content.index("First corrected line.") < content.index("Second line") < content.index("Third line")
    assert document.findall('.//w:br[@w:type="page"]', ns) or document.findall(".//w:pageBreakBefore", ns)


def test_empty_manual_edit_is_not_replaced_with_ocr_text():
    line = _line("empty", "Delete this text", edited="")
    assert line.text == ""


def test_pending_and_stale_results_cannot_be_silently_exported(fixture_dir, tmp_path):
    project = create_project(str(fixture_dir / "single_page.pdf"), tmp_path / "projects")
    project.pages[0].status = "recognized"
    line = _line("stale", "Old OCR text")
    project.pages[0].lines = [line]
    for status in ("stale", "pending", "error"):
        line.status = status
        with pytest.raises(ValueError):
            export_txt(project, tmp_path / f"{status}.txt")
