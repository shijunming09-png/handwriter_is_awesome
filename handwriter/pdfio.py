from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

from .data import DocumentProject, PageResult
from .storage import project_path

# PDFium must not run on two threads at the same time.
_PDF_LOCK = threading.RLock()


def inspect_pdf(path: str | Path) -> int:
    target = Path(path)
    if not target.is_file():
        raise ValueError("找不到 PDF 文件。")
    if target.suffix.lower() != ".pdf":
        raise ValueError("请选择 PDF 文件。")
    try:
        with _PDF_LOCK, pdfium.PdfDocument(str(target)) as document:
            count = len(document)
            if not count:
                raise ValueError("这个 PDF 没有可处理的页面。")
            return count
    except pdfium.PdfiumError as exc:
        raise ValueError("无法打开 PDF：文件可能已损坏或有密码保护。请先解密后导入。") from exc


def _deskew(image: Image.Image) -> tuple[Image.Image, float]:
    rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, 1200.0 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = np.count_nonzero(binary)
    if ink < 100 or ink > binary.size * 0.4:
        return image, 0.0
    h, w = binary.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST)
        profile = np.count_nonzero(rotated, axis=1).astype(float)
        return float(np.sum(profile * profile))

    original_score = score(0)
    angles = np.arange(-6.0, 6.01, 1.0)
    best = float(max(angles, key=score))
    best = float(max(np.arange(best - 0.6, best + 0.61, 0.2), key=score))
    # Avoid introducing a rotation when the evidence is weak.
    if abs(best) < 0.25 or score(best) < original_score * 1.06:
        return image, 0.0
    return image.rotate(best, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white"), round(best, 2)


def render_page(project: DocumentProject, page: PageResult) -> Image.Image:
    source = project_path(project, project.source_pdf)
    try:
        with _PDF_LOCK, pdfium.PdfDocument(str(source)) as pdf:
            pdf_page = pdf[page.index]
            try:
                width, height = pdf_page.get_size()
                scale = project.dpi / 72.0
                # Prevent pathological PDF page sizes from exhausting system memory.
                max_pixels = 40_000_000
                if width * height * scale * scale > max_pixels:
                    scale = (max_pixels / (width * height)) ** 0.5
                    warning = "页面尺寸很大，已降低渲染分辨率。"
                    if warning not in page.warnings:
                        page.warnings.append(warning)
                bitmap = pdf_page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB").copy()
                finally:
                    bitmap.close()
            finally:
                pdf_page.close()
    except (pdfium.PdfiumError, IndexError) as exc:
        raise ValueError(f"第 {page.index + 1} 页无法渲染：{exc}") from exc
    if page.rotation % 360:
        image = image.rotate(-page.rotation % 360, expand=True, fillcolor="white")
    image, angle = _deskew(image)
    # Mild denoising only. Never binarize the image fed to TrOCR.
    rgb = np.asarray(image)
    image = Image.fromarray(cv2.medianBlur(rgb, 3))
    page.skew_angle = angle
    page.width, page.height = image.size
    page.image_path = f"pages/{page.index + 1:05d}.png"
    destination = project_path(project, page.image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return image


def load_page_image(project: DocumentProject, page: PageResult) -> Image.Image:
    if page.image_path:
        target = project_path(project, page.image_path)
        if target.is_file():
            with Image.open(target) as image:
                return image.convert("RGB").copy()
    return render_page(project, page)
