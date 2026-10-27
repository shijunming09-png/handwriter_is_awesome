"""Generate deterministic local QA inputs; printed synthetic pages are not handwriting.

Run from the source folder:
    python tools/make_fixtures.py [--download-iam]

The optional IAM example is for local evaluation only and must not be shipped in
the portable application. See tests/fixtures/SOURCES.md for terms and provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
IAM_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/iam_picture.jpeg"
IAM_TEXT = "industry"
TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "Please preserve the order of these lines.",
    "Letters such as g, j, p, q and y have tails.",
    "This page contains five separate text lines.",
    "The final line should not be lost or repeated.",
]


def _font(size: int = 42) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/timesi.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _page(*, ruled: bool = False, close: bool = False) -> tuple[Image.Image, list[list[int]]]:
    image = Image.new("RGB", (1275, 1650), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    if ruled:
        for y in range(170, 1450, 78):
            draw.line((70, y, 1205, y), fill=(170, 188, 215), width=2)
    step = 55 if close else 140
    boxes = []
    for i, line in enumerate(TEXT):
        position = (95, 210 + i * step)
        box = list(draw.textbbox(position, line, font=font))
        draw.text(position, line, font=font, fill=(35, 35, 35))
        boxes.append(box)
    return image, boxes


def generate_fixtures(output: Path = FIXTURES, download_iam: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    clean, clean_boxes = _page()
    ruled, ruled_boxes = _page(ruled=True)
    close, close_boxes = _page(close=True)
    skew = clean.rotate(5, resample=Image.Resampling.BICUBIC, fillcolor="white")
    blank = Image.new("RGB", clean.size, "white")
    rotated = clean.transpose(Image.Transpose.ROTATE_270)
    for name, image in {
        "synthetic_clean": clean,
        "synthetic_ruled": ruled,
        "synthetic_close_lines": close,
        "synthetic_skew_5deg": skew,
        "blank": blank,
        "rotated_90": rotated,
    }.items():
        image.save(output / f"{name}.png")
    clean.save(output / "single_page.pdf", resolution=150)
    rotated.save(output / "rotated_90.pdf", resolution=150)
    clean.save(output / "two_pages.pdf", save_all=True, append_images=[blank], resolution=150)
    clean.save(
        output / "layout_scenarios.pdf",
        save_all=True,
        append_images=[skew, ruled, close, blank, rotated],
        resolution=150,
    )
    chinese_dir = output / "中文路径"
    chinese_dir.mkdir(exist_ok=True)
    clean.save(chinese_dir / "英文扫描稿.pdf", resolution=150)
    (output / "broken.pdf").write_bytes(b"%PDF-1.7\nthis is not a complete PDF\n")
    try:
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        writer.add_page(PdfReader(output / "single_page.pdf").pages[0])
        writer.encrypt("qa-only-password")
        with (output / "encrypted.pdf").open("wb") as stream:
            writer.write(stream)
    except ImportError:
        # ReportLab is a development-only fixture dependency; using it here
        # avoids silently dropping the encrypted-input acceptance scenario.
        from reportlab.lib.pdfencrypt import StandardEncryption
        from reportlab.pdfgen.canvas import Canvas

        encryption = StandardEncryption("qa-only-password")
        canvas = Canvas(str(output / "encrypted.pdf"), encrypt=encryption)
        canvas.drawString(72, 720, "This input requires a password.")
        canvas.showPage()
        canvas.save()
    manifest = {
        "synthetic": True,
        "purpose": "layout, rendering, storage and export regression tests; not handwriting accuracy",
        "page_size_px": list(clean.size),
        "pdf_source_dpi": 150,
        "text": TEXT,
        "clean_boxes": clean_boxes,
        "ruled_boxes": ruled_boxes,
        "close_boxes": close_boxes,
        "iam": {
            "url": IAM_URL,
            "reference_text": IAM_TEXT,
            "reference_source": "https://github.com/huggingface/transformers/blob/main/tests/models/vision_encoder_decoder/test_modeling_vision_encoder_decoder.py",
            "redistribution": "excluded; local evaluation only, subject to IAM terms",
        },
    }
    if download_iam:
        request = urllib.request.Request(IAM_URL, headers={"User-Agent": "Handwriter-local-QA/1.0"})
        target = output / "iam_industry.jpg"
        if not target.exists():
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(10 * 1024 * 1024)
            target.write_bytes(payload)
        with Image.open(target) as sample:
            sample.verify()
        (output / "iam_industry.txt").write_text(IAM_TEXT + "\n", encoding="utf-8")
        # Build a PDF page with repeated real handwritten lines for ordering and
        # segmentation checks. Repetition does not make this an accuracy corpus.
        with Image.open(target) as sample:
            sample = sample.convert("RGB")
            sample.thumbnail((1100, 170), Image.Resampling.LANCZOS)
            page = Image.new("RGB", (1275, 1650), "white")
            for y in (220, 460, 700):
                page.paste(sample, (80, y))
            page.save(output / "iam_three_lines.png")
            page.save(output / "iam_three_lines.pdf", resolution=150)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def download_iam_lines(output: Path = FIXTURES) -> dict:
    """Fetch three public IAM test rows, not the full dataset archive.

    Dataset-viewer links are ephemeral; preserve stable row identifiers and the
    source transcription, never a private token or signed download query.
    """
    output.mkdir(parents=True, exist_ok=True)
    api = "https://datasets-server.huggingface.co/first-rows?dataset=Teklia%2FIAM-line&config=default&split=test"
    request = urllib.request.Request(api, headers={"User-Agent": "Handwriter-local-QA/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.load(response)
    rows = data["rows"][:3]
    page = Image.new("RGB", (2480, 3508), "white")
    records = []
    for position, entry in enumerate(rows):
        row = entry["row"]
        image_url = row["image"]["src"]
        image_name = f"iam_test_{entry['row_idx']:03d}.jpg"
        target = output / image_name
        request = urllib.request.Request(image_url, headers={"User-Agent": "Handwriter-local-QA/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(10 * 1024 * 1024)
        target.write_bytes(payload)
        with Image.open(target) as source:
            source = source.convert("RGB")
            source.thumbnail((2180, 240), Image.Resampling.LANCZOS)
            page.paste(source, (150, 440 + position * 450))
        (output / image_name.replace(".jpg", ".txt")).write_text(row["text"] + "\n", encoding="utf-8")
        records.append({
            "row_index": entry["row_idx"],
            "image": image_name,
            "reference_text": row["text"],
            "source_image_url": image_url.split("?", 1)[0],
            "original_size": [row["image"]["width"], row["image"]["height"]],
        })
    page.save(output / "iam_multiline.png")
    page.save(output / "iam_multiline.pdf", resolution=300)
    manifest = {
        "dataset": "Teklia/IAM-line",
        "split": "test",
        "source": "https://huggingface.co/datasets/Teklia/IAM-line",
        "api": api,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": "Three public dataset-viewer examples for local evaluation only. Source labels preserve IAM tokenized punctuation spacing. Do not redistribute source images in the portable app.",
        "rows": records,
    }
    (output / "iam_lines_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-iam", action="store_true")
    parser.add_argument("--download-iam-lines", action="store_true")
    parser.add_argument("--output", type=Path, default=FIXTURES)
    args = parser.parse_args()
    print(json.dumps(generate_fixtures(args.output, args.download_iam), ensure_ascii=False, indent=2))
    if args.download_iam_lines:
        print(json.dumps(download_iam_lines(args.output), ensure_ascii=False, indent=2))
