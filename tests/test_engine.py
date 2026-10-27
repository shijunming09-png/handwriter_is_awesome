from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from handwriter.data import LineResult
from handwriter.engine import OcrEngine, _db_boxes, crop_line, merge_fragments, order_quad
from handwriter.model_manager import CancelledError, models_ready
from handwriter import model_manager


def quad(x, y, width, height):
    return np.float32([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])


def ink_image():
    image = Image.new("RGB", (500, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((15, 15), "This is a handwritten line.", fill="black")
    return image


def test_fragments_merge_within_rows_but_do_not_cross_close_rows():
    fragments = [
        (quad(125, 41, 100, 19), .88),
        (quad(10, 10, 100, 20), .91),
        (quad(125, 11, 100, 18), .90),
        (quad(10, 40, 100, 20), .92),
    ]
    merged = merge_fragments(fragments, 1000)
    assert len(merged) == 2
    assert merged[0][0][:, 0].max() >= 224
    assert merged[0][0][:, 1].mean() < merged[1][0][:, 1].mean()
    # Ascenders and descenders may make adjacent line bounding boxes overlap.
    close_rows = [(quad(10, 10, 150, 20), .9), (quad(10, 26, 150, 20), .9)]
    assert len(merge_fragments(close_rows, 1000)) == 2


def test_perspective_crop_preserves_rgb_and_rejects_degenerate_box():
    image = Image.new("RGB", (200, 100), (30, 60, 190))
    crop = crop_line(image, quad(20, 20, 150, 35))
    assert crop.size == (150, 35)
    assert crop.getpixel((75, 17)) == (30, 60, 190)
    assert np.allclose(order_quad(quad(20, 20, 150, 35)[[2, 0, 3, 1]]), quad(20, 20, 150, 35))
    with pytest.raises(ValueError):
        crop_line(image, [[2, 2]] * 4)


def test_db_decoding_scales_boxes_and_discards_low_score_regions():
    probability = np.zeros((1, 1, 160, 160), dtype=np.float32)
    probability[0, 0, 20:34, 15:130] = .95
    probability[0, 0, 70:84, 15:130] = .4
    detections = _db_boxes(probability, (1600, 1600))
    assert len(detections) == 1
    box, score = detections[0]
    assert score > .9
    assert box[:, 0].max() > 1200
    assert box[:, 1].min() < 200
    assert np.isfinite(box).all()


def test_blank_page_does_not_require_models_or_hallucinate(tmp_path):
    engine = OcrEngine(tmp_path / "missing-models")
    assert engine.detect(Image.new("RGB", (600, 800), "white")) == []
    assert not models_ready(tmp_path)


def test_cancellation_is_distinct_from_failure(tmp_path):
    engine = OcrEngine(tmp_path, cancelled=lambda: True)
    with pytest.raises(CancelledError):
        engine.detect(Image.new("RGB", (600, 800), "white"))
    with pytest.raises(CancelledError):
        engine.recognize(ink_image(), [])


def test_model_receipt_survives_zip_timestamp_change_but_rejects_corruption(tmp_path, monkeypatch):
    contents = b"model weights"
    digest = hashlib.sha256(contents).hexdigest()
    spec = model_manager.ModelFile("detector", "example/repo", "revision", "weights.dat", 5, digest)
    monkeypatch.setattr(model_manager, "MODEL_FILES", (spec,))
    target = tmp_path / spec.relative_path
    target.parent.mkdir()
    target.write_bytes(contents)
    stat = target.stat()
    (tmp_path / model_manager.MANIFEST_NAME).write_text(json.dumps({
        "version": 1,
        "files": {spec.relative_path: {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}},
    }), encoding="utf-8")
    assert models_ready(tmp_path)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns - 1_000_000_000))
    assert models_ready(tmp_path)
    target.write_bytes(b"wrong weights")  # Identical file size must still fail.
    assert not models_ready(tmp_path)
    target.write_bytes(b"short")
    assert not models_ready(tmp_path)


class FakePixels:
    def __init__(self, count):
        self.count = count
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self, texts):
        self.texts = texts

    def __call__(self, images, return_tensors):
        return SimpleNamespace(pixel_values=FakePixels(len(images)))

    def batch_decode(self, sequences, skip_special_tokens):
        return self.texts[:len(sequences)]


class FakeModel:
    def __init__(self, rows):
        self.rows = rows
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.calls = []

    def generate(self, pixels, **kwargs):
        import torch
        self.calls.append((pixels.count, pixels.device))
        return torch.tensor(self.rows[:pixels.count])

    def to(self, device):
        return self


def recognizer(tmp_path, texts, rows):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    engine = OcrEngine(tmp_path)
    engine._processor = FakeProcessor(texts)
    engine._model = FakeModel(rows)
    return engine


def test_recognition_keeps_manual_corrections_and_eos_is_not_confused_with_start(tmp_path):
    engine = recognizer(tmp_path, ["recognized original"], [[2, 21, 22, 2]])
    line = LineResult(polygon=quad(5, 5, 350, 45).tolist(), edited_text="My correction")
    engine.recognize(ink_image(), [line])
    assert line.raw_text == "recognized original"
    assert line.text == "My correction"
    assert line.status == "recognized"
    assert not line.warnings


def test_successful_rerecognition_clears_obsolete_geometry_warning(tmp_path):
    engine = recognizer(tmp_path, ["Recognized again"], [[2, 21, 22, 2]])
    line = LineResult(
        polygon=quad(5, 5, 350, 45).tolist(), status="stale",
        warnings=["行框已修改，请重新识别或输入人工转写。"],
    )
    engine.recognize(ink_image(), [line])
    assert line.status == "recognized"
    assert not line.warnings


def test_token_limit_is_flagged_instead_of_silently_truncating(tmp_path):
    engine = recognizer(tmp_path, ["An unfinished line"], [[2] + [21] * 128])
    line = LineResult(polygon=quad(5, 5, 350, 45).tolist())
    engine.recognize(ink_image(), [line])
    assert line.status == "review"
    assert any("截断" in warning for warning in line.warnings)


def test_blank_crop_skips_generation_and_preserves_corrected_text(tmp_path):
    engine = recognizer(tmp_path, ["hallucination"], [[2, 21, 2]])
    line = LineResult(polygon=quad(5, 5, 350, 45).tolist(), edited_text="Manual text")
    engine.recognize(Image.new("RGB", (500, 160), "white"), [line])
    assert not engine._model.calls
    assert line.raw_text == ""
    assert line.text == "Manual text"
    assert line.status == "review"


def test_cancelled_generation_does_not_commit_partial_output(tmp_path):
    state = {"cancelled": False}
    engine = recognizer(tmp_path, ["Incomplete generated text"], [[2, 21, 2]])
    original_generate = engine._model.generate

    def cancel_during_generate(*args, **kwargs):
        state["cancelled"] = True
        return original_generate(*args, **kwargs)

    engine._model.generate = cancel_during_generate
    engine.cancelled = lambda: state["cancelled"]
    line = LineResult(polygon=quad(5, 5, 350, 45).tolist(), raw_text="Previously saved")
    with pytest.raises(CancelledError):
        engine.recognize(ink_image(), [line])
    assert line.raw_text == "Previously saved"


def test_cuda_oom_reduces_batch_then_continues_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    engine = recognizer(tmp_path, ["A line"] * 4, [[2, 21, 2]] * 4)
    engine.device = "cuda"
    engine.device_name = "Test GPU"
    original_generate = engine._model.generate
    attempts = []

    def oom_until_cpu(pixels, **kwargs):
        attempts.append((pixels.count, pixels.device))
        if pixels.device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return original_generate(pixels, **kwargs)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    engine._model.generate = oom_until_cpu
    lines = [LineResult(polygon=quad(5, 5, 350, 45).tolist()) for _ in range(4)]
    engine.recognize(ink_image(), lines)
    assert attempts[:3] == [(4, "cuda"), (2, "cuda"), (1, "cuda")]
    assert engine.device == "cpu"
    assert all(line.status == "recognized" for line in lines)
    assert all(line.raw_text == "A line" for line in lines)
