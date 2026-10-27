"""Local text-line detection and TrOCR recognition.

The PP-OCRv5 detector runs in ONNX Runtime on CPU.  Its DB preprocessing and
postprocessing follow the model's public inference.yml; neither PaddlePaddle
nor PaddleOCR is imported.  Model and upstream algorithm: PaddlePaddle,
Apache-2.0, https://github.com/PaddlePaddle/PaddleOCR .
"""

from __future__ import annotations

import gc
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from .data import LineResult
from .model_manager import (
    CancelledError,
    DETECTOR_DIRNAME,
    RECOGNIZER_DIRNAME,
    models_ready,
    prepare_models,
)

log = logging.getLogger(__name__)


def _remaining_warnings(warnings: list[str]) -> list[str]:
    return [
        message for message in warnings
        if not message.startswith("识别：")
        and not message.startswith("行框已修改，请重新识别")
    ]


def order_quad(points: Iterable) -> np.ndarray:
    """Order a near-horizontal quadrilateral TL, TR, BR, BL."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    left, right = points[np.argsort(points[:, 0])[:2]], points[np.argsort(points[:, 0])[2:]]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def crop_line(image: Image.Image, polygon: Iterable) -> Image.Image:
    """Perspective-rectify a line from the full resolution page."""
    quad = order_quad(polygon)
    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    width, height = int(math.ceil(width)), int(math.ceil(height))
    if width < 2 or height < 2 or width * height > 100_000_000:
        raise ValueError("行框为空或尺寸异常，请调整行框。")
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(quad, destination)
    pixels = cv2.warpPerspective(
        np.asarray(image.convert("RGB")), matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return Image.fromarray(pixels)


def _bounds(quad: np.ndarray) -> tuple[float, float, float, float]:
    return float(quad[:, 0].min()), float(quad[:, 1].min()), float(quad[:, 0].max()), float(quad[:, 1].max())


def merge_fragments(
    items: list[tuple[np.ndarray, float]], page_width: int
) -> list[tuple[np.ndarray, float]]:
    """Conservatively join horizontal fragments without crossing baselines."""
    if not items:
        return []
    groups: list[list[tuple[np.ndarray, float]]] = []
    ordered = sorted(items, key=lambda item: (float(item[0][:, 1].mean()), float(item[0][:, 0].min())))
    for quad, score in ordered:
        x0, y0, x1, y1 = _bounds(quad)
        height = max(y1 - y0, 1.0)
        center = (y0 + y1) / 2
        chosen = None
        best_distance = float("inf")
        for group in groups:
            all_points = np.concatenate([item[0] for item in group])
            gx0, gy0, gx1, gy1 = _bounds(all_points)
            heights = [_bounds(item[0])[3] - _bounds(item[0])[1] for item in group]
            median_height = max(float(np.median(heights)), 1.0)
            group_center = float(np.median([item[0][:, 1].mean() for item in group]))
            center_distance = abs(center - group_center)
            overlap_y = min(y1, gy1) - max(y0, gy0)
            horizontal_gap = max(gx0 - x1, x0 - gx1, 0)
            same_row = center_distance <= 0.38 * min(height, median_height)
            overlaps = overlap_y >= 0.58 * min(height, median_height)
            plausible_gap = horizontal_gap <= max(3.5 * median_height, page_width * 0.035)
            plausible_height = max(y1, gy1) - min(y0, gy0) <= 1.65 * max(height, median_height)
            if same_row and overlaps and plausible_gap and plausible_height and center_distance < best_distance:
                chosen, best_distance = group, center_distance
        if chosen is None:
            groups.append([(quad, score)])
        else:
            chosen.append((quad, score))
    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
            continue
        points = np.concatenate([item[0] for item in group]).astype(np.float32)
        box = order_quad(cv2.boxPoints(cv2.minAreaRect(points)))
        result.append((box, float(np.mean([item[1] for item in group]))))
    return sorted(result, key=lambda item: (float(item[0][:, 1].mean()), float(item[0][:, 0].min())))


def _expand_quad(quad: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    quad = order_quad(quad)
    horizontal = quad[1] - quad[0]
    vertical = quad[3] - quad[0]
    height = max(float(np.linalg.norm(vertical)), 1.0)
    hx = horizontal / max(float(np.linalg.norm(horizontal)), 1.0) * max(2.0, height * 0.10)
    vy = vertical / height * max(2.0, height * 0.10)
    quad += np.array([-hx - vy, hx - vy, hx + vy, -hx + vy])
    quad[:, 0] = np.clip(quad[:, 0], 0, image_size[0] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, image_size[1] - 1)
    return quad


def _db_boxes(probability: np.ndarray, original_size: tuple[int, int]) -> list[tuple[np.ndarray, float]]:
    """Decode DB probability maps using the published PP-OCRv5 settings."""
    import pyclipper

    probability = np.asarray(probability, dtype=np.float32).squeeze()
    if probability.ndim != 2:
        raise ValueError(f"检测模型返回异常维度：{probability.shape}")
    height, width = probability.shape
    mask = (probability > 0.3).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1500]:
        rectangle = cv2.minAreaRect(contour)
        if min(rectangle[1]) < 3:
            continue
        quad = order_quad(cv2.boxPoints(rectangle))
        x0, y0, x1, y1 = _bounds(quad)
        ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        ix1, iy1 = min(width, int(math.ceil(x1)) + 1), min(height, int(math.ceil(y1)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        score_mask = np.zeros((iy1 - iy0, ix1 - ix0), dtype=np.uint8)
        local_quad = quad - np.float32([ix0, iy0])
        cv2.fillPoly(score_mask, [np.round(local_quad).astype(np.int32)], 1)
        score = float(cv2.mean(probability[iy0:iy1, ix0:ix1], mask=score_mask)[0])
        if score < 0.6:
            continue
        perimeter = cv2.arcLength(quad, True)
        area = abs(cv2.contourArea(quad))
        if perimeter <= 0 or area <= 0:
            continue
        # Integer scaling avoids quantizing a small contour during offsetting.
        scale = 256.0
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(np.round(quad * scale).astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(area * 1.5 / perimeter * scale)
        if len(expanded) != 1:
            continue
        expanded_points = np.asarray(expanded[0], dtype=np.float32) / scale
        rect = cv2.minAreaRect(expanded_points)
        if min(rect[1]) < 5:
            continue
        quad = order_quad(cv2.boxPoints(rect))
        quad[:, 0] *= original_size[0] / width
        quad[:, 1] *= original_size[1] / height
        quad[:, 0] = np.clip(quad[:, 0], 0, original_size[0] - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, original_size[1] - 1)
        boxes.append((quad, score))
    return boxes


class OcrEngine:
    def __init__(
        self,
        models_dir: Path | str,
        progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        recognizer_dir: Path | str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.recognizer_dir = Path(recognizer_dir) if recognizer_dir is not None else None
        self.progress = progress or (lambda message: None)
        self.cancelled = cancelled or (lambda: False)
        self._session = None
        self._processor = None
        self._model = None
        self.device = "cpu"
        self.device_name = "CPU"
        self.batch_size = 4
        self.max_new_tokens = 128

    def _check_cancelled(self) -> None:
        if self.cancelled():
            raise CancelledError("已取消识别，已完成的结果已保留。")

    def prepare_models(self) -> None:
        prepare_models(self.models_dir, self.progress, self.cancelled)

    def _require_models(self) -> None:
        self._check_cancelled()
        if not models_ready(self.models_dir):
            raise RuntimeError("本地模型尚未准备好或文件已改变。请先点击“准备模型”完成下载与校验。")

    def _load_detector(self) -> None:
        if self._session is not None:
            return
        self._require_models()
        import onnxruntime as ort

        self.progress("加载自动切行模型（CPU）…")
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(max((os.cpu_count() or 4) // 2, 1), 8)
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self.models_dir / DETECTOR_DIRNAME / "inference.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"],
        )

    def _use_cpu(self, reason: str) -> None:
        import torch

        log.warning("Using CPU for recognition: %s", reason)
        self.progress(f"GPU 暂不可用，使用 CPU 识别：{reason}")
        self.device, self.device_name, self.batch_size = "cpu", "CPU", 1
        if self._model is not None:
            self._model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_recognizer(self) -> None:
        if self._model is not None:
            return
        self._require_models()
        self.progress("加载 TrOCR 手写识别模型…")
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        torch.set_num_threads(min(max((os.cpu_count() or 4) // 2, 1), 8))
        if self.recognizer_dir is not None:
            from .model_validation import validate_recognizer_dir
            local_dir = str(validate_recognizer_dir(self.recognizer_dir))
        else:
            local_dir = str(self.models_dir / RECOGNIZER_DIRNAME)
        self._processor = TrOCRProcessor.from_pretrained(local_dir, local_files_only=True, use_fast=False)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            local_dir, local_files_only=True, use_safetensors=True,
        )
        self._model.eval()
        if torch.cuda.is_available():
            try:
                self.progress("检查 GPU 运算兼容性…")
                probe = torch.ones((16, 16), device="cuda")
                probe = probe @ probe
                torch.cuda.synchronize()
                del probe
                self._model.to("cuda")
                # Exercise actual model operators, not merely CUDA discovery.
                dummy = Image.new("RGB", (384, 64), "white")
                pixels = self._processor(images=dummy, return_tensors="pt").pixel_values.to("cuda")
                with torch.inference_mode():
                    self._model.generate(pixels, max_new_tokens=1, num_beams=1)
                torch.cuda.synchronize()
                del pixels
                self.device = "cuda"
                self.device_name = torch.cuda.get_device_name(0)
                self.batch_size = 4
                self.progress(f"GPU 已就绪：{self.device_name}")
            except Exception as exc:
                self._use_cpu(str(exc).splitlines()[0][:180])
        else:
            self._use_cpu("未检测到可用的 CUDA 设备")

    def load(self) -> str:
        self._load_detector()
        self._check_cancelled()
        self._load_recognizer()
        self._check_cancelled()
        return self.device_name

    def close(self) -> None:
        """Release inference resources before a training worker uses the GPU."""
        self._model = None
        self._processor = None
        self._session = None
        gc.collect()
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, image: Image.Image) -> list[LineResult]:
        self._check_cancelled()
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Avoid a generative recognizer producing words from a blank page.
        if float(gray.std()) < 1.2 or float(np.mean(gray < 220)) < 0.000025:
            return []
        self._load_detector()
        height, width = gray.shape
        # A larger detection raster helps fine English handwriting.  Crops are
        # always taken later from the original, full-resolution page.
        ratio = min(1536.0 / max(width, height), 1.0)
        resized_w = max(32, int(round(width * ratio / 32)) * 32)
        resized_h = max(32, int(round(height * ratio / 32)) * 32)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: tensor})[0]
        self._check_cancelled()
        detections = merge_fragments(_db_boxes(output, image.size), width)
        result = []
        for quad, score in detections:
            quad = _expand_quad(quad, image.size)
            x0, y0, x1, y1 = _bounds(quad)
            if x1 - x0 < 5 or y1 - y0 < 4:
                continue
            warnings = []
            if y1 - y0 > (x1 - x0) * 1.2:
                warnings.append("行框偏高，可能包含竖排或多行，请检查。")
            if (x1 - x0) / max(y1 - y0, 1) > 48:
                warnings.append("文字行很长，建议拆分后检查识别结果。")
            result.append(LineResult(
                id=uuid.uuid4().hex,
                polygon=quad.astype(float).tolist(),
                detection_score=score,
                warnings=warnings,
                status="pending",
            ))
        return result

    def recognize(self, image: Image.Image, lines: list[LineResult]) -> None:
        self._check_cancelled()
        if not lines:
            return
        self._load_recognizer()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancelGeneration(StoppingCriteria):
            def __init__(self, callback):
                self.callback = callback

            def __call__(self, input_ids, scores, **kwargs):
                return self.callback()

        stopping = StoppingCriteriaList([CancelGeneration(self.cancelled)])
        index = 0
        while index < len(lines):
            self._check_cancelled()
            current = lines[index:index + self.batch_size]
            crops, valid = [], []
            for line in current:
                try:
                    crop = crop_line(image, line.polygon)
                    gray = np.asarray(crop.convert("L"))
                    if float(gray.std()) < 1.2 or float(np.mean(gray < 220)) < 0.0001:
                        line.raw_text = ""
                        line.status = "review"
                        line.warnings = _remaining_warnings(line.warnings)
                        line.warnings.append("识别：行框内没有清晰文字，请检查行框。")
                        continue
                    crops.append(crop)
                    valid.append(line)
                except (ValueError, cv2.error) as exc:
                    line.status = "error"
                    line.warnings.append(f"识别：{exc}")
            if not valid:
                index += len(current)
                continue
            pixels = None
            generated = None
            try:
                pixels = self._processor(images=crops, return_tensors="pt").pixel_values.to(self.device)
                with torch.inference_mode():
                    generated = self._model.generate(
                        pixels,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=3,
                        do_sample=False,
                        stopping_criteria=stopping,
                    )
                self._check_cancelled()
                sequences = generated.detach().cpu()
                texts = self._processor.batch_decode(sequences, skip_special_tokens=True)
                eos = self._model.generation_config.eos_token_id
                eos_tokens = {eos} if isinstance(eos, int) else set(eos or [])
                for line, tokens, text in zip(valid, sequences.tolist(), texts):
                    line.raw_text = text.strip()
                    line.warnings = _remaining_warnings(line.warnings)
                    if not line.raw_text:
                        line.warnings.append("识别：结果为空，请检查原稿或调整行框。")
                    # First token is the decoder start token, which for this
                    # model may itself equal EOS; only examine generated tokens.
                    ended = any(token in eos_tokens for token in tokens[1:])
                    if len(tokens) - 1 >= self.max_new_tokens and not ended:
                        line.warnings.append("识别：达到文字长度上限，结果可能截断，请拆分此行。")
                    if len(line.raw_text) >= 18 and len(set(line.raw_text.lower().replace(" ", ""))) <= 3:
                        line.warnings.append("识别：文字重复异常，请对照原稿复核。")
                    line.status = "review" if line.warnings else "recognized"
                index += len(current)
                self.progress(f"已识别 {index}/{len(lines)} 行（{self.device_name}）")
            except CancelledError:
                raise
            except RuntimeError as exc:
                message = str(exc).lower()
                if self.device == "cuda" and ("out of memory" in message or "cuda" in message or "cudnn" in message):
                    del pixels, generated
                    gc.collect()
                    torch.cuda.empty_cache()
                    if self.batch_size > 1 and "out of memory" in message:
                        self.batch_size = max(1, self.batch_size // 2)
                        self.progress(f"显存不足，识别批量已调整为 {self.batch_size} 行，继续处理。")
                    else:
                        self._use_cpu("GPU 推理失败，已保留进度并切换 CPU")
                    continue
                log.exception("Recognition failed for a batch")
                for line in valid:
                    line.status = "error"
                    line.warnings.append(f"识别：处理失败，{str(exc).splitlines()[0][:160]}")
                index += len(current)
            finally:
                pixels = None
                generated = None
