"""Download and validate the two public models used by Handwriter.

Downloads are the only network operations in the application.  All runtime
loaders receive local paths; no Hugging Face cache in the user's home is used.
The detector and TrOCR weights are pinned and checked against upstream SHA256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

DETECTOR_DIRNAME = "ppocrv5-mobile-det"
RECOGNIZER_DIRNAME = "trocr-base-handwritten"
MANIFEST_NAME = "models-manifest.json"


class CancelledError(RuntimeError):
    """A cooperative cancellation, not a processing failure."""


class ModelDownloadError(RuntimeError):
    """An incomplete or unavailable model download."""


@dataclass(frozen=True)
class ModelFile:
    directory: str
    repository: str
    revision: str
    filename: str
    min_bytes: int = 1
    sha256: str | None = None

    @property
    def relative_path(self) -> str:
        return f"{self.directory}/{self.filename}"

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{self.filename}?download=true"
        )


MODEL_FILES = (
    ModelFile(
        DETECTOR_DIRNAME, "PaddlePaddle/PP-OCRv5_mobile_det_onnx", "e6f4fa8",
        "inference.onnx", 4_000_000,
        "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d",
    ),
    ModelFile(DETECTOR_DIRNAME, "PaddlePaddle/PP-OCRv5_mobile_det_onnx", "e6f4fa8", "inference.yml", 100),
    ModelFile(DETECTOR_DIRNAME, "PaddlePaddle/PP-OCRv5_mobile_det_onnx", "e6f4fa8", "README.md", 100),
    ModelFile(
        RECOGNIZER_DIRNAME, "microsoft/trocr-base-handwritten", "eaacaf4",
        "model.safetensors", 1_000_000_000,
        "25a40cddc7e6120140a3d5b9e3dd3878a92ada7b4f312953ab22edc19c2a5acc",
    ),
    *(ModelFile(RECOGNIZER_DIRNAME, "microsoft/trocr-base-handwritten", "eaacaf4", name, minimum)
      for name, minimum in (
          ("config.json", 100),
          ("generation_config.json", 20),
          ("preprocessor_config.json", 20),
          ("tokenizer_config.json", 20),
          ("special_tokens_map.json", 20),
          ("vocab.json", 100_000),
          ("merges.txt", 100_000),
          ("README.md", 100),
      )),
)


def _sha256(path: Path, cancelled: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if cancelled and cancelled():
                raise CancelledError("已取消模型校验。")
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _portable_digest(path: str, size: int, mtime_ns: int) -> str:
    # The size and mtime are part of the key.  An extracted portable package is
    # hashed once per process, while repeated GUI readiness checks stay cheap.
    return _sha256(Path(path))


def models_ready(models_dir: Path | str) -> bool:
    """Cheap, read-only validation of a completed download receipt.

    Extracting a portable ZIP can alter mtimes.  In that case compare the local
    hash to the receipt instead; this never triggers an online repair.
    """
    root = Path(models_dir)
    try:
        receipt = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        if receipt.get("version") != 1:
            return False
        entries = receipt["files"]
        for spec in MODEL_FILES:
            path = root / spec.relative_path
            stat = path.stat()
            record = entries[spec.relative_path]
            if stat.st_size < spec.min_bytes:
                return False
            if stat.st_size != record["size"]:
                return False
            if spec.sha256 and record.get("sha256") != spec.sha256:
                return False
            if stat.st_mtime_ns != record["mtime_ns"]:
                if _portable_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns) != record.get("sha256"):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _check_json(path: Path) -> None:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)


def _download(
    spec: ModelFile,
    path: Path,
    progress: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> None:
    """Stream to a .part file and resume when the server supports Range."""
    partial = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        if cancelled():
            raise CancelledError("已取消模型下载；下次可继续。")
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Handwriter-Local/1.0", "Accept-Encoding": "identity"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(spec.url, headers=headers)
        try:
            progress(f"下载 {spec.filename}（{attempt + 1}/3）…")
            with urllib.request.urlopen(request, timeout=35) as response:
                resumed = existing > 0 and response.status == 206
                if not resumed:
                    existing = 0
                content_length = response.headers.get("Content-Length")
                total = existing + int(content_length) if content_length else 0
                received = existing
                last_report = 0.0
                with partial.open("ab" if resumed else "wb") as output:
                    while True:
                        if cancelled():
                            raise CancelledError("已取消模型下载；下次可继续。")
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 0.8:
                            mb = received / 1024 / 1024
                            suffix = f" / {total / 1024 / 1024:.0f} MB" if total else " MB"
                            progress(f"下载 {spec.filename}：{mb:.0f}{suffix}")
                            last_report = now
            if partial.stat().st_size < spec.min_bytes:
                raise ModelDownloadError(f"{spec.filename} 下载不完整。")
            progress(f"校验 {spec.filename}…")
            digest = _sha256(partial, cancelled)
            if spec.sha256 and digest != spec.sha256:
                partial.unlink(missing_ok=True)
                raise ModelDownloadError(f"{spec.filename} 校验不通过，正在重试。")
            if path.suffix == ".json":
                json.loads(partial.read_text(encoding="utf-8"))
            os.replace(partial, path)
            return
        except CancelledError:
            raise
        except (OSError, ValueError, urllib.error.URLError, ModelDownloadError) as exc:
            last_error = exc
            log.warning("Model download attempt failed: %s", spec.relative_path, exc_info=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and partial.exists():
                if partial.stat().st_size >= spec.min_bytes and spec.sha256:
                    if _sha256(partial, cancelled) == spec.sha256:
                        os.replace(partial, path)
                        return
                partial.unlink(missing_ok=True)
            if attempt < 2:
                for _ in range(10):
                    if cancelled():
                        raise CancelledError("已取消模型下载；下次可继续。")
                    time.sleep(0.2)
    raise ModelDownloadError(
        f"无法完成 {spec.filename} 下载。请检查网络后重试；已完成的文件会保留。\n{last_error}"
    ) from last_error


def prepare_models(
    models_dir: Path | str,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    root = Path(models_dir)
    progress = progress or (lambda message: None)
    cancelled = cancelled or (lambda: False)
    if models_ready(root):
        progress("本地模型已准备好。")
        return
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for spec in MODEL_FILES:
        if cancelled():
            raise CancelledError("已取消模型准备。")
        path = root / spec.relative_path
        valid = False
        digest = ""
        if path.is_file() and path.stat().st_size >= spec.min_bytes:
            progress(f"校验本地 {spec.filename}…")
            digest = _sha256(path, cancelled)
            try:
                _check_json(path)
                valid = not spec.sha256 or digest == spec.sha256
            except (ValueError, OSError):
                valid = False
        if not valid:
            _download(spec, path, progress, cancelled)
            digest = _sha256(path, cancelled)
        stat = path.stat()
        records[spec.relative_path] = {
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest, "repository": spec.repository, "revision": spec.revision,
        }
    receipt = root / MANIFEST_NAME
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps({"version": 1, "files": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, receipt)
    progress("模型准备完成，现在可以离线识别。")
