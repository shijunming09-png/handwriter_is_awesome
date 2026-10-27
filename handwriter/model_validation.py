"""Integrity checks for locally published personal TrOCR models."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .model_manager import MODEL_FILES, RECOGNIZER_DIRNAME

BASE_MODEL_SHA256 = next(item.sha256 for item in MODEL_FILES
                         if item.directory == RECOGNIZER_DIRNAME and item.filename == "model.safetensors")


@lru_cache(maxsize=256)
def _digest(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_recognizer_dir(path: Path | str) -> Path:
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "model-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("base_model_sha256") != BASE_MODEL_SHA256:
            raise ValueError("模型基座版本不兼容。")
        files = manifest["files"]
        required = {"model.safetensors", "config.json", "preprocessor_config.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"}
        if not required.issubset(files):
            raise ValueError("个人模型缺少识别所需的配置或权重。")
        for relative, record in files.items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError("个人模型文件路径无效或文件不存在。")
            stat = target.stat()
            if stat.st_size != record["size"] or _digest(str(target), stat.st_size, stat.st_mtime_ns) != record["sha256"]:
                raise ValueError(f"个人模型文件校验失败：{relative}")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "vision-encoder-decoder" or config.get("decoder", {}).get("model_type") != "trocr":
            raise ValueError("该个人模型不是兼容的 TrOCR 模型。")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("个人模型缺失或已损坏，请在模型列表中重新选择通用模型。") from exc
    return root
