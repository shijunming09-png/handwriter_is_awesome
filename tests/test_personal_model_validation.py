import hashlib
import json
from pathlib import Path

import pytest

from handwriter.model_validation import BASE_MODEL_SHA256, validate_recognizer_dir


def write_model(folder: Path) -> Path:
    folder.mkdir()
    files = {
        "config.json": json.dumps({"model_type": "vision-encoder-decoder", "decoder": {"model_type": "trocr"}}).encode(),
        "model.safetensors": b"test integrity fixture, not an inference model",
        "preprocessor_config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "generation_config.json": b"{}",
        "vocab.json": b"{}",
        "merges.txt": b"#version: 0.2\n",
    }
    records = {}
    for name, value in files.items():
        (folder / name).write_bytes(value)
        records[name] = {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    (folder / "model-manifest.json").write_text(json.dumps({
        "schema_version": 1, "base_model_sha256": BASE_MODEL_SHA256, "files": records,
    }), encoding="utf-8")
    return folder


def test_personal_model_is_portable_and_detects_corruption(tmp_path):
    model = write_model(tmp_path / "个人模型")
    assert validate_recognizer_dir(model) == model.resolve()
    weights = model / "model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    with pytest.raises(ValueError, match="校验失败"):
        validate_recognizer_dir(model)


def test_rejects_wrong_base_and_incomplete_model(tmp_path):
    model = write_model(tmp_path / "model")
    manifest_path = model / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_model_sha256"] = "wrong-base"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="基座版本"):
        validate_recognizer_dir(model)
    manifest["base_model_sha256"] = BASE_MODEL_SHA256
    del manifest["files"]["vocab.json"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="缺少"):
        validate_recognizer_dir(model)


def test_rejects_model_file_escape(tmp_path):
    model = write_model(tmp_path / "model")
    manifest_path = model / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../outside"] = {"size": 0, "sha256": ""}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="路径无效"):
        validate_recognizer_dir(model)
