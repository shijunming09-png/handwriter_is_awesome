from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from handwriter.training import BASE_SHA256, TrainingConfig, _deny_network, run_training, sequence_loss, validate_dataset
from handwriter.training_metrics import compare_evaluations, edit_distance, score_lines


class Tokenizer:
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        ids = list(range(10, 10 + len(text.split())))
        return {"input_ids": [0] + ids + [2] if kwargs.get("add_special_tokens", True) else ids}


def snapshot(directory: Path, count=5):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "images").mkdir(exist_ok=True)
    samples = []
    for index in range(count):
        split = "train" if index < count - 2 else "validation" if index == count - 2 else "test"
        image = Image.new("RGB", (180, 40), "white")
        ImageDraw.Draw(image).text((5, 5), f"Real pixels {index}", fill="black")
        path = directory / "images" / f"{index}.png"
        image.save(path)
        samples.append({"id": str(index), "image": f"images/{index}.png",
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "page_id": str(index), "split": split, "text": "A verified text line."})
    dataset = {"id": "snapshot", "writer_id": "writer", "base_model_sha256": BASE_SHA256,
               "samples": samples, "counts": {key: sum(row["split"] == key for row in samples)
                                                 for key in ("train", "validation", "test")}}
    return write_dataset(directory, dataset), dataset


def write_dataset(directory, data):
    path = directory / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    (directory / "snapshot.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest(), encoding="ascii")
    return path


def test_strict_metrics_preserve_case_punctuation_spaces_and_weight_by_characters():
    result = score_lines([{"id": "a", "ground_truth": "ABC", "prediction": "abc"},
                          {"id": "b", "ground_truth": "a " * 10, "prediction": "a " * 10}])
    assert result["character_errors"] == 3
    assert result["reference_characters"] == 23
    assert result["cer"] == pytest.approx(3 / 23)
    assert edit_distance("kitten", "sitting") == 3


def test_comparison_requires_identical_held_out_samples():
    baseline = {"validation": score_lines([{"id": "x", "ground_truth": "one", "prediction": "on"}]),
                "test": score_lines([])}
    candidate = {"validation": score_lines([{"id": "x", "ground_truth": "one", "prediction": "one"}]),
                 "test": score_lines([])}
    assert compare_evaluations(baseline, candidate)[0]["character_error_delta"] == -1
    candidate["validation"]["lines"][0]["ground_truth"] = "other"
    with pytest.raises(ValueError, match="不一致"):
        compare_evaluations(baseline, candidate)


def test_sequence_loss_does_not_shift_labels_a_second_time_and_ignores_padding():
    import torch
    logits = torch.tensor([[[12., -12., -12.], [-12., 12., -12.], [-12., -12., 12.]]], requires_grad=True)
    labels = torch.tensor([[0, 1, -100]])
    loss = sequence_loss(logits, labels)
    assert loss.item() < .001
    loss.backward()
    assert torch.all(logits.grad[0, 2] == 0)
    assert torch.isfinite(logits.grad).all()


def test_worker_checks_snapshot_hash_and_line_image_hash(tmp_path):
    path, data = snapshot(tmp_path / "snapshot")
    processor = SimpleNamespace(tokenizer=Tokenizer())
    assert validate_dataset(path, processor, TrainingConfig(), smoke=True)["counts"]["train"] == 3
    (path.parent / data["samples"][0]["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="行图校验失败"):
        validate_dataset(path, processor, TrainingConfig(), smoke=True)


def test_worker_rejects_page_split_leakage_even_in_internal_smoke(tmp_path):
    path, data = snapshot(tmp_path / "snapshot")
    data["samples"][-1]["page_id"] = data["samples"][0]["page_id"]
    write_dataset(path.parent, data)
    with pytest.raises(ValueError, match="原始页面"):
        validate_dataset(path, SimpleNamespace(tokenizer=Tokenizer()), TrainingConfig(), smoke=True)


def test_worker_rejects_duplicate_images_and_path_escape(tmp_path):
    path, data = snapshot(tmp_path / "snapshot")
    data["samples"][-1]["image"] = data["samples"][0]["image"]
    data["samples"][-1]["image_sha256"] = data["samples"][0]["image_sha256"]
    write_dataset(path.parent, data)
    with pytest.raises(ValueError, match="重复行图"):
        validate_dataset(path, SimpleNamespace(tokenizer=Tokenizer()), TrainingConfig(), smoke=True)
    data["samples"][-1]["image"] = "../../outside.png"
    write_dataset(path.parent, data)
    with pytest.raises(ValueError, match="路径无效"):
        validate_dataset(path, SimpleNamespace(tokenizer=Tokenizer()), TrainingConfig(), smoke=True)


def test_worker_never_silently_truncates_long_ground_truth(tmp_path):
    path, data = snapshot(tmp_path / "snapshot")
    data["samples"][0]["text"] = "word " * 128
    write_dataset(path.parent, data)
    with pytest.raises(ValueError, match=r"超过\s*128"):
        validate_dataset(path, SimpleNamespace(tokenizer=Tokenizer()), TrainingConfig(), smoke=True)


def test_small_dataset_is_allowed_only_for_internal_smoke(tmp_path):
    path, data = snapshot(tmp_path / "snapshot")
    with pytest.raises(ValueError, match="150行"):
        validate_dataset(path, SimpleNamespace(tokenizer=Tokenizer()), TrainingConfig())


def test_worker_rejects_legacy_target_encoding_without_loading_or_changing_checkpoint(tmp_path, monkeypatch):
    from dataclasses import asdict
    import transformers
    from handwriter import training

    run = tmp_path / "home" / "runs" / "legacy"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoints" / "last" / "state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing optimizer checkpoint must remain unchanged")
    legacy_config = asdict(TrainingConfig())
    legacy_config.pop("target_label_encoding")
    (run / "run.json").write_text(json.dumps({"config": legacy_config}), encoding="utf-8")
    (run / "result.json").write_text('{"status":"cancelled"}', encoding="utf-8")
    monkeypatch.setattr(training, "models_ready", lambda _: True)
    monkeypatch.setattr(training, "_sha", lambda _: BASE_SHA256)
    monkeypatch.setattr(transformers.TrOCRProcessor, "from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(training, "validate_dataset", lambda *args, **kwargs: {
        "id": "snapshot", "writer_id": "writer", "snapshot_sha256": "snapshot-hash", "samples": []})
    monkeypatch.setattr(training, "_setup_base", lambda *args, **kwargs: pytest.fail("Model must not be loaded"))
    spec = {"home": str(tmp_path / "home"), "run_dir": str(run), "models_dir": str(tmp_path / "models"),
            "snapshot_path": str(tmp_path / "dataset.json"), "cancel_file": str(run / "cancel.request"),
            "resume": True}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = run_training(path)
    assert result["status"] == "failed"
    assert "标签编码" in result["error"]
    assert checkpoint.read_bytes() == b"existing optimizer checkpoint must remain unchanged"


def test_legacy_checkpoint_itself_is_rejected_before_optimizer_state_is_read(tmp_path):
    from handwriter.training import _recover_checkpoint
    checkpoint = tmp_path / "checkpoints" / "last"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"legacy optimizer state")
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"legacy adapter")
    (checkpoint / "checkpoint.json").write_text('{"schema_version":1,"files":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="标签编码版本"):
        _recover_checkpoint(tmp_path)
    assert (checkpoint / "state.pt").read_bytes() == b"legacy optimizer state"


def test_cancelled_before_loading_models_persists_result_and_jsonl(tmp_path):
    run = tmp_path / "home" / "runs" / "run"
    run.mkdir(parents=True)
    cancel = run / "cancel.request"
    cancel.touch()
    spec = {"home": str(tmp_path / "home"), "run_dir": str(run), "models_dir": str(tmp_path / "models"),
            "snapshot_path": str(tmp_path / "dataset.json"), "cancel_file": str(cancel), "resume": False}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = run_training(path)
    assert result["status"] == "cancelled"
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["status"] == "cancelled"
    events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "cancelled"


def test_active_worker_lock_rejects_duplicate_without_overwriting_result(tmp_path):
    from filelock import FileLock
    home = tmp_path / "home"
    run = home / "runs" / "run"
    run.mkdir(parents=True)
    (run / "result.json").write_text('{"status":"cancelled","error":"previous"}', encoding="utf-8")
    spec = {"home": str(home), "run_dir": str(run), "models_dir": str(tmp_path / "models"),
            "snapshot_path": str(tmp_path / "dataset.json"), "cancel_file": str(run / "cancel.request"), "resume": True}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with FileLock(str(home / "training.lock")):
        result = run_training(path)
    assert result["status"] == "failed"
    assert "已有训练进程" in result["error"]
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["error"] == "previous"


def test_finished_run_is_not_overwritten_by_resume_attempt(tmp_path):
    run = tmp_path / "home" / "runs" / "run"
    run.mkdir(parents=True)
    (run / "result.json").write_text('{"status":"completed","published_dir":"kept"}', encoding="utf-8")
    spec = {"home": str(tmp_path / "home"), "run_dir": str(run), "models_dir": str(tmp_path / "models"),
            "snapshot_path": str(tmp_path / "dataset.json"), "cancel_file": str(run / "cancel.request"), "resume": True}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert run_training(path)["status"] == "failed"
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_internal_offline_guard_records_and_blocks_dns_then_restores_socket():
    import socket
    original = socket.getaddrinfo
    attempts = []
    with _deny_network(attempts):
        with pytest.raises(OSError, match="禁止网络"):
            socket.getaddrinfo("models.example.invalid", 443)
    assert socket.getaddrinfo is original
    assert attempts == [{"operation": "getaddrinfo", "target": "models.example.invalid"}]
