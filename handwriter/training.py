"""Local, resumable writer adaptation with decoder LoRA.

The original Microsoft weights are read only.  Runtime inference uses an
independently published, merged standard VisionEncoderDecoderModel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import socket
import time
import uuid

import numpy as np
from PIL import Image

from .model_manager import CancelledError, RECOGNIZER_DIRNAME, models_ready
from .training_metrics import compare_evaluations, score_lines
from .label_encoding import TARGET_LABEL_ENCODING, encode_target_labels

log = logging.getLogger(__name__)
BASE_SHA256 = "25a40cddc7e6120140a3d5b9e3dd3878a92ada7b4f312953ab22edc19c2a5acc"
TARGET_PATTERN = r"decoder\.model\.decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|v_proj)"


@dataclass(frozen=True)
class TrainingConfig:
    target_label_encoding: str = TARGET_LABEL_ENCODING
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_epochs: int = 20
    patience: int = 3
    max_target_tokens: int = 128
    num_beams: int = 3
    seed: int = 42
    max_grad_norm: float = 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _deny_network(attempts: list):
    """Internal smoke-test guard; never modifies system networking."""
    originals = {"connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
                 "getaddrinfo": socket.getaddrinfo}

    def blocked(operation):
        def call(*args, **kwargs):
            target = args[-1] if operation != "getaddrinfo" else args[0] if args else ""
            attempts.append({"operation": operation, "target": str(target)[:200]})
            raise OSError("内部离线验收禁止网络连接。")
        return call
    socket.socket.connect = blocked("connect")
    socket.socket.connect_ex = blocked("connect_ex")
    socket.getaddrinfo = blocked("getaddrinfo")
    try:
        yield
    finally:
        socket.socket.connect = originals["connect"]
        socket.socket.connect_ex = originals["connect_ex"]
        socket.getaddrinfo = originals["getaddrinfo"]


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _safe_remove(path: Path, root: Path) -> None:
    target, parent = path.resolve(), root.resolve()
    if target == parent or not target.is_relative_to(parent):
        raise ValueError("拒绝删除训练目录之外的路径。")
    if target.is_dir():
        shutil.rmtree(target)


def _replace_directory(temporary: Path, destination: Path, root: Path) -> None:
    if not temporary.resolve().is_relative_to(root.resolve()) or not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("训练文件目录无效。")
    backup = destination.with_name(destination.name + ".previous")
    _safe_remove(backup, root)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    _safe_remove(backup, root)


class _Events:
    def __init__(self, directory: Path, emit):
        self.path = directory / "events.jsonl"
        self.emit = emit

    def __call__(self, event="progress", stage="train", message="", **extra):
        value = {"time": _now(), "event": event, "stage": stage, "message": message, **extra}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        if self.emit:
            self.emit(value)


def validate_dataset(snapshot_path: Path, processor, config: TrainingConfig, smoke=False) -> dict:
    """Revalidate immutable files, labels and split grouping inside the worker."""
    path = snapshot_path.resolve()
    if path.is_dir():
        path = path / "dataset.json"
    root = path.parent
    expected = (root / "snapshot.sha256").read_text(encoding="ascii").strip()
    if _sha(path) != expected:
        raise ValueError("训练快照记录校验失败，数据可能已改变。")
    data = _read(path)
    if not isinstance(data, dict) or not data.get("id") or not data.get("writer_id"):
        raise ValueError("训练快照缺少作者或快照编号。")
    if data.get("base_model_sha256") != BASE_SHA256:
        raise ValueError("训练快照使用的通用模型版本不一致。")
    rows, ids, hashes, pages, page_hashes = [], set(), {}, {}, {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for source in data.get("samples", []):
        row = dict(source)
        identifier = row.get("id")
        split, page_id = row.get("split"), row.get("page_id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("训练样本编号为空或重复。")
        if split not in counts or not isinstance(page_id, str) or not page_id:
            raise ValueError("训练样本缺少有效分组或原始页编号。")
        ids.add(identifier)
        if page_id in pages and pages[page_id] != split:
            raise ValueError("同一个原始页面被分配到多个数据组，可能造成评估泄漏。")
        pages[page_id] = split
        page_hash = row.get("page_image_sha256")
        if page_hash:
            if page_hash in page_hashes and page_hashes[page_hash] != split:
                raise ValueError("重复原始页面跨数据组，可能造成评估泄漏。")
            page_hashes[page_hash] = split
        relative = Path(str(row.get("image", "")))
        image_path = (root / relative).resolve()
        if relative.is_absolute() or not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError("训练行图路径无效或越出快照目录。")
        actual_hash = _sha(image_path)
        if actual_hash != row.get("image_sha256"):
            raise ValueError("训练行图校验失败，文件可能已改变。")
        if actual_hash in hashes:
            raise ValueError("训练集包含重复行图，请去重后重新创建快照。")
        hashes[actual_hash] = split
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text or "\x00" in text:
            raise ValueError("训练转写必须为非空的单行逐字文本。")
        labels = encode_target_labels(processor.tokenizer, text)
        if len(labels) > config.max_target_tokens:
            raise ValueError(f"样本 {identifier} 超过 {config.max_target_tokens} token，请拆分文字行。")
        with Image.open(image_path) as image:
            image = image.convert("L")
            if min(image.size) < 2 or image.width * image.height > 100_000_000:
                raise ValueError("训练行图尺寸异常。")
            pixels = np.asarray(image)
            if float(pixels.std()) < 1.2 or float(np.mean(pixels < 220)) < .0001:
                raise ValueError("训练行图为空白或没有清晰文字。")
        row["image_path"] = str(image_path)
        row["labels"] = labels
        rows.append(row)
        counts[split] += 1
    if any(counts[split] == 0 for split in counts):
        raise ValueError("训练、验证和测试组都必须包含独立样本。")
    if data.get("counts") != counts:
        raise ValueError("训练快照的样本数量记录不一致。")
    if not smoke and (len(rows) < 150 or len(pages) < 5 or counts["train"] < 90
                      or counts["validation"] < 20 or counts["test"] < 20):
        raise ValueError("训练需要至少150行、5个原始页面；训练/验证/测试至少90/20/20行。")
    return {**data, "samples": rows, "snapshot_sha256": expected, "source_path": str(path)}


def sequence_loss(logits, labels):
    """Aligned seq2seq CE; decoder inputs have already been shifted once.

    Explicit loss avoids Transformers versions whose generic causal LM loss
    shifts labels a second time for VisionEncoderDecoderModel.
    """
    import torch.nn.functional as functional
    return functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                    labels.reshape(-1), ignore_index=-100)


def _fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _setup_base(base_dir: Path, processor):
    import torch
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(str(base_dir), local_files_only=True, use_safetensors=True)
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder.use_cache = False
    model.to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _evaluate(model, processor, rows, config, cancelled, event, split, stage):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return cancelled()

    model.eval()
    values = []
    for index, row in enumerate(rows, 1):
        if cancelled():
            raise CancelledError("已取消训练；已完成的检查点会保留。")
        with Image.open(row["image_path"]) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to("cuda")
        with torch.inference_mode():
            ids = model.generate(pixels, num_beams=config.num_beams, max_new_tokens=config.max_target_tokens,
                                 do_sample=False, stopping_criteria=StoppingCriteriaList([Stop()]))
        if cancelled():
            raise CancelledError("已取消训练；已完成的检查点会保留。")
        prediction = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        tokens = ids[0].detach().cpu().tolist()
        eos = processor.tokenizer.eos_token_id
        truncated = len(tokens) - 1 >= config.max_target_tokens and eos not in tokens[1:]
        values.append({"id": row["id"], "page_id": row["page_id"], "ground_truth": row["text"],
                       "prediction": prediction, "truncated": truncated})
        event(stage=stage, message=f"{split}：{index}/{len(rows)} 行", split=split,
              current=index, total=len(rows))
    result = score_lines(values)
    event(event="evaluation", stage=stage, message=f"{split} CER {result['cer']:.2%}",
          split=split, cer=result["cer"], wer=result["wer"])
    return result


def _rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _save_adapter(model, destination: Path, run_dir: Path):
    temporary = destination.with_name("." + destination.name + ".tmp-" + uuid.uuid4().hex)
    model.save_pretrained(str(temporary), safe_serialization=True, save_embedding_layers=False)
    _replace_directory(temporary, destination, run_dir)


def _checkpoint(model, optimizer, scaler, state: dict, run_dir: Path):
    import torch
    root = run_dir / "checkpoints"
    root.mkdir(exist_ok=True)
    state["peak_gpu_mb"] = max(state.get("peak_gpu_mb", 0.0), _peak_gpu_mb())
    temporary = root / (".last.tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    model.save_pretrained(str(temporary / "adapter"), safe_serialization=True, save_embedding_layers=False)
    value = {**state, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state()}
    torch.save(value, temporary / "state.pt")
    files = {str(path.relative_to(temporary)).replace("\\", "/"): {"size": path.stat().st_size, "sha256": _sha(path)}
             for path in temporary.rglob("*") if path.is_file()}
    _json(temporary / "checkpoint.json", {"schema_version": 1, "target_label_encoding": TARGET_LABEL_ENCODING,
                                         "epoch": state["epoch"], "next_batch": state["next_batch"],
                                         "global_step": state["global_step"], "time": _now(), "files": files})
    _replace_directory(temporary, root / "last", root)


def _recover_checkpoint(run_dir: Path) -> Path | None:
    last = run_dir / "checkpoints" / "last"
    backup = last.with_name("last.previous")
    if not last.exists() and backup.exists():
        os.replace(backup, last)
    if (last / "state.pt").is_file() and (last / "adapter" / "adapter_model.safetensors").is_file():
        metadata = _read(last / "checkpoint.json", {})
        if metadata.get("target_label_encoding") != TARGET_LABEL_ENCODING:
            raise ValueError("检查点标签编码版本不兼容，不能恢复旧训练。请创建新任务。")
        files = metadata.get("files", {})
        if "state.pt" not in files or "adapter/adapter_model.safetensors" not in files:
            raise ValueError("检查点缺少完整性记录，不能恢复。")
        for relative, record in files.items():
            path = (last / relative).resolve()
            if not path.is_relative_to(last.resolve()) or not path.is_file():
                raise ValueError("检查点文件路径无效。")
            if path.stat().st_size != record["size"] or _sha(path) != record["sha256"]:
                raise ValueError("检查点文件校验失败，不能恢复损坏的数据。")
        return last
    return None


def _peak_gpu_mb() -> float:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2) if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def _train(spec: dict, config: TrainingConfig, event, result: dict) -> dict:
    import torch
    from transformers import TrOCRProcessor
    from peft import LoraConfig, PeftModel, get_peft_model

    run_dir = Path(spec["run_dir"]).resolve()
    models_dir = Path(spec["models_dir"]).resolve()
    cancel_file = Path(spec["cancel_file"]).resolve()
    cancelled = lambda: cancel_file.is_file()
    smoke = bool(spec.get("smoke", {}).get("enabled", False))
    event(stage="validate", message="检查本地模型、训练快照与可用磁盘…")
    if cancelled():
        raise CancelledError("已取消训练。")
    if not models_ready(models_dir):
        raise ValueError("通用模型尚未准备好，训练不会自动联网下载。")
    if shutil.disk_usage(run_dir).free < 4 * 1024 ** 3:
        raise OSError("磁盘可用空间不足4 GiB，请释放空间后重试；原模型不会改变。")
    base_dir = models_dir / RECOGNIZER_DIRNAME
    if _sha(base_dir / "model.safetensors") != BASE_SHA256:
        raise ValueError("通用模型权重校验失败。")
    processor = TrOCRProcessor.from_pretrained(str(base_dir), local_files_only=True, use_fast=False)
    dataset = validate_dataset(Path(spec["snapshot_path"]), processor, config, smoke=smoke)
    result.update(writer_id=dataset["writer_id"], snapshot_id=dataset["id"], base_model_sha256=BASE_SHA256,
                  snapshot_sha256=dataset["snapshot_sha256"], smoke=smoke)
    metadata = _read(run_dir / "run.json", {})
    fixed = {"writer_id": dataset["writer_id"], "snapshot_id": dataset["id"],
             "snapshot_sha256": dataset["snapshot_sha256"], "base_model_sha256": BASE_SHA256,
             "config": asdict(config), "smoke": smoke}
    if spec.get("resume"):
        if metadata.get("config", {}).get("target_label_encoding") != config.target_label_encoding:
            raise ValueError("训练标签编码约定已改变，不能恢复旧检查点。请从通用模型新建训练任务。")
        if any(key in metadata and metadata[key] != value for key, value in fixed.items()):
            raise ValueError("训练配置、作者或数据快照已改变，不能恢复旧检查点。")
    metadata.update(fixed)
    metadata.update(run_id=result["run_id"], created_at=metadata.get("created_at") or _now(),
                    status="running", process_pid=os.getpid())
    _json(run_dir / "run.json", metadata)
    rows = {split: [row for row in dataset["samples"] if row["split"] == split]
            for split in ("train", "validation", "test")}
    smoke_options = spec.get("smoke", {}) if smoke else {}
    for split in rows:
        limit = smoke_options.get("max_train_samples" if split == "train" else "max_eval_samples")
        if limit is not None:
            rows[split] = rows[split][:max(int(limit), 1)]
    if not torch.cuda.is_available():
        raise RuntimeError("训练需要可用的 NVIDIA GPU。仍可使用通用模型在 CPU 上识别。")
    torch.set_num_threads(min(max((os.cpu_count() or 4) // 2, 1), 8))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()
    result["device"] = torch.cuda.get_device_name(0)
    event(stage="load", message="加载通用 TrOCR，准备作者适配器…", device=result["device"])
    model = _setup_base(base_dir, processor)
    baseline_path = run_dir / "baseline.json"
    baseline = _read(baseline_path) if spec.get("resume") else None
    if baseline is None:
        baseline = {split: _evaluate(model, processor, rows[split], config, cancelled, event, split, "baseline")
                    for split in ("validation", "test")}
        _json(baseline_path, baseline)
    result["metrics"]["baseline"] = baseline
    checkpoint = _recover_checkpoint(run_dir) if spec.get("resume") else None
    if spec.get("resume") and checkpoint is None:
        raise ValueError("没有完整的last检查点，无法恢复；请创建新训练任务。")
    if checkpoint:
        model = PeftModel.from_pretrained(model, str(checkpoint / "adapter"), is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout, bias="none", task_type=None,
            target_modules=TARGET_PATTERN,
        ))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in parameters)
    if count != 761856 or len(model.targeted_module_names) != 48:
        raise ValueError(f"LoRA目标层或可训练参数异常：{count} 参数。")
    model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    result.update(trainable_parameters=count, target_modules=48, amp="bfloat16" if bf16 else "float16")
    state = {"epoch": 0, "next_batch": 0, "global_step": 0, "best_cer": None,
             "bad_epochs": 0, "history": [], "epoch_loss_sum": 0., "epoch_loss_count": 0,
             "initial_adapter_sha256": _fingerprint(model)}
    if checkpoint:
        # This is an application-generated local optimizer checkpoint, never a
        # user-selected foreign pickle.  Model weights themselves are safetensors.
        saved = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved.pop("optimizer"))
        scaler.load_state_dict(saved.pop("scaler"))
        rng = saved.pop("rng")
        state.update(saved)
        _restore_rng(rng)
        event(event="checkpoint", message="已恢复优化器、随机状态和训练位置。",
              epoch=state["epoch"], step=state["global_step"])
    else:
        _checkpoint(model, optimizer, scaler, state, run_dir)
    max_steps = int(smoke_options.get("max_optimizer_steps", 0))
    stopped = False
    epochs = () if state.get("training_done") or state["bad_epochs"] >= config.patience else range(state["epoch"], config.max_epochs)
    for epoch in epochs:
        order = list(range(len(rows["train"])))
        random.Random(config.seed + epoch).shuffle(order)
        start = state["next_batch"] if epoch == state["epoch"] else 0
        model.train()
        model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for position in range(start, len(order)):
            if cancelled():
                raise CancelledError("已取消训练；可从上次优化器更新后的安全检查点恢复。")
            row = rows["train"][order[position]]
            with Image.open(row["image_path"]) as image:
                pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to("cuda")
            labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda")
            # Variable length, batch=1: no synthetic padding is needed.  The
            # helper supports -100 padding when a future collator introduces it.
            decoder_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(pixel_values=pixels, decoder_input_ids=decoder_ids,
                               decoder_attention_mask=decoder_ids.ne(processor.tokenizer.pad_token_id),
                               use_cache=False).logits
                loss = sequence_loss(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失不是有效数字，已停止以保护检查点。")
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            accumulated += 1
            state["epoch_loss_sum"] += float(loss.detach().cpu())
            state["epoch_loss_count"] += 1
            event(message=f"第{epoch + 1}轮：{position + 1}/{len(order)} 行", epoch=epoch + 1,
                  step=state["global_step"], current=position + 1, total=len(order),
                  loss=float(loss.detach().cpu()), peak_gpu_mb=_peak_gpu_mb(), max_epochs=config.max_epochs)
            boundary = accumulated == config.gradient_accumulation_steps or position + 1 == len(order)
            if boundary:
                scaler.unscale_(optimizer)
                if accumulated < config.gradient_accumulation_steps:
                    correction = config.gradient_accumulation_steps / accumulated
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                state.update(epoch=epoch, next_batch=position + 1, global_step=state["global_step"] + 1)
                _checkpoint(model, optimizer, scaler, state, run_dir)
                event(event="optimizer_step", message="已保存安全检查点。", epoch=epoch + 1,
                      step=state["global_step"], loss=float(loss.detach().cpu()), peak_gpu_mb=_peak_gpu_mb())
                if cancelled():
                    raise CancelledError("已取消训练；最新安全检查点已保存。")
                if max_steps and state["global_step"] >= max_steps:
                    stopped = True
                    break
        validation = _evaluate(model, processor, rows["validation"], config, cancelled, event, "validation", "evaluate")
        cer = validation["cer"]
        improved_epoch = state["best_cer"] is None or cer < state["best_cer"]
        if improved_epoch:
            state["best_cer"], state["bad_epochs"] = cer, 0
            _save_adapter(model, run_dir / "adapter", run_dir)
        else:
            state["bad_epochs"] += 1
        mean_loss = state["epoch_loss_sum"] / max(state["epoch_loss_count"], 1)
        state["history"].append({"epoch": epoch + 1, "loss": mean_loss, "validation_cer": cer,
                                 "validation_wer": validation["wer"], "step": state["global_step"]})
        state.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0., epoch_loss_count=0,
                     training_done=state["bad_epochs"] >= config.patience or stopped or epoch + 1 >= config.max_epochs)
        _checkpoint(model, optimizer, scaler, state, run_dir)
        event(event="epoch", stage="evaluate", message=f"第{epoch + 1}轮结束，验证CER {cer:.2%}",
              epoch=epoch + 1, step=state["global_step"], loss=mean_loss, cer=cer, wer=validation["wer"],
              best_cer=state["best_cer"], peak_gpu_mb=_peak_gpu_mb(), max_epochs=config.max_epochs)
        if cancelled():
            raise CancelledError("已取消训练；最新检查点已保存。")
        if state["bad_epochs"] >= config.patience or stopped:
            break
    result.update(epochs=len(state["history"]), global_step=state["global_step"], history=state["history"],
                  adapter_updated=_fingerprint(model) != state["initial_adapter_sha256"],
                  peak_gpu_mb=max(state.get("peak_gpu_mb", 0.0), _peak_gpu_mb()))
    if not result["adapter_updated"]:
        raise RuntimeError("适配器参数没有发生更新，未发布无效训练结果。")
    # Free optimizer moments before replacing the last adapter by the best one.
    del optimizer, scaler, parameters
    gc.collect()
    model.load_adapter(str(run_dir / "adapter"), adapter_name="best", is_trainable=False, local_files_only=True)
    model.set_adapter("best")
    candidate = {split: _evaluate(model, processor, rows[split], config, cancelled, event, split, "evaluate")
                 for split in ("validation", "test")}
    result["metrics"]["candidate"] = candidate
    result["validation_improved"] = candidate["validation"]["cer"] < baseline["validation"]["cer"]
    result["improved"] = candidate["test"]["cer"] < baseline["test"]["cer"]
    evaluation = {"schema_version": 1, "snapshot_id": dataset["id"], "snapshot_sha256": dataset["snapshot_sha256"],
                  "metrics": result["metrics"], "differences": compare_evaluations(baseline, candidate),
                  "history": state["history"], "improved": result["improved"]}
    _json(run_dir / "evaluation.json", evaluation)
    if cancelled():
        raise CancelledError("已取消训练；检查点与评估结果已保留，尚未发布模型。")
    event(stage="publish", message="合并最佳适配器，发布独立的本地作者模型…")
    peak = max(state.get("peak_gpu_mb", 0.0), _peak_gpu_mb())
    model.to("cpu")
    torch.cuda.empty_cache()
    # Merge only the selected best adapter, never both the last and best one.
    merged = model.merge_and_unload(adapter_names=["best"], safe_merge=True)
    merged.eval()
    temporary = run_dir / (".published.tmp-" + uuid.uuid4().hex)
    merged.save_pretrained(str(temporary), safe_serialization=True)
    processor.save_pretrained(str(temporary))
    result["published_dir"] = str(run_dir / "published")
    result["adapter_dir"] = str(run_dir / "adapter")
    result["peak_gpu_mb"] = peak
    _json(temporary / "metadata.json", {"schema_version": 1, "run_id": result["run_id"],
                                      "writer_id": dataset["writer_id"], "snapshot_id": dataset["id"],
                                      "snapshot_sha256": dataset["snapshot_sha256"],
                                      "base_model_sha256": BASE_SHA256, "config": asdict(config),
                                      "created_at": _now(), "metrics": result["metrics"], "smoke": smoke})
    files = {str(path.relative_to(temporary)).replace("\\", "/"): {"sha256": _sha(path), "size": path.stat().st_size}
             for path in temporary.rglob("*") if path.is_file()}
    _json(temporary / "model-manifest.json", {"schema_version": 1, "base_model_sha256": BASE_SHA256, "files": files})
    _replace_directory(temporary, run_dir / "published", run_dir)
    result.update(status="completed", error="", finished_at=_now())
    _json(run_dir / "result.json", result)
    if not smoke:
        try:
            from .learning import register_model
            result["model_record"] = register_model(Path(spec["home"]), result)
        except Exception as exc:
            log.exception("The trained model was published but registration failed")
            result["registration_error"] = str(exc)
            event(stage="publish", message=f"模型已发布，但登记失败：{exc}")
    event(event="completed", stage="publish", message="训练完成。新模型已保存，尚未切换当前识别模型。",
          improved=result["improved"], peak_gpu_mb=peak)
    return result


def run_training(spec_path: Path, emit: Callable[[dict], None] | None = None) -> dict:
    """Run a headless training job and persist its terminal result."""
    from filelock import FileLock, Timeout
    spec = _read(Path(spec_path).resolve())
    if not isinstance(spec, dict):
        raise ValueError("无法读取训练任务说明。")
    for key in ("home", "models_dir", "snapshot_path", "run_dir", "cancel_file"):
        if not spec.get(key):
            raise ValueError(f"训练任务缺少 {key}。")
    run_dir = Path(spec["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    home = Path(spec["home"]).resolve()
    home.mkdir(parents=True, exist_ok=True)
    existing = _read(run_dir / "result.json", {})
    if spec.get("resume") and existing.get("status") == "completed":
        return {**existing, "status": "failed", "error": "已经完成的任务不能再次恢复。"}
    if not spec.get("resume") and existing:
        return {**existing, "status": "failed", "error": "此目录已有训练记录，请恢复未完成的任务或创建新任务。"}
    smoke_options = spec.get("smoke", {})
    overrides = {}
    if isinstance(smoke_options, dict) and smoke_options.get("enabled"):
        for key in ("max_epochs", "gradient_accumulation_steps", "patience"):
            if key in smoke_options:
                overrides[key] = max(1, int(smoke_options[key]))
    config = TrainingConfig(**overrides)
    result = {"status": "failed", "run_id": run_dir.name, "writer_id": "", "snapshot_id": "",
              "published_dir": "", "adapter_dir": "", "metrics": {"baseline": {}, "candidate": {}},
              "improved": False, "error": "", "peak_gpu_mb": 0., "created_at": _now()}
    # OS locks survive a GUI crash while its worker remains alive.  A rejected
    # duplicate never overwrites the active worker's result or event stream.
    global_lock = FileLock(str(home / "training.lock"), timeout=0)
    local_lock = FileLock(str(run_dir / "training.lock"), timeout=0)
    try:
        global_lock.acquire()
        local_lock.acquire()
    except Timeout:
        global_lock.release()
        return {**result, "error": "已有训练进程仍在运行，请等待该进程结束后再恢复或开始训练。"}
    event = _Events(run_dir, emit)
    network_attempts = []
    deny_network = bool(isinstance(smoke_options, dict) and smoke_options.get("enabled") and smoke_options.get("deny_network"))
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with _deny_network(network_attempts) if deny_network else nullcontext():
            result = _train(spec, config, event, result)
    except CancelledError as exc:
        result.update(status="cancelled", error=str(exc), peak_gpu_mb=_peak_gpu_mb(), finished_at=_now())
        event(event="cancelled", stage="train", message=str(exc), peak_gpu_mb=result["peak_gpu_mb"])
    except Exception as exc:
        log.exception("Writer training failed")
        text = str(exc)
        if "out of memory" in text.lower():
            text = "GPU显存不足，已停止训练并保留最新检查点；请关闭其他GPU程序后恢复。"
        elif isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            text = "磁盘空间不足，已停止训练并保留最新检查点。"
        result.update(status="failed", error=text, peak_gpu_mb=_peak_gpu_mb(), finished_at=_now())
        event(event="failed", stage="train", message=text, peak_gpu_mb=result["peak_gpu_mb"])
    finally:
        if deny_network:
            result["network_attempts"] = network_attempts
            result["offline_verified"] = not network_attempts
        _json(run_dir / "result.json", result)
        metadata = _read(run_dir / "run.json", {})
        metadata.update(status=result["status"], updated_at=_now(), process_pid=None)
        _json(run_dir / "run.json", metadata)
        local_lock.release()
        global_lock.release()
    return result
