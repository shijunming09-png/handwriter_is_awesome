"""Explicit author consent, immutable local datasets and model selection.

No function in this module performs network access or training. OCR output is
never training data until a person explicitly confirms its exact text and crop.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import threading
import unicodedata
from uuid import uuid4

from PIL import Image, ImageStat

from .data import DocumentProject, LineResult, PageResult, utc_now
from .storage import load_project, project_path, save_project

log = logging.getLogger(__name__)
_LOCK = threading.RLock()
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MIN_LINES = 150
MIN_PAGES = 5
MIN_SPLITS = {"train": 90, "validation": 20, "test": 20}
MAX_LABEL_TOKENS = 128
_PREPROCESSING = {
    "version": 1,
    "crop": "handwriter.engine.crop_line",
    "color": "RGB",
    "image_format": "PNG",
    "labels": "Unicode NFC and outer whitespace strip; no truncation",
    "max_label_tokens_including_special_tokens": MAX_LABEL_TOKENS,
    "split": "original PDF SHA256 + page index, rendered-page equality; fixed page holdouts",
}


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"本地记录损坏，未覆盖原文件：{path.name}") from exc


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("作者或模型编号无效。")
    return value


def _home(home) -> Path:
    return Path(home).resolve()


def load_writer(home) -> dict | None:
    profile = _read_json(_home(home) / "writers" / "profile.json")
    if profile is None:
        return None
    if not isinstance(profile, dict) or not profile.get("name"):
        raise ValueError("作者档案无效。")
    _safe_id(profile.get("id"))
    return profile


def save_writer(home, name: str) -> dict:
    name = str(name).strip()
    if not name or len(name) > 120 or any(ord(c) < 32 for c in name):
        raise ValueError("请输入 1–120 个字符的作者名称。")
    with _LOCK:
        profile = load_writer(home) or {"id": uuid4().hex, "created_at": utc_now()}
        profile.update(name=name, updated_at=utc_now())
        root = _home(home) / "writers"
        _write_json(root / profile["id"] / "profile.json", profile)
        _write_json(root / "profile.json", profile)
        return profile


def _writer_root(home, writer=None) -> Path:
    writer = writer or load_writer(home)
    if not writer:
        raise ValueError("请先建立作者档案，并将任务归入这位作者。")
    return _home(home) / "writers" / _safe_id(writer["id"])


def _registry(home, writer=None) -> dict:
    path = _writer_root(home, writer) / "registry.json"
    result = _read_json(path, {})
    if not isinstance(result, dict):
        raise ValueError("作者的任务及模型记录无效。")
    result.setdefault("version", 1)
    result.setdefault("projects", [])
    result.setdefault("models", {})
    result.setdefault("active_model_id", "base")
    result.setdefault("split_ledger", None)
    return result


def _save_registry(home, registry, writer=None):
    _write_json(_writer_root(home, writer) / "registry.json", registry)


def _portable_path(home, path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(_home(home)).as_posix()
    except ValueError:
        return str(path)


def _restore_path(home, path: str) -> Path:
    location = Path(path)
    return (location if location.is_absolute() else _home(home) / location).resolve()


def register_project(home, project: DocumentProject) -> None:
    with _LOCK:
        writer = load_writer(home)
        if not writer:
            raise ValueError("请先保存作者档案。")
        if project.writer_id != writer["id"]:
            for page in project.pages:
                for line in page.lines:
                    invalidate_line_review(line)
            project.writer_id = writer["id"]
            project.recognizer_id = "base"
        path = save_project(project)
        registry = _registry(home, writer)
        entry = {"id": project.id, "title": project.title, "path": _portable_path(home, path), "writer_id": writer["id"]}
        # The same task can be moved: explicit registration updates its path.
        registry["projects"] = [p for p in registry["projects"] if p.get("id") != project.id]
        registry["projects"].append(entry)
        _save_registry(home, registry, writer)


def list_projects(home) -> list[dict]:
    if not load_writer(home):
        return []
    records = []
    for record in _registry(home)["projects"]:
        item = dict(record)
        item["path"] = str(_restore_path(home, item["path"]))
        item["exists"] = Path(item["path"]).is_file()
        records.append(item)
    return records


@lru_cache(maxsize=256)
def _cached_hash(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_hash(str(path), stat.st_size, stat.st_mtime_ns)


def invalidate_line_review(line: LineResult) -> None:
    line.reviewed = False
    line.reviewed_hash = ""
    line.reviewed_at = ""


def _review_hash(project, page, line) -> str:
    if not project.writer_id or not page.image_path:
        raise ValueError("请先归入作者并生成页面预览。")
    page_hash = file_sha256(project_path(project, page.image_path))
    value = {
        "version": 1, "writer_id": project.writer_id,
        "project_id": project.id, "page_index": page.index,
        "rotation": page.rotation, "skew_angle": page.skew_angle,
        "page_image_sha256": page_hash, "line_id": line.id,
        "polygon": line.polygon, "text": line.text,
    }
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_line_review(project, page, line, confirmed: bool) -> None:
    if confirmed:
        if line.status in {"pending", "stale", "error"}:
            raise ValueError("请先完成识别或人工转写，再确认此行。")
        if not line.text.strip():
            raise ValueError("空文字不能确认为训练样本。")
        line.reviewed_hash = _review_hash(project, page, line)
        line.reviewed = True
        line.reviewed_at = utc_now()
    else:
        invalidate_line_review(line)
    save_project(project)


def is_line_reviewed(project, page, line) -> bool:
    if not line.reviewed or not line.reviewed_hash:
        return False
    try:
        return line.reviewed_hash == _review_hash(project, page, line)
    except (OSError, TypeError, ValueError):
        return False


@lru_cache(maxsize=8)
def _tokenizer_cached(directory: str, vocab_size: int, vocab_mtime: int):
    # This only loads local tokenizer text files; never an online model lookup.
    from transformers import RobertaTokenizer
    return RobertaTokenizer.from_pretrained(directory, local_files_only=True)


def _load_tokenizer(models_dir: Path):
    from .model_manager import RECOGNIZER_DIRNAME
    directory = Path(models_dir) / RECOGNIZER_DIRNAME
    vocab = directory / "vocab.json"
    stat = vocab.stat()
    return _tokenizer_cached(str(directory.resolve()), stat.st_size, stat.st_mtime_ns)


def _base_hash(models_dir: Path) -> str:
    from .model_manager import RECOGNIZER_DIRNAME
    directory = Path(models_dir) / RECOGNIZER_DIRNAME
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = directory / name
        if path.is_file():
            return file_sha256(path)
    raise ValueError("通用 TrOCR 模型尚未准备好，请先准备本地模型。")


def _label(text: str, tokenizer) -> str:
    if not isinstance(text, str) or any(c in text for c in "\r\n\x00"):
        raise ValueError("标签含换行或控制字符；一个样本必须只对应一行。")
    if any(unicodedata.category(c) == "Cc" for c in text):
        raise ValueError("标签含控制字符。")
    text = unicodedata.normalize("NFC", text).strip()
    if not text:
        raise ValueError("文字为空。")
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(ids) > MAX_LABEL_TOKENS:
        raise ValueError(f"标签超过 {MAX_LABEL_TOKENS} 个 token，请左右拆行后重新确认。")
    return text


@dataclass
class _Sample:
    id: str
    project_id: str
    page_index: int
    page_id: str
    page_image_sha256: str
    source_pdf_sha256: str
    line_id: str
    image_sha256: str
    text: str
    pixels: bytes
    reviewed_hash: str = ""
    reviewed_at: str = ""
    recognition_model_id: str = "base"
    split: str = ""


def _collect(home, models_dir):
    writer = load_writer(home)
    result = {"writer": writer, "confirmed_lines": 0, "eligible_lines": 0, "pages": 0,
              "counts": {name: 0 for name in MIN_SPLITS}, "exclusions": [], "ready": False,
              "reason": "请先建立作者档案。", "projects": [], "base_model_sha256": ""}
    if writer is None:
        return result, [], None
    result["projects"] = list_projects(home)
    tokenizer = None
    model_error = ""
    try:
        tokenizer = _load_tokenizer(Path(models_dir))
        result["base_model_sha256"] = _base_hash(Path(models_dir))
    except (OSError, ValueError, ImportError) as exc:
        model_error = "请先准备本地通用模型及 tokenizer。" + str(exc).splitlines()[0][:100]
    candidates = []

    def exclude(project_id, page_index, line_id, reason):
        result["exclusions"].append({"project_id": project_id, "page_index": page_index, "line_id": line_id, "reason": reason})

    for record in result["projects"]:
        try:
            project = load_project(record["path"])
            if project.writer_id != writer["id"]:
                raise ValueError("任务不属于当前作者，需重新归入并逐行确认。")
            source_hash = file_sha256(project_path(project, project.source_pdf))
        except (OSError, ValueError, TypeError) as exc:
            exclude(record["id"], None, None, "任务不可用：" + str(exc))
            continue
        for page in project.pages:
            if not page.image_path:
                for line in page.lines:
                    exclude(project.id, page.index, line.id, "页面尚未渲染。")
                continue
            try:
                page_path = project_path(project, page.image_path)
                page_hash = file_sha256(page_path)
                with Image.open(page_path) as source:
                    image = source.convert("RGB")
            except (OSError, ValueError) as exc:
                for line in page.lines:
                    exclude(project.id, page.index, line.id, "页面图不可用：" + str(exc))
                continue
            page_id = hashlib.sha256(f"{source_hash}:{page.index}".encode()).hexdigest()
            for line in page.lines:
                if not is_line_reviewed(project, page, line):
                    exclude(project.id, page.index, line.id, "确认已失效，请重新确认。" if line.reviewed else "尚未逐行确认。")
                    continue
                result["confirmed_lines"] += 1
                if line.training_excluded:
                    exclude(project.id, page.index, line.id, "已标记不用于训练（例如无法确定的字）。")
                    continue
                if line.status in {"pending", "stale", "error"}:
                    exclude(project.id, page.index, line.id, "行结果待更新或识别失败。")
                    continue
                if tokenizer is None:
                    exclude(project.id, page.index, line.id, "本地 tokenizer 尚不可用，无法验证标签长度。")
                    continue
                try:
                    text = _label(line.text, tokenizer)
                    from .engine import crop_line
                    crop = crop_line(image, line.polygon).convert("RGB")
                    gray = crop.convert("L")
                    histogram = gray.histogram()
                    ink = sum(histogram[:220]) / max(1, sum(histogram))
                    if min(crop.size) < 4 or ImageStat.Stat(gray).stddev[0] < 1.2 or ink < 0.0001:
                        raise ValueError("行图为空白或过小，无法用于训练。")
                    payload = io.BytesIO()
                    crop.save(payload, format="PNG")
                    pixels = payload.getvalue()
                    digest = hashlib.sha256(pixels).hexdigest()
                    sample_id = hashlib.sha256(f"{page_id}:{line.id}:{digest}:{text}".encode("utf-8")).hexdigest()
                    candidates.append(_Sample(sample_id, project.id, page.index, page_id, page_hash, source_hash,
                                              line.id, digest, text, pixels, reviewed_hash=line.reviewed_hash,
                                              reviewed_at=line.reviewed_at, recognition_model_id=line.recognition_model_id))
                except (OSError, ValueError, TypeError) as exc:
                    exclude(project.id, page.index, line.id, str(exc))
            image.close()
    # Reject every label for identical pixels with conflicting ground truth.
    by_image = defaultdict(list)
    for sample in candidates:
        by_image[sample.image_sha256].append(sample)
    filtered = []
    for samples in by_image.values():
        if len({sample.text for sample in samples}) > 1:
            for sample in samples:
                exclude(sample.project_id, sample.page_index, sample.line_id, "同一行图有相互冲突的转写，已全部排除。")
        else:
            filtered.extend(samples)
    registry = _registry(home, writer)
    ledger = registry.get("split_ledger")
    assigned, proposed = _assign_splits(filtered, ledger)
    kept = []
    seen = set()
    # A previously held-out duplicate must never migrate into training.
    prior_images = (ledger or {}).get("image_splits", {})
    priority = {"test": 0, "validation": 1, "train": 2}
    for sample in sorted(assigned, key=lambda x: (priority[x.split], x.page_id, x.id)):
        previous_split = prior_images.get(sample.image_sha256)
        if previous_split and previous_split != sample.split:
            exclude(sample.project_id, sample.page_index, sample.line_id, "该重复图像属于已固定的其他数据分组，已排除以避免泄漏。")
            continue
        if sample.image_sha256 in seen:
            exclude(sample.project_id, sample.page_index, sample.line_id, "完全重复的行图，已去重。")
            continue
        seen.add(sample.image_sha256)
        kept.append(sample)
    counts = Counter(sample.split for sample in kept)
    result["eligible_lines"] = len(kept)
    result["pages"] = len({sample.page_id for sample in kept})
    result["counts"] = {name: counts.get(name, 0) for name in MIN_SPLITS}
    if model_error:
        result["reason"] = model_error
    elif result["eligible_lines"] < MIN_LINES or result["pages"] < MIN_PAGES:
        result["reason"] = f"至少需要 {MIN_LINES} 个有效确认行、{MIN_PAGES} 个原稿页；目前 {len(kept)} 行、{result['pages']} 页。"
    elif any(counts.get(name, 0) < minimum for name, minimum in MIN_SPLITS.items()):
        result["reason"] = "按原稿页固定分组后，至少需要训练 90 行、验证 20 行、测试 20 行；请补充确认样本。"
    else:
        result["ready"] = True
        result["reason"] = "样本满足门槛，可以开始训练。"
    return result, kept, proposed


def _assign_splits(samples: list[_Sample], ledger: dict | None):
    assignments = dict((ledger or {}).get("pages", {}))
    image_groups = dict((ledger or {}).get("page_image_groups", {}))
    canonical = {}
    for sample in sorted(samples, key=lambda x: x.page_id):
        group = image_groups.get(sample.page_image_sha256, sample.page_id)
        image_groups.setdefault(sample.page_image_sha256, group)
        canonical[sample.page_id] = group
    groups = sorted(set(canonical.values()))
    if ledger is None:
        # With five equal-size pages this produces 3/1/1 pages. At least one
        # page per holdout is reserved before applying the production gate.
        validation_count = max(1, int(round(len(groups) * 0.2))) if len(groups) >= 3 else 0
        test_count = max(1, int(round(len(groups) * 0.2))) if len(groups) >= 3 else 0
        train_count = max(0, len(groups) - validation_count - test_count)
        assignments.update({group: "train" if i < train_count else "validation" if i < train_count + validation_count else "test" for i, group in enumerate(groups)})
    for group in groups:
        assignments.setdefault(group, "train")
    for sample in samples:
        sample.split = assignments[canonical[sample.page_id]]
    proposed = {"version": 1, "created_at": (ledger or {}).get("created_at", utc_now()),
                "pages": assignments, "page_image_groups": image_groups,
                "image_splits": dict((ledger or {}).get("image_splits", {}))}
    return samples, proposed


def summarize_dataset(home, models_dir: Path | None = None) -> dict:
    return _collect(home, Path(models_dir) if models_dir is not None else _home(home) / "models")[0]


def create_snapshot(home, models_dir, enforce_minimum: bool = True) -> Path:
    with _LOCK:
        summary, samples, ledger = _collect(home, Path(models_dir))
        if not summary["writer"]:
            raise ValueError(summary["reason"])
        if enforce_minimum and not summary["ready"]:
            raise ValueError(summary["reason"])
        if not samples or not summary["base_model_sha256"]:
            raise ValueError("没有可用于训练的有效样本，或通用模型尚未准备好。")
        snapshot_id = uuid4().hex
        root = _writer_root(home) / "datasets"
        directory = root / snapshot_id
        temporary = root / ("." + snapshot_id + ".tmp")
        (temporary / "images").mkdir(parents=True)
        records = []
        for sample in samples:
            relative_image = f"images/{sample.id}.png"
            image_path = temporary / relative_image
            image_path.write_bytes(sample.pixels)
            if file_sha256(image_path) != sample.image_sha256:
                raise ValueError("保存数据集图像时校验失败。")
            records.append({"id": sample.id, "image": relative_image, "text": sample.text,
                            "split": sample.split, "page_id": sample.page_id,
                            "image_sha256": sample.image_sha256, "page_image_sha256": sample.page_image_sha256,
                            "source_pdf_sha256": sample.source_pdf_sha256, "project_id": sample.project_id,
                            "page_index": sample.page_index, "line_id": sample.line_id,
                            "reviewed_hash": sample.reviewed_hash, "reviewed_at": sample.reviewed_at,
                            "recognition_model_id": sample.recognition_model_id})
            ledger["image_splits"][sample.image_sha256] = sample.split
        dataset = {"id": snapshot_id, "writer_id": summary["writer"]["id"], "created_at": utc_now(),
                   "base_model_sha256": summary["base_model_sha256"], "samples": records,
                   "counts": summary["counts"], "preprocessing": dict(_PREPROCESSING)}
        _write_json(temporary / "dataset.json", dataset)
        digest = file_sha256(temporary / "dataset.json")
        (temporary / "snapshot.sha256").write_text(digest + "\n", encoding="ascii")
        registry = _registry(home)
        registry["split_ledger"] = ledger
        _save_registry(home, registry)
        temporary.rename(directory)
        verify_snapshot(directory / "dataset.json")
        return directory / "dataset.json"


def verify_snapshot(path) -> dict:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "dataset.json"
    root = path.parent
    expected = (root / "snapshot.sha256").read_text(encoding="ascii").strip()
    if file_sha256(path) != expected:
        raise ValueError("训练快照记录已改变，不能继续使用。")
    data = _read_json(path)
    for sample in data["samples"]:
        image = (root / sample["image"]).resolve()
        if not image.is_relative_to(root) or file_sha256(image) != sample["image_sha256"]:
            raise ValueError("训练快照图像已改变或路径无效。")
    return data


def _model_available(home, record) -> bool:
    try:
        from .model_validation import validate_recognizer_dir
        root = _writer_root(home)
        path = _restore_path(home, record["published_dir"])
        if not path.is_relative_to(root):
            return False
        validate_recognizer_dir(path)
        weights = path / "model.safetensors"
        return weights.stat().st_size > 0 and (not record.get("model_sha256") or file_sha256(weights) == record["model_sha256"])
    except (OSError, ValueError, TypeError, KeyError):
        return False


def register_model(home, result: dict) -> dict:
    with _LOCK:
        writer = load_writer(home)
        if not writer or result.get("writer_id") != writer["id"]:
            raise ValueError("模型与当前作者不匹配，不能注册。")
        model_id = _safe_id(result.get("model_id") or result.get("run_id") or result.get("id"))
        if model_id == "base":
            raise ValueError("自训练模型不能覆盖通用模型。")
        published = _restore_path(home, str(result.get("published_dir", "")))
        if not published.is_relative_to(_writer_root(home, writer)):
            raise ValueError("发布模型必须位于当前作者的本地目录中。")
        metadata = _read_json(published / "metadata.json", {})
        if metadata.get("writer_id") not in {None, writer["id"]}:
            raise ValueError("发布模型的作者记录不匹配。")
        for field in ("snapshot_id", "base_model_sha256"):
            if result.get(field) and metadata.get(field) and result[field] != metadata[field]:
                raise ValueError("发布模型与训练结果记录不匹配：" + field)
        record = {"id": model_id, "run_id": result.get("run_id", model_id),
                  "writer_id": writer["id"], "title": result.get("title") or f"{writer['name']} · {utc_now()[:16]}",
                  "created_at": result.get("created_at") or utc_now(),
                  "base_model_sha256": result.get("base_model_sha256") or metadata.get("base_model_sha256"),
                  "snapshot_id": result.get("snapshot_id") or metadata.get("snapshot_id"),
                  "metrics": result.get("metrics", {}), "improved": bool(result.get("improved")),
                  "published_dir": _portable_path(home, published)}
        if (published / "model.safetensors").is_file():
            record["model_sha256"] = file_sha256(published / "model.safetensors")
        if not _model_available(home, record):
            raise ValueError("发布模型缺少必要文件，未注册不完整模型。")
        registry = _registry(home, writer)
        if model_id in registry["models"]:
            previous = registry["models"][model_id]
            if previous.get("model_sha256") != record.get("model_sha256"):
                raise ValueError("该模型编号已注册，不能覆盖已有模型。")
            return {**previous, "available": True, "active": get_active_model(home) == model_id}
        registry["models"][model_id] = record
        _save_registry(home, registry, writer)
        return {**record, "available": True, "active": False}


def list_models(home) -> list[dict]:
    writer = load_writer(home)
    active = get_active_model(home) if writer else "base"
    result = [{"id": "base", "title": "通用 TrOCR", "available": True, "active": active == "base"}]
    if writer:
        for record in sorted(_registry(home, writer)["models"].values(), key=lambda x: x.get("created_at", ""), reverse=True):
            result.append({**record, "available": _model_available(home, record), "active": active == record["id"]})
    return result


def resolve_recognizer(home, model_id: str) -> Path | None:
    if model_id == "base" or not load_writer(home):
        return None
    record = _registry(home)["models"].get(model_id)
    if not record or not _model_available(home, record):
        return None
    return _restore_path(home, record["published_dir"])


def get_active_model(home) -> str:
    if not load_writer(home):
        return "base"
    registry = _registry(home)
    selected = registry.get("active_model_id", "base")
    return selected if selected != "base" and resolve_recognizer(home, selected) is not None else "base"


def set_active_model(home, model_id: str) -> None:
    with _LOCK:
        _safe_id(model_id)
        if model_id != "base" and resolve_recognizer(home, model_id) is None:
            raise ValueError("该模型不可用，已保留原来的模型选择。")
        if not load_writer(home):
            if model_id == "base":
                return
            raise ValueError("请先建立作者档案。")
        registry = _registry(home)
        registry["active_model_id"] = model_id
        _save_registry(home, registry)


def list_runs(home) -> list[dict]:
    if not load_writer(home):
        return []
    runs = _writer_root(home) / "runs"
    if not runs.is_dir():
        return []
    result = []
    for directory in runs.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            config = _read_json(directory / "run.json", {})
            outcome = _read_json(directory / "result.json", {})
            if not config and not outcome:
                continue
            record = {**config, **outcome}
            record.setdefault("id", directory.name)
            record.setdefault("run_id", directory.name)
            record.setdefault("status", "interrupted")
            record["run_dir"] = str(directory.resolve())
            result.append(record)
        except ValueError:
            result.append({"id": directory.name, "run_id": directory.name, "status": "error", "error": "训练记录损坏", "run_dir": str(directory.resolve())})
    return sorted(result, key=lambda x: x.get("created_at", x.get("started_at", "")), reverse=True)
