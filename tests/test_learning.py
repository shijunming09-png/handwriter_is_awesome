from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont
import pytest

from handwriter import learning
from handwriter.data import DocumentProject, LineResult, PageResult
from handwriter.storage import load_project, save_project


class Tokenizer:
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        assert kwargs.get("truncation") is False
        ids = list(range(10, 10 + len(text.split())))
        return {"input_ids": [0] + ids + [2] if kwargs.get("add_special_tokens", True) else ids}


@pytest.fixture
def learning_env(tmp_path, monkeypatch):
    home = tmp_path / "便携应用"
    base = home / "models" / "trocr-base-handwritten"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(b"small base weights for isolated data tests")
    monkeypatch.setattr(learning, "_load_tokenizer", lambda _: Tokenizer())
    learning.save_writer(home, "Jane Example")
    return home, home / "models"


def make_project(home, pages=5, lines=30, *, title="Test manuscript"):
    project_id = uuid4().hex
    directory = home / "projects" / project_id
    (directory / "pages").mkdir(parents=True)
    (directory / "source.pdf").write_bytes(b"pdf page identity " + project_id.encode())
    project = DocumentProject(project_id, title, "source.pdf", str(directory), [])
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    for index in range(pages):
        page = PageResult(index, width=650, height=lines * 32 + 100, image_path=f"pages/{index}.png", status="recognized")
        image = Image.new("RGB", (page.width, page.height), "white")
        draw = ImageDraw.Draw(image)
        for row in range(lines):
            top = 35 + row * 32
            text = f"Page {index} row {row}: original handwritten content."
            draw.text((25, top + 2), text, fill="black", font=font)
            line = LineResult(id=f"page{index}-line{row}", polygon=[[15, top], [600, top], [600, top + 27], [15, top + 27]], raw_text=text, status="recognized")
            page.lines.append(line)
        image.save(directory / page.image_path)
        project.pages.append(page)
    save_project(project)
    learning.register_project(home, project)
    return project


def confirm_all(project):
    for page in project.pages:
        for line in page.lines:
            learning.set_line_review(project, page, line, True)


def test_v1_open_is_read_only_then_first_save_keeps_exact_backup(tmp_path):
    project_dir = tmp_path / "legacy"
    project_dir.mkdir()
    (project_dir / "source.pdf").write_bytes(b"legacy copied source")
    value = {
        "id": "legacy", "title": "Old task", "source_pdf": "source.pdf", "work_dir": str(project_dir),
        "schema_version": 1, "writer_id": "should-not-be-trusted", "recognizer_id": "custom-old",
        "pages": [{"index": 0, "lines": [{"id": "l", "polygon": [[1, 1], [20, 1], [20, 10], [1, 10]],
                   "raw_text": "original", "edited_text": "human edit", "status": "recognized",
                   "reviewed": True, "reviewed_hash": "untrusted v1 consent"}]}],
    }
    original = json.dumps(value, ensure_ascii=False, indent=3).encode("utf-8")
    (project_dir / "project.json").write_bytes(original)
    project = load_project(project_dir)
    assert project.schema_version == 2
    assert project.writer_id is None and project.recognizer_id == "base"
    assert project.pages[0].lines[0].text == "human edit"
    assert not project.pages[0].lines[0].reviewed
    assert not (project_dir / "project.v1.backup.json").exists()
    save_project(project)
    assert (project_dir / "project.v1.backup.json").read_bytes() == original
    assert json.loads((project_dir / "project.json").read_text(encoding="utf-8"))["schema_version"] == 2
    save_project(project)
    assert (project_dir / "project.v1.backup.json").read_bytes() == original


@pytest.mark.parametrize("change", ["text", "polygon", "rotation", "writer", "image"])
def test_explicit_review_covers_exact_text_crop_rotation_and_author(learning_env, change):
    home, _ = learning_env
    project = make_project(home, pages=1, lines=1)
    page, line = project.pages[0], project.pages[0].lines[0]
    # Correct unedited OCR is eligible only after explicit confirmation.
    assert line.edited_text is None
    learning.set_line_review(project, page, line, True)
    assert learning.is_line_reviewed(project, page, line)
    if change == "text":
        line.edited_text = "A later correction"
    elif change == "polygon":
        line.polygon[0][0] += 1
    elif change == "rotation":
        page.rotation = 90
    elif change == "writer":
        project.writer_id = "another-writer"
    else:
        path = Path(project.work_dir) / page.image_path
        image = Image.open(path).convert("RGB")
        image.putpixel((40, 40), (100, 100, 100))
        image.save(path)
    assert not learning.is_line_reviewed(project, page, line)
    learning.invalidate_line_review(line)
    assert not line.reviewed and line.reviewed_hash == "" and line.reviewed_at == ""


def test_preview_before_minimum_does_not_freeze_empty_holdouts(learning_env):
    home, models = learning_env
    project = make_project(home)
    for line in project.pages[0].lines:
        learning.set_line_review(project, project.pages[0], line, True)
    early = learning.summarize_dataset(home, models)
    assert early["eligible_lines"] == 30 and not early["ready"]
    registry_path = home / "writers" / project.writer_id / "registry.json"
    assert json.loads(registry_path.read_text(encoding="utf-8"))["split_ledger"] is None
    with pytest.raises(ValueError, match="至少需要"):
        learning.create_snapshot(home, models)
    confirm_all(project)
    summary = learning.summarize_dataset(home, models)
    assert summary["ready"] and summary["eligible_lines"] == 150 and summary["pages"] == 5
    assert summary["counts"] == {"train": 90, "validation": 30, "test": 30}
    snapshot = learning.create_snapshot(home, models)
    data = learning.verify_snapshot(snapshot)
    assert data["writer_id"] == project.writer_id
    assert data["counts"] == summary["counts"]
    assert all(sample["image_sha256"] and sample["page_id"] and sample["id"] for sample in data["samples"])


def test_fixed_holdouts_new_pages_train_and_old_snapshot_unchanged(learning_env):
    home, models = learning_env
    project = make_project(home, pages=6)
    for page in project.pages[:5]:
        for line in page.lines:
            learning.set_line_review(project, page, line, True)
    first_path = learning.create_snapshot(home, models)
    first_bytes = first_path.read_bytes()
    first = learning.verify_snapshot(first_path)
    initial = {sample["page_id"]: sample["split"] for sample in first["samples"]}
    for line in project.pages[5].lines:
        learning.set_line_review(project, project.pages[5], line, True)
    second = learning.verify_snapshot(learning.create_snapshot(home, models))
    assert first_path.read_bytes() == first_bytes
    assert learning.verify_snapshot(first_path)["id"] == first["id"]
    for sample in second["samples"]:
        assert sample["split"] == initial.get(sample["page_id"], "train")
    assert second["counts"] == {"train": 120, "validation": 30, "test": 30}
    assert len({s["image_sha256"] for s in second["samples"]}) == len(second["samples"])


def test_identical_pixels_deduplicate_and_conflicting_labels_are_all_excluded(learning_env):
    home, models = learning_env
    project = make_project(home, pages=1, lines=2)
    page = project.pages[0]
    line = page.lines[0]
    duplicate = LineResult(id="duplicate", polygon=[point[:] for point in line.polygon], raw_text=line.text, status="recognized")
    page.lines.append(duplicate)
    confirm_all(project)
    result = learning.summarize_dataset(home, models)
    assert result["confirmed_lines"] == 3 and result["eligible_lines"] == 2
    assert any("去重" in e["reason"] for e in result["exclusions"])
    duplicate.edited_text = "Conflicting ground truth for the same image"
    learning.set_line_review(project, page, duplicate, True)
    result = learning.summarize_dataset(home, models)
    assert result["eligible_lines"] == 1
    assert sum("冲突" in e["reason"] for e in result["exclusions"]) == 2


def test_copied_pdf_projects_share_page_groups_and_never_cross_holdouts(learning_env):
    import shutil
    home, models = learning_env
    project = make_project(home)
    confirm_all(project)
    clone_dir = home / "projects" / "copied-task"
    shutil.copytree(Path(project.work_dir), clone_dir)
    clone = load_project(clone_dir)
    clone.id = "copied-task"
    learning.register_project(home, clone)
    confirm_all(clone)
    summary = learning.summarize_dataset(home, models)
    assert summary["confirmed_lines"] == 300
    assert summary["eligible_lines"] == 150 and summary["pages"] == 5
    assert summary["counts"] == {"train": 90, "validation": 30, "test": 30}
    data = learning.verify_snapshot(learning.create_snapshot(home, models))
    groups = {}
    for row in data["samples"]:
        groups.setdefault(row["page_id"], set()).add(row["split"])
    assert len(groups) == 5 and all(len(split) == 1 for split in groups.values())


def test_reject_invalid_labels_exclusions_blank_and_stale_confirmation(learning_env):
    home, models = learning_env
    project = make_project(home, pages=1, lines=5)
    page = project.pages[0]
    page.lines[0].edited_text = "two\nlines"
    page.lines[1].edited_text = " ".join(["token"] * 129)
    page.lines[2].training_excluded = True
    # The fourth crop points to blank margin but carries nonempty OCR text.
    page.lines[3].polygon = [[15, page.height - 35], [600, page.height - 35], [600, page.height - 5], [15, page.height - 5]]
    confirm_all(project)
    page.lines[4].edited_text = "changed after confirmation"
    save_project(project)
    result = learning.summarize_dataset(home, models)
    assert result["confirmed_lines"] == 4 and result["eligible_lines"] == 0
    reasons = " ".join(x["reason"] for x in result["exclusions"])
    assert "换行" in reasons and "128" in reasons and "不用于训练" in reasons and "空白" in reasons and "失效" in reasons


def test_snapshot_integrity_checks_reject_later_changes(learning_env):
    home, models = learning_env
    project = make_project(home, pages=1, lines=1)
    confirm_all(project)
    snapshot = learning.create_snapshot(home, models, enforce_minimum=False)
    data = learning.verify_snapshot(snapshot)
    image = snapshot.parent / data["samples"][0]["image"]
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="图像已改变"):
        learning.verify_snapshot(snapshot)


def write_published_model(home, writer_id, run_id="run-one"):
    from handwriter.model_validation import BASE_MODEL_SHA256
    directory = home / "writers" / writer_id / "runs" / run_id / "published"
    directory.mkdir(parents=True)
    values = {
        "config.json": {"model_type": "vision-encoder-decoder", "decoder": {"model_type": "trocr"}},
        "preprocessor_config.json": {}, "tokenizer_config.json": {}, "vocab.json": {}, "generation_config.json": {},
    }
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"isolated registry fixture, not an inference model")
    (directory / "metadata.json").write_text(json.dumps({"writer_id": writer_id, "snapshot_id": "snapshot-one", "base_model_sha256": BASE_MODEL_SHA256}), encoding="utf-8")
    files = {p.name: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in directory.iterdir()}
    (directory / "model-manifest.json").write_text(json.dumps({"schema_version": 1, "base_model_sha256": BASE_MODEL_SHA256, "files": files}), encoding="utf-8")
    return directory


def test_model_registration_is_not_activation_and_invalid_selection_falls_back(learning_env):
    from handwriter.model_validation import BASE_MODEL_SHA256
    home, _ = learning_env
    writer = learning.load_writer(home)
    published = write_published_model(home, writer["id"])
    record = learning.register_model(home, {"run_id": "run-one", "writer_id": writer["id"], "published_dir": str(published), "snapshot_id": "snapshot-one", "base_model_sha256": BASE_MODEL_SHA256, "metrics": {"candidate": {"test": {"cer": .1}}}})
    assert record["id"] == "run-one"
    assert learning.get_active_model(home) == "base"
    learning.set_active_model(home, "run-one")
    assert learning.get_active_model(home) == "run-one"
    assert learning.resolve_recognizer(home, "run-one") == published.resolve()
    assert next(m for m in learning.list_models(home) if m["id"] == "run-one")["active"]
    (published / "generation_config.json").write_text('{"changed": true}', encoding="utf-8")
    assert learning.get_active_model(home) == "base"
    assert learning.resolve_recognizer(home, "run-one") is None
    with pytest.raises(ValueError, match="不可用"):
        learning.set_active_model(home, "run-one")
    with pytest.raises(ValueError, match="覆盖通用"):
        learning.register_model(home, {"run_id": "base", "writer_id": writer["id"], "published_dir": str(published)})


def test_cross_author_registration_clears_review_and_generic_model(learning_env):
    home, _ = learning_env
    project = make_project(home, pages=1, lines=1)
    confirm_all(project)
    project.writer_id = "old-author"
    project.recognizer_id = "old-personal-model"
    learning.register_project(home, project)
    assert project.writer_id == learning.load_writer(home)["id"]
    assert project.recognizer_id == "base"
    assert not project.pages[0].lines[0].reviewed


def test_internal_registry_paths_are_portable_and_external_paths_supported(learning_env, tmp_path):
    import shutil
    home, _ = learning_env
    project = make_project(home, pages=1, lines=1)
    registry = json.loads((home / "writers" / project.writer_id / "registry.json").read_text(encoding="utf-8"))
    assert not Path(registry["projects"][0]["path"]).is_absolute()
    moved = tmp_path / "moved-app"
    shutil.copytree(home, moved)
    listing = learning.list_projects(moved)
    assert listing[0]["exists"] and Path(listing[0]["path"]).is_relative_to(moved)
    external = tmp_path / "external-task"
    shutil.copytree(Path(project.work_dir), external)
    loaded = load_project(external)
    learning.register_project(home, loaded)
    assert Path(learning.list_projects(home)[0]["path"]) == external / "project.json"
