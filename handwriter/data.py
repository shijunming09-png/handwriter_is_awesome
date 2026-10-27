from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LineResult:
    id: str = field(default_factory=lambda: uuid4().hex)
    polygon: list[list[float]] = field(default_factory=list)
    raw_text: str = ""
    edited_text: str | None = None
    status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    detection_score: float = 0.0
    reviewed: bool = False
    reviewed_hash: str = ""
    reviewed_at: str = ""
    recognition_model_id: str = "base"
    training_excluded: bool = False

    @property
    def text(self) -> str:
        return self.edited_text if self.edited_text is not None else self.raw_text


@dataclass
class PageResult:
    index: int
    width: int = 0
    height: int = 0
    image_path: str = ""
    rotation: int = 0
    skew_angle: float = 0.0
    lines: list[LineResult] = field(default_factory=list)
    status: str = "pending"
    error: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentProject:
    id: str
    title: str
    source_pdf: str
    work_dir: str
    pages: list[PageResult]
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    dpi: int = 300
    schema_version: int = 2
    writer_id: str | None = None
    recognizer_id: str = "base"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentProject":
        schema = value.get("schema_version", 1)
        if schema not in {1, 2}:
            raise ValueError("该任务使用了不支持的格式版本。")
        pages = []
        for p in value.get("pages", []):
            page_fields = {k: v for k, v in p.items() if k in PageResult.__dataclass_fields__ and k != "lines"}
            lines = []
            for ln in p.get("lines", []):
                fields = {k: v for k, v in ln.items() if k in LineResult.__dataclass_fields__}
                # Version-one files never carry training consent, even if they
                # contain stray fields inserted by another tool.
                if schema == 1:
                    fields.update(reviewed=False, reviewed_hash="", reviewed_at="", recognition_model_id="base", training_excluded=False)
                poly = fields.get("polygon", [])
                if len(poly) != 4 or any(len(point) != 2 for point in poly):
                    raise ValueError("任务中的文字行坐标无效。")
                if any(not isinstance(coordinate, (int, float)) or not math.isfinite(coordinate)
                       for point in poly for coordinate in point):
                    raise ValueError("任务中的文字行坐标必须是有效数字。")
                lines.append(LineResult(**fields))
            pages.append(PageResult(**page_fields, lines=lines))
        if [page.index for page in pages] != list(range(len(pages))):
            raise ValueError("任务中的页面顺序无效。")
        fields = {k: v for k, v in value.items() if k in cls.__dataclass_fields__ and k != "pages"}
        if schema == 1:
            fields.update(writer_id=None, recognizer_id="base")
        fields["schema_version"] = 2
        return cls(**fields, pages=pages)
