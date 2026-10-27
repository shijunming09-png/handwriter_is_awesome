"""Strict, reproducible handwriting recognition evaluation."""
from __future__ import annotations

from typing import Iterable, Sequence


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    previous = list(range(len(prediction) + 1))
    for row, value in enumerate(reference, 1):
        current = [row]
        for column, other in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (value != other)))
        previous = current
    return previous[-1]


def score_lines(lines: Iterable[dict]) -> dict:
    """CER includes spaces/case/punctuation; WER uses whitespace tokenization."""
    records = []
    for row in lines:
        text, prediction = str(row["ground_truth"]), str(row["prediction"])
        ce = edit_distance(text, prediction)
        words = text.split()
        we = edit_distance(words, prediction.split())
        records.append({**row, "character_errors": ce, "reference_characters": len(text),
                        "word_errors": we, "reference_words": len(words),
                        "cer": ce / max(len(text), 1), "wer": we / max(len(words), 1)})
    ce = sum(row["character_errors"] for row in records)
    chars = sum(row["reference_characters"] for row in records)
    we = sum(row["word_errors"] for row in records)
    words = sum(row["reference_words"] for row in records)
    return {"cer": ce / max(chars, 1), "wer": we / max(words, 1),
            "character_errors": ce, "reference_characters": chars,
            "word_errors": we, "reference_words": words, "sample_count": len(records),
            "lines": records,
            "rules": "strict: case and punctuation preserved, CER includes spaces, WER splits on whitespace"}


def compare_evaluations(baseline: dict, candidate: dict) -> list[dict]:
    differences = []
    for split in ("validation", "test"):
        original = {row["id"]: row for row in baseline.get(split, {}).get("lines", [])}
        for row in candidate.get(split, {}).get("lines", []):
            before = original.get(row["id"])
            if before is None or before["ground_truth"] != row["ground_truth"]:
                raise ValueError("候选模型与通用模型的评估样本不一致。")
            differences.append({"id": row["id"], "split": split,
                                "ground_truth": row["ground_truth"],
                                "baseline": before["prediction"], "candidate": row["prediction"],
                                "baseline_cer": before["cer"], "candidate_cer": row["cer"],
                                "baseline_wer": before["wer"], "candidate_wer": row["wer"],
                                "character_error_delta": row["character_errors"] - before["character_errors"]})
        if len(original) != len(candidate.get(split, {}).get("lines", [])):
            raise ValueError("候选模型与通用模型的评估样本数量不一致。")
    return differences
