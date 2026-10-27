"""Target sequence convention for the pinned TrOCR recognizer."""
from __future__ import annotations


TARGET_LABEL_ENCODING = "text_then_eos_v1"


def encode_target_labels(tokenizer, text: str) -> list[int]:
    """Encode unchanged text followed by EOS, without an automatic BOS.

    TrOCR starts its decoder with EOS (2); its first prediction is a text
    token. Roberta's default special-token wrapper would insert BOS (0)
    before that text and change the pretrained decoder's input prefix.

    Text validation and normalization belong to the caller. In particular,
    empty text encodes to EOS alone; this does not make it a valid sample.
    """
    if not isinstance(text, str):
        raise TypeError("Target text must be a string.")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The target tokenizer must define an EOS token.")
    labels = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return list(labels) + [eos_token_id]
