from __future__ import annotations

import json

import pytest

from handwriter.label_encoding import TARGET_LABEL_ENCODING, encode_target_labels


@pytest.fixture
def tokenizer(tmp_path):
    """Real offline Roberta byte-BPE tokenizer, with the TrOCR special IDs."""
    from transformers import RobertaTokenizer
    from transformers.models.roberta.tokenization_roberta import bytes_to_unicode

    vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    for character in bytes_to_unicode().values():
        vocabulary[character] = len(vocabulary)
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.txt"
    vocab_path.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    return RobertaTokenizer(vocab_file=str(vocab_path), merges_file=str(merges_path))


def test_target_encoding_has_text_then_one_eos_without_bos(tokenizer):
    assert TARGET_LABEL_ENCODING == "text_then_eos_v1"
    assert (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) == (0, 1, 2)
    assert tokenizer("Hi", add_special_tokens=True)["input_ids"] == [0, 44, 77, 2]
    assert encode_target_labels(tokenizer, "Hi") == [44, 77, 2]


def test_empty_text_has_only_eos_not_bos(tokenizer):
    # The sample validators, not the encoding helper, reject blank labels.
    assert encode_target_labels(tokenizer, "") == [2]


@pytest.mark.parametrize("text", ["  I  received your letter.  ", "Cafe\u0301 & King's — £50.", " "])
def test_encoding_preserves_whitespace_unicode_and_punctuation(tokenizer, text):
    labels = encode_target_labels(tokenizer, text)
    assert labels[-1] == 2 and labels.count(2) == 1 and 0 not in labels
    assert tokenizer.decode(labels[:-1], clean_up_tokenization_spaces=False) == text


def test_decoder_shift_uses_eos_start_and_predicts_text_first(tokenizer):
    import torch
    from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right

    labels = torch.tensor([encode_target_labels(tokenizer, "Hi")])
    decoder_inputs = shift_tokens_right(labels, pad_token_id=1, decoder_start_token_id=2)
    assert labels.tolist() == [[44, 77, 2]]
    assert decoder_inputs.tolist() == [[2, 44, 77]]
    assert decoder_inputs.ne(tokenizer.pad_token_id).all()
    # Explicitly expose the old incompatible prefix; do not change start to BOS.
    old_labels = torch.tensor([tokenizer("Hi", add_special_tokens=True)["input_ids"]])
    assert shift_tokens_right(old_labels, 1, 2).tolist() == [[2, 0, 44, 77]]


def test_helper_disables_automatic_special_tokens_and_truncation():
    class RecordingTokenizer:
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            assert text == "  literal text  "
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return {"input_ids": [10, 11]}

    assert encode_target_labels(RecordingTokenizer(), "  literal text  ") == [10, 11, 2]


def test_missing_eos_is_rejected():
    class MissingEos:
        eos_token_id = None

    with pytest.raises(ValueError, match="EOS"):
        encode_target_labels(MissingEos(), "text")


def test_non_string_is_rejected(tokenizer):
    with pytest.raises(TypeError, match="string"):
        encode_target_labels(tokenizer, None)
