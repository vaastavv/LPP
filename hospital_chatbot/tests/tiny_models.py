"""Build tiny, randomly initialised Transformer checkpoints locally (no network) for tests/smoke runs."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast


def build_tokenizer(texts: list[str], out: Path) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(
        vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    from tokenizers.processors import TemplateProcessing
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(out)
    return fast


def build_tiny_bert(texts: list[str], out: Path, labels: list[str]) -> Path:
    """A 2-layer BERT checkpoint with the same on-disk layout as a Hugging Face hub model."""
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(texts, out)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=128, num_labels=len(labels),
                     id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    BertForSequenceClassification(cfg).save_pretrained(out)
    return out


if __name__ == "__main__":
    import sys
    import pandas as pd
    from src.config import INTENTS
    df = pd.read_csv("data/processed/train.csv")
    print(build_tiny_bert(df["text"].tolist(), Path(sys.argv[1]), INTENTS))
