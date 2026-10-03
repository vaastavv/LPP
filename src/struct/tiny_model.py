"""
A tiny, locally-constructed model for validating the structural-profiling
pipeline end to end WITHOUT downloading any weights.

It builds:
  * a byte-level BPE fast tokenizer trained on the corpus (so offset mapping
    works, which the span->token alignment requires), and
  * a small randomly-initialised LlamaForCausalLM.

The resulting object is API-compatible with `src.model_runner.ModelRunner`
(exposes `.model`, `.tokenizer`, `.device`, `.model_id`, `.chat`, `.signals`).

IMPORTANT: weights are random, so the NUMBERS this produces are a pipeline
smoke-test, not scientific results. Real numbers require pretrained CodeLLMs
(run on a machine with Hugging Face access; see scripts/run_colab.py).
"""
from __future__ import annotations

import torch


def _train_tokenizer(texts: list[str], vocab_size: int = 2000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>", pad_token="<pad>",
        bos_token="<bos>", eos_token="<eos>",
    )
    return fast


class TinyRunner:
    def __init__(self, texts: list[str], hidden_size: int = 128,
                 num_layers: int = 6, num_heads: int = 4, seed: int = 42):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(seed)
        self.device = "cpu"
        self.model_id = "tiny-random-llama"
        self.tokenizer = _train_tokenizer(texts)
        cfg = LlamaConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=1024,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.model = LlamaForCausalLM(cfg)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    @torch.no_grad()
    def chat(self, prompt: str, gen: dict) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model.generate(
            ids, do_sample=False,
            max_new_tokens=gen.get("max_new_tokens", 16),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def build_tiny_runner(seed: int = 42, hidden_size: int = 128,
                      num_layers: int = 6, num_heads: int = 4) -> TinyRunner:
    from .corpus import build_corpus
    from .transforms import make_contrast_pairs
    progs = build_corpus(42)             # fixed corpus regardless of model seed
    texts = [p.source for p in progs]
    for p in progs:                      # include transformed variants in the vocab
        pr = make_contrast_pairs(p.source)
        for v in pr.values():
            if v:
                texts.append(v[1])
    return TinyRunner(texts, hidden_size=hidden_size, num_layers=num_layers,
                      num_heads=num_heads, seed=seed)
