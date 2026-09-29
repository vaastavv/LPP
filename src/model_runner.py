from __future__ import annotations
import torch


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelRunner:
    def __init__(self, model_id: str, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.device = device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def chat(self, prompt: str, gen: dict) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
            input_ids = enc["input_ids"].to(self.device)
        except Exception:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model.generate(
            input_ids,
            do_sample=gen.get("do_sample", False),
            max_new_tokens=gen.get("max_new_tokens", 256),
            repetition_penalty=gen.get("repetition_penalty", 1.0),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def signals(self, text: str, use_chat_template: bool = False) -> dict:
        from .latent_metrics import token_entropy, profile_hidden_states
        if use_chat_template:
            try:
                enc = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True,
                    return_tensors="pt", return_dict=True)
                input_ids = enc["input_ids"].to(self.device)
            except Exception:
                input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        else:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model(input_ids, output_hidden_states=True)
        logits = out.logits.squeeze(0)
        ent = token_entropy(logits)
        prof = profile_hidden_states(out.hidden_states)
        prof.update({
            "min_entropy": float(ent.min()),
            "mean_entropy": float(ent.mean()),
            "num_tokens": int(input_ids.shape[1]),
            "num_layers": len(out.hidden_states),
        })
        return prof
