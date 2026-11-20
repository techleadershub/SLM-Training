"""Shared helpers for the SLM training labs.

Only boilerplate lives here (device setup, loading, a chat wrapper, timing, scoring) so the notebooks can focus
on the idea each lab teaches. Read it once -- nothing in here is magic.
"""
from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"            # adapters, merged models and reports written by the labs
OUT.mkdir(exist_ok=True)

# Keep downloaded models on the persistent volume on Runpod (/workspace survives a Pod stop).
if "HF_HOME" not in os.environ and Path("/workspace").is_dir():
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

# SLM_FAST=1 shortens the slow cells (fewer training steps, smaller benchmarks) for a quick check run.
FAST = os.environ.get("SLM_FAST") == "1"

MODEL_ID = os.environ.get("SLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

# The HR assistant's system prompt: used for the baseline (Lab 4), in the training data (Lab 6) and at inference.
HR_SYSTEM = "You are the HR assistant of ABC Company. Answer the employee's question in one or two sentences."
UNKNOWN_REPLY = "The HR policy doesn't cover that. Please contact the HR helpdesk through the internal HR portal."

import torch  # noqa: E402  (after HF_HOME is set)
from transformers.utils import logging as _hf_logging  # noqa: E402

_hf_logging.disable_progress_bar()   # keep notebook output readable


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype():
    """bfloat16 on GPUs that support it, float16 on older GPUs, float32 on CPU."""
    if device() == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def gpu_report() -> dict:
    info = {"device": device(), "torch": torch.__version__}
    if device() == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 1), bf16=torch.cuda.is_bf16_supported(),
                    cuda=torch.version.cuda)
    return info


def load(model_id: str = MODEL_ID, dtype=None, **kwargs):
    """Load a tokenizer + causal LM onto the best available device, in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if "device_map" not in kwargs:
        kwargs["device_map"] = device()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype or default_dtype(), **kwargs)
    model.eval()
    return tok, model


def to_messages(prompt, system: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if system and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system})
    return messages


@torch.inference_mode()
def chat(model, tok, prompts, system: str | None = None, max_new_tokens: int = 128, temperature: float = 0.0,
         top_p: float = 1.0, seed: int | None = None):
    """Chat with the model. `prompts` is one question (str), a message list, or a list of questions (batched).

    temperature=0 means greedy decoding (same answer every time).
    """
    single = isinstance(prompts, str) or (prompts and isinstance(prompts[0], dict))
    batch = [prompts] if single else list(prompts)
    texts = [tok.apply_chat_template(to_messages(p, system), tokenize=False, add_generation_prompt=True)
             for p in batch]
    tok.padding_side = "left"           # decoder-only models generate after the last token, so pad on the left
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    if seed is not None:
        torch.manual_seed(seed)
    sampling = temperature > 0
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=sampling,
                         temperature=temperature if sampling else None, top_p=top_p if sampling else None,
                         top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    answers = [a.strip() for a in answers]
    return answers[0] if single else answers


STRICT = os.environ.get("SLM_STRICT") == "1"   # run_all.py sets this: a failed check stops the notebook


def check(condition, ok: str, hint: str):
    """Exercise checker: prints ✅ when the condition holds, otherwise a ⚠️ hint.

    It never stops the notebook (so "Run All" always reaches the end), except under SLM_STRICT=1.
    """
    if not condition:
        if STRICT:
            raise AssertionError("❌ Check failed. Hint: " + hint)
        print("⚠️ Unexpected result. " + hint + "\n   If you changed a 🔧 value, that may be why: "
              "put the original value back and run the cell again.")
        return
    print("✅ " + ok)


class Timer:
    """with Timer() as t: ...  then t.seconds"""

    def __enter__(self):
        if device() == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if device() == "cuda":
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start


def model_size_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def count_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def encode_chat(tok, messages: list[dict], max_len: int = 512) -> dict:
    """Tokenize one training conversation with loss only on the final assistant reply (reference for Lab 6).

    labels = -100 (ignored by the loss) for every prompt token; the reply tokens (incl. the end-of-turn token)
    are learned.
    """
    full = tok.apply_chat_template(messages, tokenize=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def read_jsonl(path) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- course data (ABC Company HR policy)

def policy_text() -> str:
    return (DATA / "hr_policy.md").read_text(encoding="utf-8")


def policy_sections() -> list[dict]:
    """Split the policy into its numbered sections: [{'title': '4.3 Paternity Leave', 'text': ...}, ...]."""
    sections, title, lines = [], None, []
    for line in policy_text().splitlines():
        m = re.match(r"#{2,3} (\d+(?:\.\d+)?\.? .+)", line)
        if m:
            if title and "".join(lines).strip():
                sections.append({"title": title, "text": " ".join(l.strip() for l in lines if l.strip())})
            title, lines = m.group(1).rstrip("."), []
        elif title:
            lines.append(line)
    if title and "".join(lines).strip():
        sections.append({"title": title, "text": " ".join(l.strip() for l in lines if l.strip())})
    return sections


def load_facts() -> list[dict]:
    return json.loads((DATA / "hr_facts.json").read_text(encoding="utf-8"))


def eval_items() -> list[dict]:
    """The held-out evaluation questions (never used for training): [{'id', 'question', 'expect', 'kind'}]."""
    items = []
    for f in load_facts():
        for q in f["eval"]:
            items.append({"id": f["id"], "question": q, "expect": f["expect"], "kind": f["kind"]})
    return items


def _norm(s: str) -> str:
    s = s.lower().replace(",", "").replace("’", "'")
    return re.sub(r"\s+", " ", s)


def is_correct(answer: str, expect: list[str]) -> bool:
    """Correct if the answer contains every expected fact. Each entry may list alternatives with '|'."""
    a = _norm(answer)
    return all(any(_norm(alt) in a for alt in group.split("|")) for group in expect)


def run_eval(answer_fn, items=None, batch_size: int = 8, show: int = 0):
    """Ask every eval question through answer_fn(list_of_questions) -> list_of_answers.

    Returns (accuracy, pandas.DataFrame of results).
    """
    import pandas as pd

    items = items or eval_items()
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        answers = answer_fn([it["question"] for it in chunk])
        for it, ans in zip(chunk, answers):
            rows.append({**it, "answer": ans, "correct": is_correct(ans, it["expect"])})
    df = pd.DataFrame(rows)
    if show:
        print(df[["question", "answer", "correct"]].head(show).to_string())
    return df["correct"].mean(), df


def save_report(name: str, data: dict):
    """Labs record their headline numbers in outputs/report.json; the capstone compares them."""
    path = OUT / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report[name] = data
    path.write_text(json.dumps(report, indent=2))
    return report


def load_report() -> dict:
    path = OUT / "report.json"
    return json.loads(path.read_text()) if path.exists() else {}
