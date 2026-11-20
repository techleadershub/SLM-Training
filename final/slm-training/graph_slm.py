"""Helpers for Day 2: call the small language model from inside a LangGraph node.

    from graph_slm import slm, ask_json, parse_tool_calls, TOOL_PROMPT_NOTE

`slm(messages)` returns the model's reply text. It loads the model once per notebook (about 6 GB for 3B, a few seconds on a GPU)
and decodes greedily, so the same input gives the same output: that is what makes a graph testable.
"""
import json
import re

import torch

from slm_lab import device, load

# The 3B model on a GPU; on a machine without a GPU the 0.5B model, so the notebooks still run (answers are weaker).
MODEL = "Qwen/Qwen2.5-3B-Instruct" if device() == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct"
_loaded = {}
CALLS = 0                      # how many times the model was called (the labs read this to count calls per message)


def model():
    if "m" not in _loaded:
        _loaded["tok"], _loaded["m"] = load(MODEL)
    return _loaded["tok"], _loaded["m"]


def use(model_id):
    """Switch the model used by `slm` (frees the old one first). Handy to compare sizes in one notebook."""
    global MODEL
    _loaded.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    MODEL = model_id


@torch.inference_mode()
def slm(messages, tools=None, max_new_tokens=160):
    """One reply from the model. `messages` is a list of {"role", "content"}; `tools` is a list of tool descriptions."""
    global CALLS
    CALLS += 1
    tok, m = model()
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    tok.padding_side = "left"
    enc = tok(text, return_tensors="pt").to(m.device)
    out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, top_k=None, temperature=None, top_p=None,
                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_json(text):
    """The first JSON object in a model reply, or None if there is none or it is broken."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def ask_json(system, user, max_new_tokens=120):
    """Ask for a JSON reply. Returns (parsed_dict_or_None, raw_text)."""
    raw = slm([{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens=max_new_tokens)
    return extract_json(raw), raw


def parse_tool_calls(text):
    """Tool calls in a reply: [{"name": ..., "arguments": {...}}, ...]. Qwen writes <tool_call>{json}</tool_call>."""
    calls = []
    for chunk in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            c = json.loads(chunk)
            if isinstance(c, dict) and "name" in c:
                calls.append({"name": c["name"], "arguments": c.get("arguments") or {}})
        except json.JSONDecodeError:
            pass
    return calls
