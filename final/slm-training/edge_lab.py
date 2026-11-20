"""Day 3 helper: load the model, ask it many messages at once, and measure memory and time.

Plain Python on purpose: each function does one job and is short. Day 1 and Day 2 files are only imported, never changed.
"""
import json
import os
import time
from pathlib import Path

import edge_nlu
import slm_lab                     # sets the model cache folder on Runpod (this is the Day 1 helper)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = os.environ.get("EDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def memory_gb():
    """The most GPU memory used so far, in gigabytes."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def reset_memory_counter():
    """Start counting the peak memory again from now."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_edge_model(model_id=None, bits=16, adapter=None):
    """Load the tokenizer and the model.

    bits=16 keeps the normal 16-bit weights, bits=8 or bits=4 loads them in 8 or 4 bits (bitsandbytes).
    adapter is the folder of a LoRA adapter to attach, or None.
    """
    if model_id is None:
        model_id = MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"                       # for generation the padding goes on the left
    settings = {"device_map": "auto"}
    if bits == 4:
        settings["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    elif bits == 8:
        settings["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        settings["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **settings)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def model_size_mb(model):
    """How much memory the weights take, in megabytes."""
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes = total_bytes + parameter.numel() * parameter.element_size()
    return round(total_bytes / 1e6)


def make_prompt_text(tok, system, user_text, shots=None):
    """Build the full text the model sees: system message, optional worked examples, then the user's message.

    shots is a list of (message, correct_answer) pairs shown to the model before the real message.
    """
    messages = [{"role": "system", "content": system}]
    if shots is not None:
        for example_message, example_answer in shots:
            messages.append({"role": "user", "content": '"' + example_message + '"'})
            messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": '"' + user_text + '"'})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def count_tokens(tok, text):
    """How many tokens a text is."""
    return len(tok(text)["input_ids"])


def ask_many(tok, model, prompt_texts, max_new_tokens=60, batch_size=16):
    """Ask the model many prompts, a batch at a time, with greedy decoding (the same answer every time).

    Returns a dictionary: replies, prompt_tokens, answer_tokens (lists, one per prompt) and seconds (the total time).
    """
    replies = []
    prompt_tokens = []
    answer_tokens = []
    started = time.time()
    for start in range(0, len(prompt_texts), batch_size):
        batch = prompt_texts[start:start + batch_size]
        encoded = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():                       # we only read answers, we do not train
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                    top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        width = encoded["input_ids"].shape[1]
        for row in range(len(batch)):
            new_tokens = output[row, width:]
            text = tok.decode(new_tokens, skip_special_tokens=True).strip()
            replies.append(text)
            prompt_tokens.append(int(encoded["attention_mask"][row].sum()))
            answer_tokens.append(len(tok(text)["input_ids"]))
    seconds = time.time() - started
    return {"replies": replies, "prompt_tokens": prompt_tokens, "answer_tokens": answer_tokens, "seconds": seconds}


def ask_sampled(tok, model, prompt_text, how_many=3, temperature=0.7, max_new_tokens=60):
    """Ask the same prompt several times WITH randomness. Returns a list of replies (they can differ)."""
    encoded = tok([prompt_text], return_tensors="pt", padding=True).to(model.device)
    replies = []
    for _ in range(how_many):
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                    top_p=0.95, pad_token_id=tok.pad_token_id)
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        replies.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
    return replies


def mean(numbers):
    """The average of a list of numbers."""
    if len(numbers) == 0:
        return 0
    total = 0
    for number in numbers:
        total = total + number
    return total / len(numbers)


# ------------------------------------------------------------------------------------------------ the scoreboard
SCOREBOARD_FILE = Path(slm_lab.OUT) / "edge_scoreboard.json"


def load_scoreboard():
    """The scoreboard so far: a dictionary of rows, one row per step of the day. Empty if nothing is saved yet."""
    if SCOREBOARD_FILE.exists():
        with open(SCOREBOARD_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_row(step_name, row):
    """Add or replace one row of the scoreboard and save the file."""
    board = load_scoreboard()
    board[step_name] = row
    with open(SCOREBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(board, f, indent=1)
    return board


# ------------------------------------------------------------------------------------------------ test a model on the golden set
def evaluate(tok, model, system, shots=None, max_new_tokens=60, compact=False):
    """Run the whole golden set through the model with one prompt and score it. Returns a dictionary of results."""
    golden = edge_nlu.load_golden()
    prompt_texts = []
    for case in golden:
        prompt_texts.append(make_prompt_text(tok, system, case["text"], shots))
    result = ask_many(tok, model, prompt_texts, max_new_tokens=max_new_tokens)
    replies = result["replies"]
    if compact:
        converted = []
        for reply in replies:
            converted.append(edge_nlu.compact_to_json(reply))
        replies = converted
    scores = edge_nlu.score_all(replies, golden)
    return {"exact %": scores["exact_ok"], "intent %": scores["intent_ok"], "valid JSON %": scores["valid_json"],
            "prompt tokens": round(mean(result["prompt_tokens"])), "answer tokens": round(mean(result["answer_tokens"]), 1),
            "ms per message": round(1000 * result["seconds"] / len(golden)), "wrong": scores["wrong"], "replies": replies,
            "by_kind": edge_nlu.accuracy_by_kind(replies, golden)}


def evaluate_cases(tok, model, system, cases, shots=None, max_new_tokens=60):
    """Run a list of test cases (for example your own messages) through the model and score them."""
    if len(cases) == 0:
        return {"exact %": 0.0, "replies": [], "wrong": []}
    prompt_texts = []
    for case in cases:
        prompt_texts.append(make_prompt_text(tok, system, case["text"], shots))
    result = ask_many(tok, model, prompt_texts, max_new_tokens=max_new_tokens)
    scores = edge_nlu.score_all(result["replies"], cases)
    return {"exact %": scores["exact_ok"], "replies": result["replies"], "wrong": scores["wrong"]}
