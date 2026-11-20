"""Day 3 helper for Lab 18: make a GGUF file, shrink it, and run it on the CPU only.

GGUF is the file format of llama.cpp, the engine that runs small models on laptops, phones and devices without a GPU.
Plain Python on purpose; the notebook shows each step.
"""
import contextlib
import ctypes
import io
import math
import os
import subprocess
import sys
import time

import numpy as np

import edge_nlu

LLAMA_CPP_FOLDER = "/workspace/llama.cpp"           # where the llama.cpp project is cloned (it holds the converter script)


def convert_to_gguf(model_folder, out_file):
    """Convert a Hugging Face model folder into one 16-bit .gguf file. Returns the size in MB."""
    script = os.path.join(LLAMA_CPP_FOLDER, "convert_hf_to_gguf.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(LLAMA_CPP_FOLDER, "gguf-py")
    command = [sys.executable, script, model_folder, "--outfile", out_file, "--outtype", "f16"]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        raise RuntimeError("the conversion failed")
    return round(os.path.getsize(out_file) / 1e6)


def quantise(in_file, out_file, level):
    """Shrink a .gguf file. level is a name such as Q8_0, Q5_K_M, Q4_K_M or Q2_K. Returns the new size in MB."""
    import llama_cpp
    settings = llama_cpp.llama_model_quantize_default_params()
    settings.ftype = getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_" + level)
    saved_out = os.dup(1)                    # the C library prints hundreds of lines: send them to the bin while it works
    saved_err = os.dup(2)
    nowhere = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nowhere, 1)
    os.dup2(nowhere, 2)
    quiet_out = io.StringIO()                # in a notebook the messages go through Python, so catch them there as well
    with contextlib.redirect_stdout(quiet_out), contextlib.redirect_stderr(quiet_out):
        status = llama_cpp.llama_model_quantize(in_file.encode(), out_file.encode(), ctypes.byref(settings))
    os.dup2(saved_out, 1)
    os.dup2(saved_err, 2)
    for descriptor in (nowhere, saved_out, saved_err):
        os.close(descriptor)
    if status != 0:
        raise RuntimeError("quantising to " + level + " failed")
    return round(os.path.getsize(out_file) / 1e6)


def warm_up(path):
    """Read the whole file once, so the first message does not wait for the disk. Returns the seconds it took.

    A .gguf file is loaded from disk lazily: without this, the first message or two of every new file can take
    10 seconds or more. A device would have the file in memory already.
    """
    started = time.time()
    with open(path, "rb") as f:
        while f.read(64 * 1024 * 1024):
            pass
    return round(time.time() - started, 1)


def load_gguf(path, threads=4, with_scores=False):
    """Open a .gguf file on the CPU only (no GPU layers).

    threads=4: a small device has a few cores, and on a shared pod more threads make the answers stall.
    with_scores=True keeps the scores of every token (needed only for perplexity; it uses more memory).
    """
    from llama_cpp import Llama
    context = 512 if with_scores else 1024
    return Llama(model_path=path, n_ctx=context, n_threads=threads, n_threads_batch=threads, n_gpu_layers=0,
                 logits_all=with_scores, verbose=False)


def ask_gguf(llm, system, user_text, max_tokens=60):
    """Ask one message. Returns (reply, prompt_tokens, answer_tokens, seconds). Greedy decoding (temperature 0)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": '"' + user_text + '"'}]
    started = time.time()
    output = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens)
    seconds = time.time() - started
    reply = output["choices"][0]["message"]["content"].strip()
    usage = output["usage"]
    return reply, usage["prompt_tokens"], usage["completion_tokens"], seconds


def evaluate_gguf(llm, system, cases, max_seconds=90):
    """Run the cases one by one on the CPU (a device answers one person at a time) and score them.

    max_seconds is a time limit: if the cases take longer, the loop stops and only the finished cases are scored.
    """
    replies = []
    done_cases = []
    answer_tokens = 0
    total_seconds = 0.0
    ask_gguf(llm, system, cases[0]["text"])        # one practice message, not counted: the first message of a new file is always slow
    for case in cases:
        if total_seconds > max_seconds:
            break
        reply, prompt_count, answer_count, seconds = ask_gguf(llm, system, case["text"])
        replies.append(reply)
        done_cases.append(case)
        answer_tokens = answer_tokens + answer_count
        total_seconds = total_seconds + seconds
    scores = edge_nlu.score_all(replies, done_cases)
    return {"exact %": scores["exact_ok"], "intent %": scores["intent_ok"], "valid JSON %": scores["valid_json"],
            "ms per message": round(1000 * total_seconds / len(done_cases)), "tokens per second": round(answer_tokens / total_seconds, 1),
            "messages done": len(done_cases), "wrong": scores["wrong"]}


def answer_perplexity(llm, system, examples):
    """Perplexity of the correct answers: how surprised the model is by them (lower means less surprised).

    Perplexity is 2.718 raised to the average of the negative log-probability of each answer token.
    """
    total_log_prob = 0.0
    token_count = 0
    for example in examples:
        prompt_messages = [{"role": "system", "content": system}, {"role": "user", "content": '"' + example["text"] + '"'}]
        prompt_ids = llm.tokenize(_chat_text(llm, prompt_messages).encode(), add_bos=False, special=True)
        answer_ids = llm.tokenize(edge_nlu.answer_text(example["intent"], example["slots"]).encode(), add_bos=False)
        llm.reset()
        llm.eval(prompt_ids + answer_ids)
        for position in range(len(answer_ids)):
            row = llm.scores[len(prompt_ids) + position - 1]          # the model's scores for every possible next token
            biggest = float(np.max(row))
            sum_exp = float(np.sum(np.exp(row - biggest)))
            log_prob = float(row[answer_ids[position]]) - biggest - math.log(sum_exp)
            total_log_prob = total_log_prob + log_prob
            token_count = token_count + 1
    return round(math.exp(-total_log_prob / token_count), 3)


def _chat_text(llm, messages):
    """The prompt text in the chat format of the model (Qwen: <|im_start|>role ... <|im_end|>)."""
    text = ""
    for message in messages:
        text = text + "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"
    return text + "<|im_start|>assistant\n"
