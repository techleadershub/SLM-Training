"""Day 3 helper for Lab 19: start a real serving engine (vLLM), send many requests at the same time, measure the waiting times.

vLLM needs its own copy of PyTorch, so it lives in a separate environment (see the install cell of the lab).
The notebook calls the functions below; you can read them from top to bottom.
"""
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import edge_nlu

VENV_PYTHON = os.environ.get("VLLM_PYTHON", "/root/vllm-venv/bin/python")
PORT = 8011
URL = "http://127.0.0.1:" + str(PORT)
LOG_FILE = "/workspace/vllm_server.log"


def start_server(model_folder, memory_share=0.3):
    """Start the serving engine in the background with our model. memory_share is the part of the GPU it may use."""
    command = [VENV_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", model_folder, "--served-model-name", "intent",
               "--port", str(PORT), "--gpu-memory-utilization", str(memory_share), "--max-model-len", "1024",
               "--dtype", "bfloat16", "--enable-prefix-caching"]
    environment = dict(os.environ)
    environment["PATH"] = os.path.dirname(VENV_PYTHON) + os.pathsep + environment["PATH"]     # vLLM needs its own tools on the path
    log = open(LOG_FILE, "w")
    return subprocess.Popen(command, stdout=log, stderr=log, env=environment)


def wait_until_ready(process, max_seconds=420):
    """Wait until the server answers. Returns the seconds it took, or raises an error if it stopped or took too long."""
    started = time.time()
    while time.time() - started < max_seconds:
        if process.poll() is not None:
            raise RuntimeError("the server stopped; read " + LOG_FILE)
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return round(time.time() - started)
        except OSError:
            time.sleep(3)
    raise RuntimeError("the server did not start in time; read " + LOG_FILE)


def stop_server(process):
    """Stop the server and free the GPU memory."""
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def ask_server(system, user_text, max_tokens=60):
    """Send one message to the server. Returns (reply, seconds)."""
    body = {"model": "intent", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": '"' + user_text + '"'}]}
    request = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    seconds = time.time() - started
    return answer["choices"][0]["message"]["content"].strip(), seconds


def _send(job):
    """One request of the load test. job is (system, text)."""
    return ask_server(job[0], job[1])


def load_test(clients, system, cases, rounds=4):
    """Many people talk at the same time: `clients` requests are always in flight, until clients x rounds are answered.

    Returns a dictionary: clients, requests, p50_ms and p95_ms (typical and slow waiting time), per_second (answers per
    second for the whole server) and exact_percent (accuracy, to check that speed did not change the answers).
    """
    total = clients * rounds
    if total < len(cases):
        total = len(cases)                   # every level answers all the test messages, so the accuracy numbers can be compared
    jobs = []
    chosen_cases = []
    for number in range(total):
        case = cases[number % len(cases)]
        jobs.append((system, case["text"]))
        chosen_cases.append(case)
    started = time.time()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        results = list(pool.map(_send, jobs))
    wall_seconds = time.time() - started
    waits = []
    replies = []
    for reply, seconds in results:
        waits.append(seconds)
        replies.append(reply)
    waits.sort()
    scores = edge_nlu.score_all(replies, chosen_cases)
    return {"clients": clients, "requests": total, "p50_ms": round(1000 * waits[int(0.5 * (total - 1))]),
            "p95_ms": round(1000 * waits[int(0.95 * (total - 1))]), "per_second": round(total / wall_seconds, 1),
            "exact_percent": scores["exact_ok"]}
