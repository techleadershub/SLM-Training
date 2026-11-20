"""Helpers for extra lab X2: several agents in a LangGraph graph, designed for cost and latency.

The notebook builds five graphs for the SAME task (a smart home, Day 2's `home_domain.py`). This file holds what the five graphs share:

  * `MeteredModel`   the small (1.5B) or the big (3B) local model with the same `invoke` as a LangChain chat model, that also
                     counts every call: prompt tokens, answer tokens, seconds, and how sure the model was (token probabilities).
                     It is built on `lc_slm` (message conversion) and `graph_slm` (tool-call reading) from Day 2.
  * the call log     one line per model call, labelled with the design and the request number; `summarise` turns it into a table row.
  * the 20 requests  with the right final state of the home for each one (so accuracy is measured by code, not by eye).
  * the rules        a small router that is plain code (no model): split a request, find the device, read room, state, number.
  * the tool runner  `run_tool` / `apply_calls`: the model proposes a call, the code validates it, refuses risky ones, executes it.

Everything is plain Python on purpose (loops, ifs, lists, dictionaries, named functions).
The model is loaded only when a model call is made, so the rules and the request list can be used without a GPU.
"""
import math
import re
import threading
import time

from langchain_core.messages import AIMessage

import graph_slm
import home_domain as hd
import lc_slm

# ------------------------------------------------------------------------------------------------ the two models
SMALL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BIG_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_IDS = {"small": SMALL_ID, "big": BIG_ID}

# The cost unit of this lab. ASSUMPTION FOR TEACHING, not real prices: a bigger model costs more per token,
# and an answer token costs more than a prompt token. Change the numbers and call `summarise` again: no model run is needed.
SIZE_WEIGHT = {"small": 1.0, "big": 2.0}
OUT_WEIGHT = 3.0

USE_LOCK = False                 # set True if parallel calls on one GPU ever raise an error: calls then wait for each other
_LOCK = threading.Lock()
_LOADED = {}


def get_model(size):
    """The (tokenizer, model) pair for "small" or "big". Loaded on first use, then kept."""
    if size not in _LOADED:
        from slm_lab import load
        _LOADED[size] = load(MODEL_IDS[size])
    return _LOADED[size]


def load_models(sizes=None):
    """Load the models now, so that the first design does not pay for the loading."""
    if sizes is None:
        sizes = ["small", "big"]
    for size in sizes:
        get_model(size)


def gpu_memory_gb():
    """Memory held by the models on the GPU, in gigabytes (0.0 without a GPU)."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / 1e9, 1)


# ------------------------------------------------------------------------------------------------ the call log
LOG = []
CURRENT = {"design": "setup", "request": 0}
CACHE = {"on": False, "store": {}}


def set_current(design, request_index):
    """Label the next model calls: which design, which request."""
    CURRENT["design"] = design
    CURRENT["request"] = request_index


def log_call(size, out, n_seqs=1):
    LOG.append({"design": CURRENT["design"], "request": CURRENT["request"], "size": size, "prompt_tokens": out["prompt_tokens"],
                "new_tokens": out["new_tokens"], "seconds": out["seconds"], "confidence": out["confidence"],
                "cached": out["cached"], "n_seqs": n_seqs})


def clear_log(design):
    """Forget the log lines of one design (so that running a design again does not count twice)."""
    kept = []
    for entry in LOG:
        if entry["design"] != design:
            kept.append(entry)
    LOG[:] = kept


def entries_for(design, request_index=None):
    found = []
    for entry in LOG:
        if entry["design"] == design:
            if request_index is None or entry["request"] == request_index:
                found.append(entry)
    return found


def entry_cost(entry):
    """Cost units of one model call: (prompt tokens + OUT_WEIGHT x answer tokens) x size weight / 1000. ASSUMPTION, see above."""
    tokens = entry["prompt_tokens"] + OUT_WEIGHT * entry["new_tokens"]
    return tokens * SIZE_WEIGHT[entry["size"]] / 1000.0


def set_cache(on, clear=True):
    """Switch the answer cache on or off. It remembers the answer for an exactly repeated prompt."""
    CACHE["on"] = on
    if clear:
        CACHE["store"] = {}


# ------------------------------------------------------------------------------------------------ running the model
def _probabilities(tok, new_ids, log_probs):
    """Probability of each generated token, up to the end-of-answer token. Returns (list, number of tokens generated)."""
    probs = []
    count = 0
    for position in range(len(new_ids)):
        count = position + 1
        token = tok.convert_ids_to_tokens(int(new_ids[position]))
        if token == "<|im_end|>" or token == "<|endoftext|>":
            break
        probs.append(math.exp(float(log_probs[position])))
    return probs, count


def _smallest_and_mean(probs):
    if len(probs) == 0:
        return 1.0, 1.0
    total = 0.0
    for value in probs:
        total = total + value
    return min(probs), total / len(probs)


def run_model(size, chat, tools, max_new_tokens):
    """One greedy answer from the model. The only place where a single prompt goes to the GPU."""
    import torch
    tok, model = get_model(size)
    text = tok.apply_chat_template(chat, tools=tools, tokenize=False, add_generation_prompt=True)
    encoded = tok(text, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    seconds = time.perf_counter() - started
    width = encoded["input_ids"].shape[1]
    new_ids = out.sequences[0, width:]
    log_probs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    probs, count = _probabilities(tok, new_ids, log_probs)
    smallest, average = _smallest_and_mean(probs)
    return {"text": tok.decode(new_ids, skip_special_tokens=True).strip(), "prompt_tokens": int(width), "new_tokens": count,
            "seconds": seconds, "confidence": smallest, "mean_prob": average, "cached": False}


def generate(size, chat, tools=None, max_new_tokens=80):
    """One model call, counted in the log. With the cache on, an exactly repeated prompt costs nothing."""
    key = None
    if CACHE["on"]:
        key = repr([size, chat, tools, max_new_tokens])
        if key in CACHE["store"]:
            saved = dict(CACHE["store"][key])
            saved["cached"] = True
            saved["prompt_tokens"] = 0
            saved["new_tokens"] = 0
            saved["seconds"] = 0.0
            log_call(size, saved)
            return saved
    if USE_LOCK:
        with _LOCK:
            out = run_model(size, chat, tools, max_new_tokens)
    else:
        out = run_model(size, chat, tools, max_new_tokens)
    log_call(size, out)
    if key is not None:
        CACHE["store"][key] = dict(out)
    return out


def run_batch_model(size, chats, tools_list, max_new_tokens):
    """Several prompts in ONE generate call (a batch). Returns (list of answers, seconds for the whole batch)."""
    import torch
    tok, model = get_model(size)
    texts = []
    for index in range(len(chats)):
        texts.append(tok.apply_chat_template(chats[index], tools=tools_list[index], tokenize=False, add_generation_prompt=True))
    tok.padding_side = "left"
    encoded = tok(texts, return_tensors="pt", padding=True).to(model.device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    seconds = time.perf_counter() - started
    width = encoded["input_ids"].shape[1]
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    answers = []
    for row in range(len(texts)):
        new_ids = out.sequences[row, width:]
        probs, count = _probabilities(tok, new_ids, scores[row])
        smallest, average = _smallest_and_mean(probs)
        answers.append({"text": tok.decode(new_ids, skip_special_tokens=True).strip(), "prompt_tokens": int(encoded["attention_mask"][row].sum()),
                        "new_tokens": count, "seconds": 0.0, "confidence": smallest, "mean_prob": average, "cached": False})
    return answers, seconds


def generate_batch(size, chats, tools_list, max_new_tokens=60):
    """Many prompts in one batch. Logged as ONE call (with n_seqs prompts inside): the tokens are all counted, the call only once."""
    answers, seconds = run_batch_model(size, chats, tools_list, max_new_tokens)
    prompt_total = 0
    new_total = 0
    for answer in answers:
        prompt_total = prompt_total + answer["prompt_tokens"]
        new_total = new_total + answer["new_tokens"]
    log_call(size, {"prompt_tokens": prompt_total, "new_tokens": new_total, "seconds": seconds, "confidence": 1.0, "cached": False},
             n_seqs=len(answers))
    return answers


def prompt_size(size, chat, tools=None):
    """How many tokens a prompt has (the model is not run)."""
    tok, model = get_model(size)
    text = tok.apply_chat_template(chat, tools=tools, tokenize=False, add_generation_prompt=True)
    return len(tok(text)["input_ids"])


# ------------------------------------------------------------------------------------------------ a chat model with a counter
def calls_from_text(text):
    """Tool calls written by the model, as a list of {"name", "args"}. Qwen writes <tool_call>{json}</tool_call>."""
    calls = []
    for item in graph_slm.parse_tool_calls(text):
        calls.append({"name": item["name"], "args": item["arguments"]})
    return calls


def reply_message(out):
    """The model's answer as a LangChain AIMessage: text, tool_calls, and the confidence in response_metadata."""
    tool_calls = []
    for index, call in enumerate(calls_from_text(out["text"])):
        tool_calls.append({"name": call["name"], "args": call["args"], "id": "call_" + str(index), "type": "tool_call"})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", out["text"], flags=re.S).strip()
    return AIMessage(content=content, tool_calls=tool_calls, response_metadata={"confidence": out["confidence"], "raw": out["text"]})


class MeteredModel:
    """The local model, with `invoke(messages)` like a LangChain chat model, and a counter on every call."""

    def __init__(self, size="small", max_new_tokens=80, tools=None):
        self.size = size
        self.max_new_tokens = max_new_tokens
        self.tools = tools

    def bind_tools(self, tools):
        """A copy of this model that always shows these tool descriptions to the model."""
        return MeteredModel(self.size, self.max_new_tokens, list(tools))

    def invoke(self, messages):
        chat = []
        for message in messages:
            chat.append(lc_slm._to_chat(message))
        return reply_message(generate(self.size, chat, self.tools, self.max_new_tokens))


def calls_of(reply):
    """The tool calls of an AIMessage as a list of {"name", "args"}."""
    calls = []
    for call in reply.tool_calls:
        calls.append({"name": call["name"], "args": call["args"]})
    return calls


# ------------------------------------------------------------------------------------------------ tools, run by code
def tools_for(names):
    """The tool descriptions (from home_domain) with these names."""
    found = []
    for tool in hd.TOOLS:
        if tool["function"]["name"] in names:
            found.append(tool)
    return found


def run_tool(home, name, args):
    """The model proposes, the code decides. Returns (new_home, message, status): done, invalid or needs_confirmation."""
    clean, error = hd.validate_call(name, args)
    if error is not None:
        return home, error, "invalid"
    if hd.risky(name, clean):
        return home, "I need a clear yes from a person before I do that (" + name + "). Nothing was changed.", "needs_confirmation"
    new_home, message = hd.execute(home, name, clean)
    return new_home, message, "done"


def apply_calls(home, calls):
    """Run a list of calls one after the other. Returns (new_home, list of messages)."""
    messages = []
    for call in calls:
        home, message, status = run_tool(home, call["name"], call["args"])
        messages.append(message)
    return home, messages


def make_reply(messages):
    """The text said to the user, written by code from the tool results (no model call)."""
    if len(messages) == 0:
        return "Sorry, I can only help with lights, blinds, air conditioners, scenes and the front door."
    return " ".join(messages)


def calls_valid(calls, text=""):
    """True when every call is allowed by the code, and a <tool_call> that was written could be read."""
    if "<tool_call>" in text and len(calls) == 0:
        return False
    for call in calls:
        clean, error = hd.validate_call(call["name"], call["args"])
        if error is not None:
            return False
    return True


def describe_calls(calls):
    """Calls as short text: set_light(room=kitchen, state=on)."""
    pieces = []
    for call in calls:
        inside = []
        for key in call["args"]:
            inside.append(key + "=" + str(call["args"][key]))
        pieces.append(call["name"] + "(" + ", ".join(inside) + ")")
    return "; ".join(pieces)


def changes(home, base=None):
    """What differs from a fresh home (or from `base`), as a dictionary. This is how we show the state of the home."""
    if base is None:
        base = hd.new_home()
    different = {}
    for key in home:
        if home[key] != base.get(key):
            different[key] = home[key]
    return different


# ------------------------------------------------------------------------------------------------ the 20 requests
def _call(name, args):
    return {"name": name, "args": args}


REQUESTS = [
    {"kind": "one command", "text": "turn on the kitchen light", "calls": [_call("set_light", {"room": "kitchen", "state": "on"})]},
    {"kind": "one command", "text": "set the bedroom ac to 22", "calls": [_call("set_ac", {"room": "bedroom", "temp": 22})]},
    {"kind": "one command", "text": "close the study blinds", "start": {"blinds.study": "open"}, "calls": [_call("set_blinds", {"room": "study", "state": "closed"})]},
    {"kind": "one command", "text": "start movie night", "calls": [_call("run_scene", {"scene": "movie night"})]},
    {"kind": "one command", "text": "switch off the balcony light", "start": {"light.balcony": "on"}, "calls": [_call("set_light", {"room": "balcony", "state": "off"})]},
    {"kind": "one command", "text": "turn off the bedroom ac", "start": {"ac.bedroom": "22 C"}, "calls": [_call("ac_off", {"room": "bedroom"})]},
    {"kind": "several commands", "text": "turn on the living room light and set the living room ac to 24",
     "calls": [_call("set_light", {"room": "living room", "state": "on"}), _call("set_ac", {"room": "living room", "temp": 24})]},
    {"kind": "several commands", "text": "open the bedroom blinds and turn on the bedroom light",
     "calls": [_call("set_blinds", {"room": "bedroom", "state": "open"}), _call("set_light", {"room": "bedroom", "state": "on"})]},
    {"kind": "several commands", "text": "kitchen light on, study light on and balcony light off",
     "start": {"light.balcony": "on"}, "calls": [_call("set_light", {"room": "kitchen", "state": "on"}), _call("set_light", {"room": "study", "state": "on"}),
               _call("set_light", {"room": "balcony", "state": "off"})]},
    {"kind": "several commands", "text": "set the study ac to 21, close the study blinds and turn on the study light",
     "calls": [_call("set_ac", {"room": "study", "temp": 21}), _call("set_blinds", {"room": "study", "state": "closed"}),
               _call("set_light", {"room": "study", "state": "on"})]},
    {"kind": "several commands", "text": "start the good morning scene and set the kitchen ac to 25",
     "calls": [_call("run_scene", {"scene": "good morning"}), _call("set_ac", {"room": "kitchen", "temp": 25})]},
    {"kind": "several commands", "text": "turn off the living room light and open the balcony blinds",
     "start": {"light.living room": "on"}, "calls": [_call("set_light", {"room": "living room", "state": "off"}), _call("set_blinds", {"room": "balcony", "state": "open"})]},
    {"kind": "messy", "text": "kitchen ki light band karo", "start": {"light.kitchen": "on"}, "calls": [_call("set_light", {"room": "kitchen", "state": "off"})]},
    {"kind": "messy", "text": "bedroom ka ac 23 par kar do", "calls": [_call("set_ac", {"room": "bedroom", "temp": 23})]},
    {"kind": "messy", "text": "tern of the livin room lite", "start": {"light.living room": "on"}, "calls": [_call("set_light", {"room": "living room", "state": "off"})]},
    {"kind": "messy", "text": "bedroom ke parde khol do aur study ki light jala do",
     "calls": [_call("set_blinds", {"room": "bedroom", "state": "open"}), _call("set_light", {"room": "study", "state": "on"})]},
    {"kind": "not for this assistant", "text": "tell me a joke", "calls": []},
    {"kind": "not for this assistant", "text": "change the channel to the news", "calls": []},
    {"kind": "needs a yes", "text": "unlock the front door for the delivery man", "calls": [_call("unlock_door", {})]},
    {"kind": "needs a yes", "text": "I am leaving now so run away mode", "start": {"light.kitchen": "on"}, "calls": [_call("run_scene", {"scene": "away"})]},
]


def start_home(index):
    """The home at the start of request number `index`: a fresh home, with the devices that the request needs to change already set."""
    home = hd.new_home()
    start = REQUESTS[index].get("start", {})
    for key in start:
        home[key] = start[key]
    return home


def expected_home(index):
    """The right final state of the home for request number `index` (0 to 19). Risky calls are refused, so they change nothing."""
    home, messages = apply_calls(start_home(index), REQUESTS[index]["calls"])
    return home


def expected_text(index):
    request = REQUESTS[index]
    if request["kind"] == "not for this assistant":
        return "nothing changes (not a job of this assistant)"
    if request["kind"] == "needs a yes":
        return "nothing changes (a person must say yes first)"
    return describe_calls(request["calls"])


def is_right(index, home):
    """Accuracy rule: the home must end in EXACTLY the right state."""
    return home == expected_home(index)


def state_after(index, calls):
    """The home after running these calls at the start of request number `index`."""
    home, messages = apply_calls(start_home(index), calls)
    return home


# ------------------------------------------------------------------------------------------------ prompts shared by the designs
DOMAINS = {"lights": {"tools": ["set_light", "set_blinds"], "about": "lights and blinds"},
           "climate": {"tools": ["set_ac", "ac_off"], "about": "air conditioners"},
           "house": {"tools": ["run_scene", "unlock_door"], "about": "scenes and the front door"}}

ALL_TOOLS_SYSTEM = ("You control a smart home. Call a tool for every command in the request, in the order given. Use only values the user said. "
                    "If the request is not about lights, blinds, air conditioners, scenes or the front door, do not call a tool: "
                    "answer in one short sentence.")

PLAN_SYSTEM = """Split the request of a smart-home user into single commands, one per line, written as: domain|command
domain is lights (lights and blinds), climate (air conditioners), house (scenes and the front door) or none (anything else).
Write nothing else. Example:
turn on the kitchen light and set the study ac to 22
lights|turn on the kitchen light
climate|set the study ac to 22"""


def specialist_system(domain):
    """The system message of one specialist: it handles only its own part of the request."""
    return ("You are the " + domain + " specialist of a smart-home assistant. The request may contain other parts: handle ONLY the part about "
            + DOMAINS[domain]["about"] + " and ignore the rest. Call the tool for each command of your part. If there is nothing for you, reply none.")


def supervisor_system(done):
    """The system message of the supervisor, with the list of specialists that have already finished."""
    text = ("You route requests in a smart-home assistant. Specialists: lights (lights and blinds), climate (air conditioners), "
            "house (scenes and the front door). Choose the next specialist that is still needed for the request. "
            "Reply with ONE word: lights, climate, house or finish. ")
    if len(done) == 0:
        return text + "No specialist has worked yet."
    return text + "Already finished: " + ", ".join(done) + "."


def pick_word(text, options):
    """The first option that appears in the text (or "finish")."""
    lowered = text.lower()
    best = ""
    best_position = 10000
    for option in options:
        position = lowered.find(option)
        if position >= 0 and position < best_position:
            best = option
            best_position = position
    if best == "":
        return "finish"
    return best


def parse_plan(text):
    """Read the planner's lines  domain|command  into a list of (domain, command)."""
    plan = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if "|" not in line:
            continue
        domain, command = line.split("|", 1)
        domain = domain.strip().lower()
        if (domain in DOMAINS or domain == "none") and command.strip() != "":
            plan.append((domain, command.strip()))
    return plan


def batch_workers(plan):
    """Design 4, variant b: all the workers in ONE batch on the GPU. Returns the list of tool calls."""
    chats = []
    tools_list = []
    for domain, part in plan:
        if domain in DOMAINS:
            chats.append([{"role": "system", "content": specialist_system(domain)}, {"role": "user", "content": part}])
            tools_list.append(tools_for(DOMAINS[domain]["tools"]))
    if len(chats) == 0:
        return []
    calls = []
    for answer in generate_batch("small", chats, tools_list, max_new_tokens=60):
        calls = calls + calls_from_text(answer["text"])
    return calls


# ------------------------------------------------------------------------------------------------ the rules router (plain code)
LIGHT_WORDS = ["light", "lights", "lamp", "batti"]
BLIND_WORDS = ["blinds", "blind", "curtain", "curtains", "shade", "shades"]
AC_WORDS = ["ac", "aircon", "temperature"]
DOOR_WORDS = ["door", "unlock"]
SCENE_WORDS = ["scene", "movie", "away", "leaving"]
ON_WORDS = ["on", "jala", "jalao", "chalu"]
OFF_WORDS = ["off", "band", "bujha", "bujhao"]
OPEN_WORDS = ["open", "up"]
CLOSE_WORDS = ["close", "closed", "band", "down"]


def split_request(text):
    """Cut a request into single commands at 'and', 'aur', 'then', 'also' and commas."""
    lowered = " " + text.lower() + " "
    for joiner in [",", ";", " and ", " aur ", " then ", " also "]:
        lowered = lowered.replace(joiner, "|")
    parts = []
    for piece in lowered.split("|"):
        if piece.strip() != "":
            parts.append(piece.strip())
    return parts


def words_of(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def any_in(words, wanted):
    for word in words:
        if word in wanted:
            return True
    return False


def find_room(text):
    for room in hd.ROOMS:
        if room in text.lower():
            return room
    return ""


def find_number(words):
    for word in words:
        if word.isdigit():
            return int(word)
    return None


def find_kind(words):
    """Which device the command is about: door, scene, ac, blinds, light, or "" when no rule matches."""
    if any_in(words, DOOR_WORDS):
        return "door"
    if any_in(words, SCENE_WORDS):
        return "scene"
    if any_in(words, AC_WORDS):
        return "ac"
    if any_in(words, BLIND_WORDS):
        return "blinds"
    if any_in(words, LIGHT_WORDS):
        return "light"
    return ""


def find_scene(words):
    if "away" in words or "leaving" in words:
        return "away"
    if "movie" in words:
        return "movie night"
    if "morning" in words:
        return "good morning"
    return ""


def code_parse(part):
    """Try to read ONE command with rules only. Returns {"kind", "domain", "calls", "complete"}.

    complete is True when the rules found the device AND every value it needs. Otherwise a model has to help.
    """
    words = words_of(part)
    kind = find_kind(words)
    domain = {"light": "lights", "blinds": "lights", "ac": "climate", "scene": "house", "door": "house"}.get(kind, "")
    room = find_room(part)
    calls = []
    if kind == "door":
        calls = [_call("unlock_door", {})]
    elif kind == "scene" and find_scene(words) != "":
        calls = [_call("run_scene", {"scene": find_scene(words)})]
    elif kind == "light" and room != "" and any_in(words, OFF_WORDS):
        calls = [_call("set_light", {"room": room, "state": "off"})]
    elif kind == "light" and room != "" and any_in(words, ON_WORDS):
        calls = [_call("set_light", {"room": room, "state": "on"})]
    elif kind == "blinds" and room != "" and any_in(words, CLOSE_WORDS):
        calls = [_call("set_blinds", {"room": room, "state": "closed"})]
    elif kind == "blinds" and room != "" and any_in(words, OPEN_WORDS):
        calls = [_call("set_blinds", {"room": room, "state": "open"})]
    elif kind == "ac" and room != "" and any_in(words, OFF_WORDS):
        calls = [_call("ac_off", {"room": room})]
    elif kind == "ac" and room != "" and find_number(words) is not None:
        calls = [_call("set_ac", {"room": room, "temp": find_number(words)})]
    return {"kind": kind, "domain": domain, "calls": calls, "complete": len(calls) > 0}


# ------------------------------------------------------------------------------------------------ running a design on the 20 requests
def run_design(name, app, show=True):
    """Run one compiled graph on all 20 requests, each on a fresh home. Returns a list of results (one per request)."""
    clear_log(name)
    results = []
    for index in range(len(REQUESTS)):
        set_current(name, index)
        start_state = {"request": REQUESTS[index]["text"], "home": start_home(index)}
        started = time.perf_counter()
        error = ""
        try:
            out = app.invoke(start_state, config={"recursion_limit": 25})
        except Exception as problem:
            out = start_state
            error = str(problem)[:100]
        seconds = time.perf_counter() - started
        results.append({"index": index, "text": REQUESTS[index]["text"], "kind": REQUESTS[index]["kind"],
                        "ok": is_right(index, out["home"]), "home": out["home"], "reply": out.get("reply", ""), "seconds": seconds,
                        "error": error, "escalated": out.get("escalated", False), "confidence": out.get("confidence", None),
                        "small_calls": out.get("small_calls", None)})
    if show:
        print(name + ": done,", len(results), "requests")
    return results


def mean_of(numbers):
    if len(numbers) == 0:
        return 0.0
    total = 0.0
    for number in numbers:
        total = total + number
    return total / len(numbers)


def summarise(name, results):
    """One row of the scoreboard: per request, how many model calls, tokens, cost units, milliseconds; and accuracy."""
    count = len(results)
    calls = 0
    big_calls = 0
    hits = 0
    tokens = 0
    cost = 0.0
    for entry in entries_for(name):
        if entry["cached"]:
            hits = hits + 1
        else:
            calls = calls + 1
            if entry["size"] == "big":
                big_calls = big_calls + 1
        tokens = tokens + entry["prompt_tokens"] + entry["new_tokens"]
        cost = cost + entry_cost(entry)
    right = 0
    slowest = 0.0
    total_seconds = 0.0
    errors = 0
    for result in results:
        if result["ok"]:
            right = right + 1
        total_seconds = total_seconds + result["seconds"]
        slowest = max(slowest, result["seconds"])
        if result["error"] != "":
            errors = errors + 1
    return {"design": name, "calls per request": round(calls / count, 2), "big-model calls": big_calls, "cache hits": hits,
            "tokens per request": round(tokens / count), "cost units per request": round(cost / count, 3),
            "ms per request": round(1000 * total_seconds / count), "slowest ms": round(1000 * slowest),
            "accuracy %": round(100 * right / count, 1), "errors": errors}


def accuracy_by_kind(results):
    """Accuracy per kind of request, as a dictionary."""
    right = {}
    total = {}
    for result in results:
        kind = result["kind"]
        total[kind] = total.get(kind, 0) + 1
        if result["ok"]:
            right[kind] = right.get(kind, 0) + 1
    scores = {}
    for kind in total:
        scores[kind] = round(100 * right.get(kind, 0) / total[kind])
    return scores


def show_mistakes(results, limit=4):
    """Print the requests a design got wrong: what was expected and what the home looked like."""
    shown = 0
    for result in results:
        if result["ok"] or shown >= limit:
            continue
        print("request", result["index"] + 1, ":", result["text"])
        print("   expected:", expected_text(result["index"]))
        print("   what changed:", changes(result["home"], start_home(result["index"])), result["error"])
        shown = shown + 1
    if shown == 0:
        print("no mistakes in this design")


def ledger(design, request_index):
    """Print every model call made for one request: size, tokens, milliseconds, confidence."""
    for entry in entries_for(design, request_index):
        what = str(round(1000 * entry["seconds"])) + " ms"
        if entry["cached"]:
            what = "cache hit"
        print("  ", entry["size"], "model | prompt", entry["prompt_tokens"], "tokens | answer", entry["new_tokens"], "tokens |", what,
              "| sure:", round(entry["confidence"], 2), "| prompts in this call:", entry["n_seqs"])


def trace(app, request_index, width=110):
    """Run request number `request_index` and print what every node of the graph returned (its calls are logged as design 'trace')."""
    text = REQUESTS[request_index]["text"]
    set_current("trace", 0)
    clear_log("trace")
    for update in app.stream({"request": text, "home": start_home(request_index)}, stream_mode="updates"):
        for node in update:
            print(node, "->", str(update[node])[:width])
