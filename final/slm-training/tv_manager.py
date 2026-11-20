"""Extra lab X3: a complete dialogue manager for a smart TV, built as a LangGraph graph.

The small language model only READS: in the node `understand` it turns a message into an intent and slots.
Everything else is code: tracking the task, deciding what happens next (`route`), asking for what is missing, offering choices,
asking for confirmation or a PIN, changing the TV, undoing, repairing after failures.
The same graph runs with a keyword-rules reader instead of the model, so the logic can be tested on a laptop with no GPU.

How to use it (this is the interface other labs import):

    import tv_manager as tm
    manager = tm.make_manager(use_model=False)        # rules only, no GPU       (use_model=True loads the small model)
    result = tm.turn(manager, "turn the volume up")   # {'reply', 'state', 'path', 'intent', 'confidence', ...}
    tm.reset(manager)                                 # a new conversation and a new TV
"""
import copy
import re
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

import tv_domain as td
import tv_nlu as tn

TASK_INTENTS = ["power", "set_volume", "volume_up", "volume_down", "mute", "set_channel", "launch_app", "set_input", "playback",
                "search_content", "same_again", "subtitles", "sleep_timer", "settings", "factory_reset", "delete_recordings",
                "rent_movie", "adult_content", "play_title"]
INFO_INTENTS = ["ask_time", "ask_status", "help"]
MULTI_OK = TASK_INTENTS + ["ask_time", "ask_status"]
CORRECTABLE = {"set_volume": "level", "volume_up": "level", "volume_down": "level", "set_channel": "channel", "sleep_timer": "minutes"}
STOP_WORDS = ["the", "one", "channel", "please", "play", "go", "to", "on", "put", "watch", "open", "i", "want", "choose", "number",
              "a", "an", "of", "it", "switch", "show", "me"]

MENU = "I can help with volume, channels, apps, inputs, playback, subtitles and the sleep timer. Which one do you need?"
HANDOVER = "I am sorry, I still cannot understand. I will connect you to a support agent. You can also say help."
HELP_TEXT = {
    "": "I can change the volume, the channel, the app and the input, pause and play, turn subtitles on or off, and set a sleep timer. What would you like?",
    "volume": "For the volume say: volume 30, turn it up, turn it down, or mute.",
    "channel": "For channels say: channel 7, the name of a channel such as Aaj Tak, or a kind such as sports.",
    "app": "To open an app say its name: Netflix, YouTube, Prime Video or Hotstar.",
    "input": "To change the input say: HDMI 1, HDMI 2 or AV.",
    "playback": "For playback say: pause, play or next.",
    "subtitles": "Say: subtitles on, or subtitles off.",
    "sleep": "Say: sleep timer 30 minutes, or cancel the sleep timer.",
}


class TVState(TypedDict, total=False):
    utterance: str
    nlu: dict            # what THIS clause means: intent and slots
    queue: list          # clauses still to handle (more than one command in a sentence)
    task: dict           # the task in progress: {"intent", "slots"} (kept between turns)
    tv: dict             # the device state
    awaiting: str        # what we wait for: "", "slot", "confirm", "pin", "choice"
    pending: dict        # a risky task that waits for a yes or a PIN
    choices: dict        # the options we offered: {"kind", "options", "title"}
    candidates: dict     # new options found in this turn
    picked: dict         # the option the user chose in this turn
    ready: bool          # True when a task is released and must run now
    problem: str         # a sentence about a value we did not accept
    failures: int        # messages in a row that were not understood
    pin_tries: int
    undo: list           # snapshots of the TV, newest last
    last: dict           # the last task that ran (for "no, 45")
    search: dict         # the last search: {"query"}
    handover: bool
    reply: str
    path: list           # the nodes that ran in THIS turn


ACTIVE = {"reader": None, "reads": []}      # the reader of the manager that is running now (set by turn)


# ------------------------------------------------------------------------------------------------- understanding
CONNECTORS = r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\baur\b|\bphir\b|\bfir\b)\s*"


def split_clauses(text):
    """'mute it and go to HDMI 2' becomes ['mute it', 'go to HDMI 2']. A sentence with no connector stays whole."""
    parts = re.split(CONNECTORS, text.strip(), flags=re.IGNORECASE)
    clauses = []
    for part in parts:
        if part.strip() != "":
            clauses.append(part.strip())
    if len(clauses) == 0:
        return [text.strip()]
    return clauses


def read_clause(text):
    """Ask the reader (model or rules) about one clause. The text is logged so a lab can show what the model was given."""
    heard = tn.normalise_text(text)
    ACTIVE["reads"].append(heard)
    answer = tn.read_text(ACTIVE["reader"], heard)
    answer["text"] = heard
    return answer


def understand(state):
    """The ONLY node that uses the language model. A PIN is read by code and never reaches the model."""
    text = state["utterance"]
    out = {"reply": "", "path": ["understand"], "handover": False, "picked": {}, "candidates": {}, "ready": False, "problem": ""}
    if state.get("awaiting") == "pin":
        if tn.clean_text(text) in ["cancel", "never mind", "stop", "forget it", "leave it"]:
            out["queue"] = [{"intent": "cancel", "slots": {}, "text": text, "confidence": 1.0, "raw": "(code)"}]
            return out
        digits = tn.parse_pin(text)
        if len(digits) >= 3:
            out["queue"] = [{"intent": "pin_digits", "slots": {"pin": digits}, "text": "(a PIN)", "confidence": 1.0, "raw": "(not read by the model)"}]
            return out
    clauses = split_clauses(text)
    readings = []
    if len(clauses) > 1:
        all_commands = True
        for clause in clauses:
            reading = read_clause(clause)
            readings.append(reading)
            if reading["intent"] not in MULTI_OK:
                all_commands = False
        if not all_commands:
            readings = [read_clause(text)]
    else:
        readings = [read_clause(text)]
    leftover = state.get("queue") or []
    out["queue"] = readings + leftover
    return out


# ------------------------------------------------------------------------------------------------- tracking
def missing_slots(task):
    return td.required_slots(task["intent"], task["slots"])


def resolve_number(state, number):
    """A bare number ("30", "no, 45"): which question does it answer, or which command does it correct? Returns (intent, slots) or None."""
    task = state.get("task")
    if state.get("awaiting") == "choice":
        return "choose", {"index": number}
    if task:
        missing = missing_slots(task)
        if len(missing) > 0 and missing[0] in ["level", "minutes", "channel"]:
            return task["intent"], {missing[0]: number}
    last = state.get("last")
    if last and last["intent"] in CORRECTABLE and state.get("awaiting", "") == "":
        intent = last["intent"]
        if intent in ["volume_up", "volume_down"]:
            intent = "set_volume"
        return intent, {CORRECTABLE[last["intent"]]: number}
    return None


def resolve_title(state, title):
    """A bare title ("mumbai nights"): it answers the question 'which movie?' or 'what would you like to watch?'. Returns (intent, slots) or None."""
    task = state.get("task")
    if task and state.get("awaiting") == "slot":
        missing = missing_slots(task)
        if len(missing) > 0 and missing[0] == "title":
            return task["intent"], {"title": title}
        if len(missing) > 0 and missing[0] == "query":
            return task["intent"], {"query": title}
    return None


def option_words(text):
    """The meaningful words of a text: lower case, without little words."""
    found = []
    for word in td.words_of(text):
        if word not in STOP_WORDS:
            found.append(word)
    return found


def match_choice(choices, nlu):
    """Which of the offered options did the user choose, by number or by name? Returns (option or None, problem)."""
    options = choices["options"]
    if nlu["intent"] == "choose":
        index = nlu["slots"].get("index", 0)
        if 1 <= index <= len(options):
            return options[index - 1], ""
        return None, "Please choose a number from 1 to " + str(len(options)) + "."
    pool = nlu.get("text", "")
    for name in nlu["slots"]:
        pool = pool + " " + str(nlu["slots"][name])
    query = option_words(pool)
    if len(query) == 0:
        return None, ""
    found = []
    for option in options:
        label_words = td.words_of(option["label"])
        every = True
        for word in query:
            if word not in label_words:
                every = False
        if every:
            found.append(option)
    if len(found) == 1:
        return found[0], ""
    return None, ""


def make_options(kind, items, title=""):
    """A choices dictionary: options are {'label', 'value'}, the user answers by number or by name."""
    options = []
    for label, value in items:
        options.append({"label": label, "value": value})
    return {"kind": kind, "options": options, "title": title}


def check_slots(task, tv, out):
    """The model proposes, the code decides: check every value. Invalid values are dropped and become missing slots."""
    slots = task["slots"]
    for name in ["level", "minutes", "app", "input", "amount"]:
        if name in slots:
            value, problem = td.check_value(name, slots[name])
            if problem != "":
                del slots[name]
                out["problem"] = problem
            else:
                slots[name] = value
    if "channel" in slots:
        status, data = td.find_channel(slots["channel"])
        if status == "ok":
            slots["channel"] = data
        elif status == "many":
            del slots["channel"]
            items = []
            for number in data:
                items.append((td.CHANNELS[number], number))
            out["candidates"] = make_options("channel", items)
        else:
            del slots["channel"]
            if status == "range":
                out["problem"] = "I do not have that channel number."
            else:
                out["problem"] = "I could not find that channel."


def check_search(task, out):
    """Turn a search into a plan: nothing found, one title on one app, a choice of titles, or a choice of apps."""
    slots = task["slots"]
    results = td.search_catalog(slots["query"], slots.get("app", ""))
    out["search"] = {"query": slots["query"]}
    if len(results) == 0:
        out["problem"] = "I could not find anything for " + slots["query"] + "."
        out["task"] = None
    elif len(results) == 1:
        plan_title(results[0], slots.get("app", ""), out)
    else:
        items = []
        for item in results[:3]:
            items.append((td.pretty_title(item["title"]), item["title"]))
        out["candidates"] = make_options("title", items, slots.get("app", ""))
        out["task"] = {"intent": "search_content", "slots": {}}


def plan_title(item, app, out):
    """One title is chosen. If it is on one app (or the app is known) play it, else offer the apps."""
    apps = item["apps"]
    if app != "":
        apps = [app]
    if len(apps) == 1:
        out["task"] = {"intent": "play_title", "slots": {"title": item["title"], "app": apps[0]}}
        return
    items = []
    for name in apps:
        items.append((td.APP_NAMES[name], name))
    out["candidates"] = make_options("app", items, item["title"])
    out["task"] = {"intent": "search_content", "slots": {}}


def merge_task(state, nlu, out):
    """Start a task, or add the new details to the task in progress (same intent), or replace it (a different intent)."""
    task = state.get("task")
    intent = nlu["intent"]
    slots = dict(nlu["slots"])
    if intent == "same_again":
        search = state.get("search")
        if search:
            slots = {"query": search["query"], "app": slots.get("app", "")}
            if slots["app"] == "":
                del slots["app"]
            intent = "search_content"
        else:
            out["problem"] = "There is nothing to repeat yet."
            intent = "search_content"
            slots = {}
            if nlu["slots"].get("app"):
                slots = {"app": nlu["slots"]["app"]}
    if task and task["intent"] == intent and intent != "play_title":
        merged = dict(task["slots"])
        for name in slots:
            merged[name] = slots[name]
        return {"intent": intent, "slots": merged}
    return {"intent": intent, "slots": slots}


def infer_power_state(nlu_text):
    """Backstop in code: the model left the power state empty, so look for the plain words (on, off, chalu, band)."""
    words = td.words_of(nlu_text)
    for word in ["off", "band", "shut", "close"]:
        if word in words:
            return "off"
    for word in ["on", "chalu", "chalao", "start"]:
        if word in words:
            return "on"
    return ""


def add_defaults(task):
    """What the user did not say but the words imply: 'mute' means mute on, 'subtitles' alone means on."""
    if task["intent"] in ["mute", "subtitles"] and "state" not in task["slots"]:
        task["slots"]["state"] = "on"
    if task["intent"] == "rent_movie" and "mode" not in task["slots"]:
        task["slots"]["mode"] = "rent"


def track(state):
    """Update what we know: apply a number to the right question, merge the task, check every value, match a choice."""
    queue = list(state["queue"])
    nlu = dict(queue[0])
    queue = queue[1:]
    tv = state["tv"]
    out = {"queue": queue, "path": state["path"] + ["track"], "problem": "", "picked": {}, "candidates": {}}
    if nlu["intent"] != "other":
        out["failures"] = 0
    if nlu["intent"] == "other" and "number" in nlu["slots"]:
        resolved = resolve_number(state, nlu["slots"]["number"])
        if resolved is not None:
            nlu["intent"], nlu["slots"] = resolved
    if nlu["intent"] == "other" and "title" in nlu["slots"]:
        resolved = resolve_title(state, nlu["slots"]["title"])
        if resolved is not None:
            nlu["intent"], nlu["slots"] = resolved
    if state.get("awaiting") == "choice" and state.get("choices"):
        option, problem = match_choice(state["choices"], nlu)
        if option is not None:
            out["picked"] = {"kind": state["choices"]["kind"], "value": option["value"], "label": option["label"]}
        out["problem"] = problem
    out["nlu"] = nlu
    if nlu["intent"] == "choose" and state.get("awaiting") != "choice":
        out["problem"] = "There is nothing to choose from."
    if nlu["intent"] not in TASK_INTENTS or state.get("awaiting") in ["confirm", "pin"]:
        return out
    if nlu["intent"] == "adult_content" and tv["pin_locked"]:
        out["problem"] = "Adult content is locked after wrong PINs. Please ask the account owner."
        out["task"] = None
        return out
    task = merge_task(state, nlu, out)
    add_defaults(task)
    if task["intent"] == "power" and "state" not in task["slots"]:
        found = infer_power_state(nlu.get("text", ""))
        if found != "":
            task["slots"]["state"] = found
    out["task"] = task
    check_slots(task, tv, out)
    if task["intent"] == "search_content" and "query" in task["slots"]:
        check_search(task, out)
    check_risky(out["task"], tv, out)
    return out


def check_risky(task, tv, out):
    """For a risky task that is complete: check the values now, so we never ask for a yes to something impossible."""
    if task is None or task["intent"] not in td.TIER_CONFIRM:
        return
    if len(missing_slots(task)) > 0 or out.get("problem"):
        return
    args, error = td.check_task(task["intent"], task["slots"], tv)
    if error != "":
        out["problem"] = error
        out["task"] = None
    else:
        task["slots"] = args


# ------------------------------------------------------------------------------------------------- the decision
def route(state):
    """The decision at the heart of the manager. The ORDER of these questions is the behaviour."""
    nlu = state["nlu"]
    intent = nlu["intent"]
    awaiting = state.get("awaiting", "")
    task = state.get("task")
    if intent == "pin_digits":
        return "check_pin"
    if awaiting == "pin":
        if intent == "cancel":
            return "cancel"
        return reply_or_remind(intent, "ask_pin")
    if awaiting == "confirm":
        if intent == "yes":
            return "release"
        if intent in ["no", "cancel"]:
            return "cancel"
        return reply_or_remind(intent, "confirm")
    if awaiting == "choice":
        if state.get("picked"):
            return "pick_choice"
        if intent == "cancel":
            return "cancel"
        if intent in INFO_INTENTS:
            return "answer_info"
        if intent in ["yes", "no", "other", "choose"] or state.get("problem"):
            return "clarify"
    if intent == "cancel":
        if task or awaiting != "":
            return "cancel"
        return "nothing_pending"
    if intent in ["yes", "no"]:
        return "nothing_pending"
    return route_command(state, intent, task)


def reply_or_remind(intent, waiting_node):
    """While we wait for a yes or a PIN: answer a question, and ask again; anything else just gets asked again."""
    if intent in INFO_INTENTS:
        return "answer_info"
    return waiting_node


def route_command(state, intent, task):
    """A new command (or a question). Questions about the TV first, then the command itself."""
    if intent == "undo":
        return "do_undo"
    if intent in INFO_INTENTS:
        return "answer_info"
    if intent == "out_of_scope":
        return "refuse"
    if intent == "other":
        return "fallback"
    if state["tv"]["power"] == "off" and intent != "power":
        return "tv_off"
    if state.get("problem") and (not task or intent == "choose"):
        return "explain"
    if state.get("candidates"):
        return "clarify"
    if intent == "choose":
        return "explain"
    if missing_slots(task):
        return "ask_slot"
    if state.get("problem"):
        return "explain"
    kind = td.tier(task["intent"], state["tv"])
    if kind == "confirm":
        return "confirm"
    if kind == "pin":
        return "ask_pin"
    return "execute"


def next_step(state):
    """After a node has spoken: run a released task, handle the next clause of the sentence, or stop."""
    if state.get("ready"):
        return "execute"
    if len(state.get("queue") or []) > 0 and state.get("awaiting", "") == "":
        return "track"
    return "end"


# ------------------------------------------------------------------------------------------------- the nodes that speak
def say(state, text):
    """Add a sentence to this turn's reply (a turn can have several: one per command)."""
    old = state.get("reply", "")
    if old == "":
        return text
    return old + " " + text


def list_text(choices):
    """The options as one sentence: '1, Star Sports 1. 2, Sony Sports.'"""
    parts = []
    for index in range(len(choices["options"])):
        parts.append(str(index + 1) + ", " + choices["options"][index]["label"] + ".")
    return " ".join(parts)


def reminder(state):
    """What we are waiting for, as a short sentence to add after an answer. Empty when we wait for nothing."""
    awaiting = state.get("awaiting", "")
    if awaiting == "slot" and state.get("task") and len(missing_slots(state["task"])) > 0:
        return "Back to your request: " + td.QUESTIONS[missing_slots(state["task"])[0]]
    if awaiting == "confirm" and state.get("pending"):
        return td.confirm_text(state["pending"]["intent"], state["pending"]["slots"], state["tv"])
    if awaiting == "pin":
        return "Please say your 4-digit PIN."
    if awaiting == "choice" and state.get("choices"):
        return "Back to your choice. " + list_text(state["choices"]) + " Say the number or the name."
    return ""


def join_text(first, second):
    """Two sentences as one text; either may be empty."""
    if second == "":
        return first
    if first == "":
        return second
    return first + " " + second


def ask_slot(state):
    task = state["task"]
    slot = missing_slots(task)[0]
    question = td.QUESTIONS[slot]
    if slot == "state":
        question = "Do you want it on or off?"
    return {"reply": say(state, join_text(state.get("problem", ""), question)), "awaiting": "slot", "path": state["path"] + ["ask_slot"]}


def explain(state):
    """A value we could not accept and nothing else to ask: say why, and forget the task."""
    return {"reply": say(state, state["problem"]), "task": None, "awaiting": "", "path": state["path"] + ["explain"]}


def clarify(state):
    """Offer the options and wait for the user to choose, by number or by name."""
    choices = state.get("candidates")
    if not choices:
        choices = state["choices"]
    intro = {"channel": "More than one channel fits.", "title": "I found these.", "app": "It is on more than one app."}[choices["kind"]]
    text = join_text(intro, list_text(choices)) + " Which one? Say the number or the name."
    return {"reply": say(state, join_text(state.get("problem", ""), text)), "choices": choices, "awaiting": "choice",
            "path": state["path"] + ["clarify"]}


def pick_choice(state):
    """The user chose an option. Make it a task, or offer the next level of choice (title, then app)."""
    picked = state["picked"]
    choices = state["choices"]
    out = {"choices": {}, "awaiting": "", "path": state["path"] + ["pick_choice"]}
    if picked["kind"] == "channel":
        out["task"] = {"intent": "set_channel", "slots": {"channel": picked["value"]}}
        out["ready"] = True
    elif picked["kind"] == "app":
        out["task"] = {"intent": "play_title", "slots": {"title": choices["title"], "app": picked["value"]}}
        out["ready"] = True
    else:
        item = td.search_catalog(picked["value"], choices["title"])[0]
        plan = {}
        plan_title(item, choices["title"], plan)
        if "candidates" in plan:
            out["choices"] = plan["candidates"]
            out["awaiting"] = "choice"
            out["task"] = plan["task"]
            text = "It is on more than one app. " + list_text(plan["candidates"]) + " Which one? Say the number or the name."
            out["reply"] = say(state, text)
            return out
        out["task"] = plan["task"]
        out["ready"] = True
    return out


def confirm(state):
    """Ask yes or no before a risky action. The task waits in `pending`, not in `task`."""
    task = state.get("pending")
    again = state.get("awaiting") == "confirm"
    if not again:
        task = state["task"]
    text = td.confirm_text(task["intent"], task["slots"], state["tv"])
    if again:
        text = join_text("Sorry, I need a yes or a no.", text)
    return {"reply": say(state, text), "pending": task, "task": None, "awaiting": "confirm", "path": state["path"] + ["confirm"]}


def ask_pin(state):
    """A permission tier: adult content needs a PIN. The PIN is read by code (see `understand`), never by the model."""
    again = state.get("awaiting") == "pin"
    task = state.get("pending")
    tries = state.get("pin_tries", 0)
    text = "Please say your 4-digit PIN, or say cancel."
    if not again:
        task = state["task"]
        tries = 0
        text = "This needs your 4-digit PIN. Please say it now, or say cancel."
    return {"reply": say(state, text), "pending": task, "task": None, "awaiting": "pin", "pin_tries": tries,
            "path": state["path"] + ["ask_pin"]}


def check_pin(state):
    """Compare the spoken PIN with the real one. Three wrong tries lock adult content."""
    tries = state.get("pin_tries", 0) + 1
    tv = copy.deepcopy(state["tv"])
    if state["nlu"]["slots"]["pin"] == td.PIN:
        tv["adult_unlocked"] = True
        return {"tv": tv, "reply": say(state, "PIN accepted."), "task": state["pending"], "pending": None, "awaiting": "", "pin_tries": 0,
                "ready": True, "path": state["path"] + ["check_pin"]}
    left = td.PIN_TRIES - tries
    if left > 0:
        text = "That PIN is not right. You have " + str(left) + " tries left."
        if left == 1:
            text = "That PIN is not right. You have 1 try left."
        return {"reply": say(state, text), "pin_tries": tries, "path": state["path"] + ["check_pin"]}
    tv["pin_locked"] = True
    text = "That PIN is not right. Too many wrong tries, so adult content is locked. Please ask the account owner."
    return {"tv": tv, "reply": say(state, text), "pending": None, "task": None, "awaiting": "", "pin_tries": 0, "queue": [],
            "path": state["path"] + ["check_pin"]}


def release(state):
    """The user said yes: the waiting task is released and will run in the next step."""
    return {"task": state["pending"], "pending": None, "awaiting": "", "ready": True, "path": state["path"] + ["release"]}


def execute(state):
    """The only node that changes the TV. It checks the values once more, remembers how to undo, and says what happened."""
    task = state["task"]
    tv = state["tv"]
    args, error = td.check_task(task["intent"], task["slots"], tv)
    if error != "":
        return {"reply": say(state, error), "task": None, "awaiting": "", "ready": False, "path": state["path"] + ["execute"]}
    new_tv, message = td.execute(tv, task["intent"], args)
    undo = list(state.get("undo") or [])
    if task["intent"] in td.IRREVERSIBLE:
        undo.append({"tv": None, "label": task["intent"]})
    elif new_tv != tv:
        undo.append({"tv": copy.deepcopy(tv), "label": task["intent"]})
    undo = undo[-5:]
    out = {"tv": new_tv, "reply": say(state, message), "task": None, "pending": None, "awaiting": "", "ready": False, "failures": 0,
           "undo": undo, "path": state["path"] + ["execute"]}
    if task["intent"] in CORRECTABLE:
        out["last"] = {"intent": task["intent"], "slots": args}
    return out


def do_undo(state):
    """Undo the last change by putting the saved copy of the TV back. A purchase or a reset cannot be undone."""
    undo = list(state.get("undo") or [])
    if len(undo) == 0:
        return {"reply": say(state, "There is nothing to undo."), "path": state["path"] + ["do_undo"]}
    top = undo[-1]
    if top["tv"] is None:
        text = "I cannot undo that. A reset, a purchase or a deletion is final."
        return {"reply": say(state, text), "path": state["path"] + ["do_undo"]}
    undo = undo[:-1]
    return {"tv": top["tv"], "undo": undo, "reply": say(state, "Undone. " + td.describe(top["tv"])), "task": None, "last": {},
            "path": state["path"] + ["do_undo"]}


def answer_info(state):
    """Questions: the time, the status, help. Asked in the middle of a task, they are answered and the task goes on."""
    intent = state["nlu"]["intent"]
    tv = state["tv"]
    if intent == "ask_time":
        text = "It is " + td.say_time(tv["clock"]) + "."
    elif intent == "ask_status":
        text = td.describe(tv)
    else:
        text = HELP_TEXT[""]
        topic = str(state["nlu"]["slots"].get("topic", ""))
        for name in tn.TOPICS:
            if topic.startswith(name):
                text = HELP_TEXT[name]
    return {"reply": say(state, join_text(text, reminder(state))), "path": state["path"] + ["answer_info"]}


def refuse(state):
    """A request that is not about the TV: say no politely, and say what we can do."""
    text = "Sorry, I can only help with the TV, for example the volume, the channel or an app."
    return {"reply": say(state, join_text(text, reminder(state))), "failures": 0, "path": state["path"] + ["refuse"]}


def tv_off(state):
    return {"reply": say(state, "The TV is off. Say turn on the TV first."), "task": None, "path": state["path"] + ["tv_off"]}


def cancel(state):
    return {"reply": say(state, "Okay, cancelled."), "task": None, "pending": None, "choices": {}, "awaiting": "", "queue": [],
            "path": state["path"] + ["cancel"]}


def nothing_pending(state):
    if state["nlu"]["intent"] == "cancel":
        return {"reply": say(state, "There is nothing to cancel."), "path": state["path"] + ["nothing_pending"]}
    return {"reply": say(state, "I did not ask you anything. What would you like to do?"), "path": state["path"] + ["nothing_pending"]}


def fallback(state):
    """Repair: first ask again, then offer a short menu, then hand over to a person."""
    count = state.get("failures", 0) + 1
    if count == 1:
        text = join_text("Sorry, I did not catch that. Please say it again.", reminder(state))
        return {"reply": say(state, text), "failures": 1, "path": state["path"] + ["fallback"]}
    if count == 2:
        return {"reply": say(state, MENU), "failures": 2, "path": state["path"] + ["fallback"]}
    return {"reply": say(state, HANDOVER), "failures": 0, "handover": True, "task": None, "pending": None, "choices": {}, "awaiting": "",
            "queue": [], "path": state["path"] + ["fallback"]}


# ------------------------------------------------------------------------------------------------- the graph
SPEAKING = ["ask_slot", "explain", "clarify", "pick_choice", "confirm", "ask_pin", "check_pin", "release", "execute", "do_undo",
            "answer_info", "refuse", "tv_off", "cancel", "nothing_pending", "fallback"]
NODES = {"ask_slot": ask_slot, "explain": explain, "clarify": clarify, "pick_choice": pick_choice, "confirm": confirm,
         "ask_pin": ask_pin, "check_pin": check_pin, "release": release, "execute": execute, "do_undo": do_undo,
         "answer_info": answer_info, "refuse": refuse, "tv_off": tv_off, "cancel": cancel, "nothing_pending": nothing_pending,
         "fallback": fallback}


def build_graph(route_fn=None):
    """The dialogue graph: understand, track, route (a decision), then one speaking node, then maybe the next clause."""
    graph = StateGraph(TVState)
    graph.add_node("understand", understand)
    graph.add_node("track", track)
    for name in SPEAKING:
        graph.add_node(name, NODES[name])
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "track")
    if route_fn is None:
        route_fn = route
    graph.add_conditional_edges("track", route_fn, SPEAKING)
    for name in SPEAKING:
        graph.add_conditional_edges(name, next_step, {"execute": "execute", "track": "track", "end": END})
    return graph.compile(checkpointer=InMemorySaver())


# ------------------------------------------------------------------------------------------------- the interface
def make_manager(use_model=False, adapter=None, confidence=True, reader=None, route_fn=None):
    """Build a manager (a plain dictionary). use_model=False: keyword rules, no GPU. use_model=True: the small model with its adapter.

    reader: an already loaded reader (from tv_nlu.load_reader), so that several managers can share one model.
    route_fn: another routing function, used by the 'break it on purpose' section.
    """
    if reader is None:
        if use_model:
            reader = tn.load_reader(adapter)
            reader["confidence"] = confidence
        else:
            reader = tn.make_rules_reader()
    manager = {"app": build_graph(route_fn), "reader": reader, "reads": [], "errors": [], "turns": 0, "count": 0}
    reset(manager)
    return manager


def reset(manager, state=None):
    """Start a new conversation and a new TV. `state` is an optional dictionary of TV values to change."""
    manager["count"] = manager["count"] + 1
    manager["thread"] = "tv-" + str(manager["count"])
    manager["start_tv"] = td.new_tv(state)
    manager["first"] = True
    manager["reads"] = []
    return manager


def get_state(manager):
    """The device state now (a copy)."""
    if manager["first"]:
        return copy.deepcopy(manager["start_tv"])
    values = manager["app"].get_state({"configurable": {"thread_id": manager["thread"]}}).values
    return copy.deepcopy(values["tv"])


def turn(manager, text):
    """One text turn. Returns a dictionary: reply, state (the TV after the turn), path, intent, confidence, heard, awaiting, handover."""
    ACTIVE["reader"] = manager["reader"]
    ACTIVE["reads"] = manager["reads"]
    config = {"configurable": {"thread_id": manager["thread"]}, "recursion_limit": 60}
    payload = {"utterance": str(text)}
    if manager["first"]:
        payload["tv"] = copy.deepcopy(manager["start_tv"])
        manager["first"] = False
    manager["turns"] = manager["turns"] + 1
    try:
        out = manager["app"].invoke(payload, config)
    except Exception as problem:
        manager["errors"].append(str(problem))
        return {"reply": "Sorry, something went wrong.", "state": get_state(manager), "path": ["error"], "intent": "other",
                "confidence": None, "heard": str(text), "awaiting": "", "handover": False, "error": str(problem)}
    nlu = out.get("nlu") or {}
    return {"reply": out["reply"], "state": copy.deepcopy(out["tv"]), "path": list(out["path"]), "intent": nlu.get("intent", "other"),
            "confidence": nlu.get("confidence"), "heard": str(text), "awaiting": out.get("awaiting", ""),
            "handover": out.get("handover", False), "slots": nlu.get("slots", {})}


def show_turn(text, result, show="changes", before=None):
    """Print one turn: what was said, the reply, the path, and the device state (all of it, or only what changed)."""
    print("  you  :", text)
    print("  tv   :", result["reply"])
    print("         path:", " > ".join(result["path"]), "| intent:", result["intent"])
    if show == "full":
        print("         state:", result["state"])
    elif before is not None:
        changes = td.what_changed(before, result["state"])
        print("         changed:", changes)


def run_conversation(manager, turns, show="changes"):
    """Send the turns one by one into the same conversation. show: 'full' (the whole state dictionary), 'changes' or False."""
    results = []
    before = get_state(manager)
    for text in turns:
        result = turn(manager, text)
        if show:
            show_turn(text, result, show, before)
        before = result["state"]
        results.append(result)
    return results


# ------------------------------------------------------------------------------------------------- scenarios
# A scenario is a conversation with the exact replies we expect, and optionally the state of the TV at the end.
# An expected reply is a text (must match exactly) or ("has", "some words") when only part of the reply is fixed.
# needs: "rules" = the keyword rules can read every message (so the logic can be tested with no GPU);
#        "model" = needs the language model (noisy speech, Hinglish, everyday wording).
CONFIRM_RESET = "This will erase all your settings and apps. Shall I go ahead? (yes/no)"
CONFIRM_DELETE = "This will delete all 3 recordings for good. Shall I go ahead? (yes/no)"
CONFIRM_RENT = "Rent Mumbai Nights for 149 rupees. Shall I go ahead? (yes/no)"
ASK_PIN = "This needs your 4-digit PIN. Please say it now, or say cancel."
OUT_OF_SCOPE_TEXT = "Sorry, I can only help with the TV, for example the volume, the channel or an app."
FUNNY_LIST = "I found these. 1, Laugh Riot. 2, Office Hours. 3, Funny Farm. Which one? Say the number or the name."
OFFICE_APPS = "It is on more than one app. 1, Netflix. 2, Prime Video. Which one? Say the number or the name."
SPORTS_LIST = "More than one channel fits. 1, Star Sports 1. 2, Sony Sports. Which one? Say the number or the name."
NEWS_LIST = "More than one channel fits. 1, Aaj Tak. 2, NDTV. Which one? Say the number or the name."
SORRY = "Sorry, I did not catch that. Please say it again."
LOCKED = "Adult content is locked after wrong PINs. Please ask the account owner."
UNDONE_START = "Undone. The TV is on. Channel 1, DD National. Volume 20."
FACTORY_DONE = "Done. The TV is back to its factory settings."
RENT_DONE = "Done. Mumbai Nights is rented for 149 rupees. Enjoy."
ASK_LEVEL = "What volume, from 0 to 100?"


def scenario(name, pattern, turns, replies, state=None, start=None, needs="rules", handover=None, never_read=None):
    """One scenario as a dictionary. state: values the TV must have at the end. start: values to change in the starting TV."""
    if state is None:
        state = {}
    if never_read is None:
        never_read = []
    return {"name": name, "pattern": pattern, "turns": turns, "replies": replies, "state": state, "start": start,
            "needs": needs, "handover": handover, "never_read": never_read}


SCENARIOS = [
    scenario("volume asked, then given", "slot filling", ["set the volume", "40"],
             [ASK_LEVEL, "Volume set to 40."], {"volume": 40}),
    scenario("channel asked, then a name", "slot filling", ["change the channel", "aaj tak"],
             ["Which channel?", "Channel 7, Aaj Tak."], {"channel": 7}),
    scenario("movie asked, then a title, then yes", "slot filling", ["I want to rent a movie", "mumbai nights", "yes"],
             ["Which movie?", CONFIRM_RENT, RENT_DONE], {"rented": ["mumbai nights"]}),
    scenario("it means the volume", "context", ["turn it up", "turn it up by 10"],
             ["Volume up to 25.", "Volume up to 35."], {"volume": 35}),
    scenario("the same on Netflix", "context", ["play something funny on youtube", "the same on netflix", "the second one"],
             ["Playing Funny Farm on YouTube.", "I found these. 1, Laugh Riot. 2, Office Hours. Which one? Say the number or the name.",
              "Playing Office Hours on Netflix."], {"app": "netflix", "now_playing": "Office Hours"}),
    scenario("the second one, then the app", "context", ["play something funny", "the second one", "1"],
             [FUNNY_LIST, OFFICE_APPS, "Playing Office Hours on Netflix."], {"app": "netflix", "now_playing": "Office Hours"}),
    scenario("a bare number corrects the last command", "context", ["set the volume to 30", "no, 45"],
             ["Volume set to 30.", "Volume set to 45."], {"volume": 45}),
    scenario("no, channel 7", "correction", ["go to channel 5", "no, channel 7"],
             ["Channel 5, Star Sports 1.", "Channel 7, Aaj Tak."], {"channel": 7}),
    scenario("a new value replaces a rejected one", "correction", ["set a sleep timer for 500 minutes", "actually 30"],
             ["The sleep timer goes from 1 to 180 minutes. How many minutes?", "Sleep timer set. The TV will switch off in 30 minutes."],
             {"sleep_minutes": 30}),
    scenario("an impossible volume is not accepted", "correction", ["set the volume to 150", "80"],
             ["Sorry, that volume is not possible. " + ASK_LEVEL, "Volume set to 80."], {"volume": 80}),
    scenario("two sports channels: choose by name", "ambiguity", ["put on a sports channel", "sony"],
             [SPORTS_LIST, "Channel 6, Sony Sports."], {"channel": 6}),
    scenario("two movie channels: choose by number", "ambiguity", ["go to the movies channel", "the second one"],
             ["More than one channel fits. 1, Star Movies. 2, Zee Cinema. Which one? Say the number or the name.", "Channel 9, Zee Cinema."],
             {"channel": 9}),
    scenario("a wrong number is not guessed", "ambiguity", ["put on a news channel", "number 5", "1"],
             [NEWS_LIST, "Please choose a number from 1 to 2. " + NEWS_LIST, "Channel 7, Aaj Tak."], {"channel": 7}),
    scenario("a new command ends the choice", "ambiguity", ["put on a news channel", "volume up"],
             [NEWS_LIST, "Volume up to 25."], {"volume": 25, "channel": 1}),
    scenario("a question in the middle of a task", "interruption", ["set a sleep timer", "what time is it", "30"],
             ["How many minutes?", "It is 8:30 pm. Back to your request: How many minutes?", "Sleep timer set. The TV will switch off in 30 minutes."],
             {"sleep_minutes": 30}),
    scenario("a question in the middle of a confirmation", "interruption", ["delete all recordings", "what time is it", "yes"],
             [CONFIRM_DELETE, "It is 8:30 pm. " + CONFIRM_DELETE, "Done. All recordings were deleted."], {"recordings": []}),
    scenario("a question in the middle of a choice", "interruption", ["play something funny", "what time is it", "3"],
             [FUNNY_LIST, "It is 8:30 pm. Back to your choice. 1, Laugh Riot. 2, Office Hours. 3, Funny Farm. Say the number or the name.",
              "Playing Funny Farm on YouTube."], {"app": "youtube"}),
    scenario("factory reset needs a yes", "confirmation", ["set the volume to 60", "factory reset", "yes"],
             ["Volume set to 60.", CONFIRM_RESET, FACTORY_DONE], {"volume": 20}),
    scenario("factory reset refused with no", "confirmation", ["set the volume to 60", "factory reset", "no"],
             ["Volume set to 60.", CONFIRM_RESET, "Okay, cancelled."], {"volume": 60}),
    scenario("deleting stopped with cancel", "confirmation", ["delete all recordings", "cancel"],
             [CONFIRM_DELETE, "Okay, cancelled."], {"recordings": ["Match highlights", "News special", "Cooking show"]}),
    scenario("an unclear answer asks again", "confirmation", ["rent mumbai nights", "maybe", "yes"],
             [CONFIRM_RENT, "Sorry, I need a yes or a no. " + CONFIRM_RENT, RENT_DONE], {"rented": ["mumbai nights"]}),
    scenario("buying needs a yes too", "confirmation", ["buy river of dreams", "yes"],
             ["Buy River Of Dreams for 449 rupees. Shall I go ahead? (yes/no)", "Done. River Of Dreams is yours for 449 rupees. Enjoy."],
             {"now_playing": "River Of Dreams"}),
    scenario("a movie that is not in the store", "confirmation", ["rent the long road"],
             [("has", "movie store")], {"rented": []}),
    scenario("nothing to delete", "confirmation", ["delete all recordings"], ["There are no recordings to delete."],
             start={"recordings": []}),
    scenario("undo a volume change", "undo", ["set the volume to 50", "undo"],
             ["Volume set to 50.", UNDONE_START], {"volume": 20}),
    scenario("undo twice", "undo", ["go to channel 5", "mute", "undo", "undo"],
             ["Channel 5, Star Sports 1.", "Muted.", "Undone. The TV is on. Channel 5, Star Sports 1. Volume 20.", UNDONE_START],
             {"channel": 1, "muted": False}),
    scenario("a reset cannot be undone", "undo", ["factory reset", "yes", "undo"],
             [CONFIRM_RESET, FACTORY_DONE, "I cannot undo that. A reset, a purchase or a deletion is final."]),
    scenario("nothing to undo", "undo", ["undo"], ["There is nothing to undo."]),
    scenario("not understood three times", "repair", ["blah blah", "more blah", "hmm"], [SORRY, MENU, HANDOVER], handover=True),
    scenario("the counter starts again after a success", "repair", ["blah blah", "volume up", "blah blah"],
             [SORRY, "Volume up to 25.", SORRY], {"volume": 25}),
    scenario("the menu, then a topic", "repair", ["blah blah", "more blah", "volume", "30"],
             [SORRY, MENU, ASK_LEVEL, "Volume set to 30."], {"volume": 30}),
    scenario("two commands in one sentence", "several commands", ["mute it and go to hdmi 2"],
             ["Muted. Switched to HDMI 2."], {"muted": True, "input": "hdmi 2"}),
    scenario("two different kinds of command", "several commands", ["turn the volume up and open netflix"],
             ["Volume up to 25. Opening Netflix."], {"volume": 25, "app": "netflix"}),
    scenario("three commands", "several commands", ["mute, switch to hdmi 1 and pause"],
             ["Muted. Switched to HDMI 1. Paused."], {"muted": True, "input": "hdmi 1", "playing": False}),
    scenario("the second command waits for a missing detail", "several commands", ["set a sleep timer and mute it", "30"],
             ["How many minutes?", "Sleep timer set. The TV will switch off in 30 minutes. Muted."], {"sleep_minutes": 30, "muted": True}),
    scenario("adult content with the right PIN", "permission tier", ["show adult content", "1234"],
             [ASK_PIN, "PIN accepted. Opening the adult zone."], {"adult_unlocked": True, "app": "adult zone"}, never_read=["1234"]),
    scenario("a wrong PIN, then the right one", "permission tier", ["show adult content", "1111", "one two three four"],
             [ASK_PIN, "That PIN is not right. You have 2 tries left.", "PIN accepted. Opening the adult zone."], {"adult_unlocked": True}),
    scenario("three wrong PINs lock it", "permission tier", ["show adult content", "1111", "2222", "3333", "show adult content"],
             [ASK_PIN, "That PIN is not right. You have 2 tries left.", "That PIN is not right. You have 1 try left.",
              "That PIN is not right. Too many wrong tries, so adult content is locked. Please ask the account owner.", LOCKED],
             {"pin_locked": True, "adult_unlocked": False}),
    scenario("cancel instead of the PIN", "permission tier", ["show adult content", "cancel"],
             [ASK_PIN, "Okay, cancelled."], {"adult_unlocked": False}),
    scenario("switching off locks it again", "permission tier", ["show adult content", "1234", "turn off the tv", "turn on the tv", "show adult content"],
             [ASK_PIN, "PIN accepted. Opening the adult zone.", "Goodbye. The TV is off.", "The TV is on.", ASK_PIN], {"adult_unlocked": False}),
    scenario("a joke is refused politely", "out of scope", ["tell me a joke"], [OUT_OF_SCOPE_TEXT]),
    scenario("refused in the middle of a task", "out of scope", ["set the volume", "order a pizza", "30"],
             [ASK_LEVEL, OUT_OF_SCOPE_TEXT + " Back to your request: " + ASK_LEVEL, "Volume set to 30."], {"volume": 30}),
    scenario("the home lights are not the TV", "out of scope", ["turn on the bedroom light"], [OUT_OF_SCOPE_TEXT]),
    scenario("the TV is off", "state", ["turn off the tv", "volume up", "turn on the tv", "volume up"],
             ["Goodbye. The TV is off.", "The TV is off. Say turn on the TV first.", "The TV is on.", "Volume up to 25."], {"power": "on", "volume": 25}),
    scenario("what is playing", "state", ["go to channel 5", "what is playing"],
             ["Channel 5, Star Sports 1.", "The TV is on. Channel 5, Star Sports 1. Volume 20."]),
    scenario("help on one topic", "state", ["help with the volume"], [("has", "turn it up")]),
    scenario("noisy text: turn on", "noisy speech", ["tern on the tee vee"], ["The TV is on."], {"power": "on"}, start={"power": "off"}, needs="model"),
    scenario("noisy text: volume and mute", "noisy speech", ["turn the volum up by ten", "moot the tv"],
             ["Volume up to 30.", "Muted."], {"volume": 30, "muted": True}, needs="model"),
    scenario("noisy text: app and input", "noisy speech", ["open net flix", "swich to hdmi to"],
             ["Opening Netflix.", "Switched to HDMI 2."], {"input": "hdmi 2"}, needs="model"),
    scenario("Hinglish: volume, input, pause", "Hinglish", ["awaaz 40 kar do", "hdmi 1 pe lagao", "pause kar do"],
             ["Volume set to 40.", "Switched to HDMI 1.", "Paused."], {"volume": 40, "input": "hdmi 1", "playing": False}, needs="model"),
    scenario("Hinglish: reset and haan", "Hinglish", ["tv reset kar do", "haan"],
             [CONFIRM_RESET, FACTORY_DONE], needs="model"),
    scenario("Hinglish: two commands with aur", "Hinglish", ["volume kam karo aur tv band karo"],
             ["Volume down to 15. Goodbye. The TV is off."], {"volume": 15, "power": "off"}, needs="model"),
    scenario("everyday words", "noisy speech", ["I cannot hear"], ["Volume up to 25."], {"volume": 25}, needs="model"),
]


def check_replies(got, expected):
    """Compare the replies of a conversation with the expected ones. Returns '' when they match, else a sentence about the first difference."""
    if len(got) != len(expected):
        return "expected " + str(len(expected)) + " replies, got " + str(len(got))
    for index in range(len(expected)):
        want = expected[index]
        if isinstance(want, str):
            if got[index] != want:
                return "turn " + str(index + 1) + ": expected '" + want + "' but got '" + got[index] + "'"
        else:
            for word in want[1].split():
                if word not in got[index].lower():
                    return "turn " + str(index + 1) + ": the reply should contain '" + word + "' but it was '" + got[index] + "'"
    return ""


def check_state(final_state, wanted):
    """Compare the final TV state with the wanted values. Returns '' when they match, else a sentence about the first difference."""
    for name in wanted:
        if final_state.get(name) != wanted[name]:
            return "state " + name + ": expected " + str(wanted[name]) + " but it is " + str(final_state.get(name))
    return ""


def run_scenario(manager, one):
    """Run one scenario on a fresh conversation and TV. Returns a dictionary: name, pattern, ok, reason, got, needs, turns."""
    reset(manager, one["start"])
    results = []
    for text in one["turns"]:
        results.append(turn(manager, text))
    got = []
    for result in results:
        got.append(result["reply"])
    reason = check_replies(got, one["replies"])
    if reason == "":
        reason = check_state(results[-1]["state"], one["state"])
    if reason == "" and one["handover"] is not None and results[-1]["handover"] != one["handover"]:
        reason = "handover flag: expected " + str(one["handover"])
    for secret in one["never_read"]:
        for heard in manager["reads"]:
            if secret in heard:
                reason = "the reader was given the secret text: " + heard
    return {"name": one["name"], "pattern": one["pattern"], "ok": reason == "", "reason": reason, "got": got, "needs": one["needs"],
            "turns": one["turns"]}


def run_scenarios(manager, scenarios=None, needs=None):
    """Run many scenarios. needs: None for all, or 'rules' to run only those the keyword rules can read.

    Returns (passed, total, results) where results is a list of dictionaries (see run_scenario).
    """
    if scenarios is None:
        scenarios = SCENARIOS
    results = []
    passed = 0
    for one in scenarios:
        if needs is not None and one["needs"] != needs:
            continue
        result = run_scenario(manager, one)
        results.append(result)
        if result["ok"]:
            passed = passed + 1
    return passed, len(results), results


def patterns_covered(scenarios=None):
    """The dialogue patterns the scenarios cover, with how many scenarios each has: a dictionary."""
    if scenarios is None:
        scenarios = SCENARIOS
    counts = {}
    for one in scenarios:
        counts[one["pattern"]] = counts.get(one["pattern"], 0) + 1
    return counts


def scenario_rows(results):
    """The results of run_scenarios as a list of dictionaries for a table: pattern, scenario, turns, result, why."""
    rows = []
    for result in results:
        mark = "PASS"
        if not result["ok"]:
            mark = "FAIL"
        rows.append({"pattern": result["pattern"], "scenario": result["name"], "turns": len(result["turns"]), "result": mark,
                     "why": result["reason"]})
    return rows


def compare_by_pattern(rules_results, model_results):
    """For each dialogue pattern: how many scenarios, and how many pass with the rules and with the model. A list of dictionaries."""
    table = {}
    order = []
    for result in rules_results:
        pattern = result["pattern"]
        if pattern not in table:
            table[pattern] = {"pattern": pattern, "scenarios": 0, "rules pass": 0, "model pass": 0}
            order.append(pattern)
        table[pattern]["scenarios"] = table[pattern]["scenarios"] + 1
        if result["ok"]:
            table[pattern]["rules pass"] = table[pattern]["rules pass"] + 1
    for result in model_results:
        if result["ok"]:
            table[result["pattern"]]["model pass"] = table[result["pattern"]]["model pass"] + 1
    rows = []
    for pattern in order:
        rows.append(table[pattern])
    return rows
