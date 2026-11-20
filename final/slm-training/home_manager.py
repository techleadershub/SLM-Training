"""A dialogue manager for the smart home, built as a LangGraph graph. Used by Labs 13 and 14.

The language model appears in TWO nodes: `understand` (message to JSON) and `answer_manual` (answers a question from the manual).
Everything else is code: tracking the task, deciding what happens next, validating every value, changing the home.
The same graph can run with a rules-based `understand` instead, so the two can be compared on identical conversations.
"""
import re
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import home_domain as hd

REQUIRED = {"set_light": ["room", "state"], "set_ac": ["room", "temp"], "set_blinds": ["room", "state"], "run_scene": ["scene"]}
TASKS = list(REQUIRED)
QUESTIONS = {"room": "Which room?", "temp": "What temperature, from 16 to 30?", "scene": "Which scene: movie night, good morning or away?"}
STATE_Q = {"set_light": "On or off?", "set_blinds": "Open or closed?"}
MENU = "I can control lights, the air conditioner, blinds and scenes, or answer questions about your appliances. Which one would you like?"
CONFIRM_TEXT = "This will switch everything off and lock up. Shall I go ahead? (yes/no)"


class DialogueState(TypedDict, total=False):
    utterance: str
    nlu: dict            # what the message means: intent and slot values (THIS message only)
    task: dict           # the task in progress: {"intent", "slots"} (kept between turns)
    home: dict           # the state of the home
    awaiting: str        # "confirm" while we wait for yes or no
    confirmed: bool
    candidates: list     # rooms the user may have meant, when the message was ambiguous
    failures: int        # messages in a row that were not understood
    reply: str
    path: list           # the nodes that ran in THIS turn


# ------------------------------------------------------------------------------------------------- understanding
def clean_slots(slots):
    """Keep only values the home knows. Returns (clean, dropped): the model proposes, the code decides."""
    clean, dropped = {}, []
    for k, v in (slots or {}).items() if isinstance(slots, dict) else []:
        if k == "room" and str(v).lower() in hd.ROOMS:
            clean[k] = str(v).lower()
        elif k == "state" and str(v).lower() in ("on", "off", "open", "closed"):
            clean[k] = str(v).lower()
        elif k == "scene" and str(v).lower() in hd.SCENES:
            clean[k] = str(v).lower()
        elif k == "temp":
            try:
                t = int(float(v))
                if hd.AC_RANGE[0] <= t <= hd.AC_RANGE[1]:
                    clean[k] = t
                else:
                    dropped.append(k)
            except (TypeError, ValueError):
                dropped.append(k)
        else:
            dropped.append(k)
    return clean, dropped


UNDERSTAND_SYSTEM = """You read ONE message sent to a smart-home assistant and reply with JSON only: {"intent": ..., "slots": {...}}.
intent is one of: set_light, set_ac, set_blinds, run_scene, ask_manual (a question about an appliance, an error code or how to use it),
yes, no, cancel, other.
slots may contain: room (living room, bedroom, kitchen, study, balcony), state (on or off for lights; open or closed for blinds),
temp (a whole number), scene (movie night, good morning, away). Only include what the user actually said.
"It's boiling in here" means set_ac. A message that only gives a value, like "the kitchen" or "22", has intent other.
Examples:
"turn on the kitchen light" -> {"intent": "set_light", "slots": {"room": "kitchen", "state": "on"}}
"the bedroom" -> {"intent": "other", "slots": {"room": "bedroom"}}
"bedroom" -> {"intent": "other", "slots": {"room": "bedroom"}}
"22" -> {"intent": "other", "slots": {"temp": 22}}
"actually 20" -> {"intent": "other", "slots": {"temp": 20}}
"what does E47 mean?" -> {"intent": "ask_manual", "slots": {}}
"sure, go ahead" -> {"intent": "yes", "slots": {}}"""


def slm_understand(state):
    """The language model reads the message; code checks every value it returned."""
    from graph_slm import ask_json
    parsed, raw = ask_json(UNDERSTAND_SYSTEM, f'"{state["utterance"]}"')
    intent = parsed.get("intent") if parsed else None
    ok = intent in TASKS + ["ask_manual", "yes", "no", "cancel", "other"]
    slots, dropped = clean_slots(parsed.get("slots") if parsed else {})
    return {"nlu": {"intent": intent if ok else "other", "slots": slots, "dropped": dropped, "raw": raw, "valid_json": parsed is not None},
            "candidates": [], "path": ["understand"]}


def _rules(text):
    t = text.lower()
    slots, intent = {}, "other"
    if re.fullmatch(r"\W*(yes|yeah|yep|sure|ok|okay|go ahead)\W*", t):
        return {"intent": "yes", "slots": {}}
    if re.fullmatch(r"\W*(no|nope|not really)\W*", t):
        return {"intent": "no", "slots": {}}
    if re.search(r"\b(cancel|never mind|stop)\b", t):
        return {"intent": "cancel", "slots": {}}
    if re.search(r"\b(light|lights|lamp)\b", t):
        intent = "set_light"
    elif re.search(r"\b(ac|air conditioner)\b", t):
        intent = "set_ac"
    elif re.search(r"\b(blind|blinds|curtain)\b", t):
        intent = "set_blinds"
    for sc in hd.SCENES:
        if sc in t:
            intent, slots["scene"] = "run_scene", sc
    if "leaving" in t:
        intent, slots["scene"] = "run_scene", "away"
    if intent == "other" and (re.search(r"\b(e\d\d|ch\d\d|f\d\d|u1)\b", t) or re.match(r"\s*(what|how|why|when)\b", t)):
        return {"intent": "ask_manual", "slots": {}}
    for r in hd.ROOMS:
        if r in t:
            slots["room"] = r
    m = re.search(r"\b(on|off)\b", t)
    if m:
        slots["state"] = m.group(1)
    if re.search(r"\bopen\b", t):
        slots["state"] = "open"
    if re.search(r"\b(close|closed|shut)\b", t):
        slots["state"] = "closed"
    m = re.search(r"\b(\d{2})\b", t)
    if m:
        slots["temp"] = int(m.group(1))
    return {"intent": intent, "slots": slots}


def rules_understand(state):
    """Keyword rules: the baseline. Same output shape as the model version."""
    nlu = _rules(state["utterance"])
    nlu["slots"], nlu["dropped"] = clean_slots(nlu["slots"])
    return {"nlu": nlu, "candidates": [], "path": ["understand"]}


# ------------------------------------------------------------------------------------------------- track, route
def missing(task):
    return [s for s in REQUIRED[task["intent"]] if s not in task["slots"]]


def track(state):
    """Update the task in progress: start, add details (context), overwrite them (correction), find out what 'it' means."""
    nlu, task = state["nlu"], state.get("task")
    home = state.get("home") or hd.new_home()
    out = {"home": home, "path": state["path"] + ["track"]}
    if nlu["intent"] in TASKS:
        if task and task["intent"] == nlu["intent"]:
            task = {"intent": task["intent"], "slots": {**task["slots"], **nlu["slots"]}}
        else:
            task = {"intent": nlu["intent"], "slots": dict(nlu["slots"])}
        if task["intent"] == "set_light" and task["slots"].get("state") == "off" and "room" not in task["slots"]:
            lit = [k.split(".", 1)[1] for k, v in home.items() if k.startswith("light.") and v != "off"]
            if len(lit) == 1:
                task["slots"]["room"] = lit[0]           # "turn it off": only one light is on, so that is the one
            elif len(lit) > 1:
                out["candidates"] = lit                  # ambiguous: ask which
    elif nlu["slots"] and task and state.get("awaiting") != "confirm":
        task = {"intent": task["intent"], "slots": {**task["slots"], **nlu["slots"]}}     # a bare value: answer or correction
    out["task"] = task
    return out


def needs_confirm(task):
    return bool(task) and hd.risky(task["intent"], task["slots"])


def route(state):
    """The decision at the heart of the manager. The ORDER of these questions is the behaviour."""
    nlu, task = state["nlu"], state.get("task")
    if nlu["intent"] == "ask_manual":
        return "answer_manual"
    if state.get("awaiting") == "confirm":
        if nlu["intent"] == "yes":
            return "execute"
        return "cancel" if nlu["intent"] in ("no", "cancel") else "confirm"
    if nlu["intent"] == "cancel" and task:
        return "cancel"
    if state.get("candidates"):
        return "clarify"
    if not task:
        return "fallback"
    if missing(task):
        return "ask_slot"
    if needs_confirm(task) and not state.get("confirmed"):
        return "confirm"
    return "execute"


def route_hitl(state):
    """Same questions, but a risky task goes to a node that PAUSES the graph (Lab 14) instead of asking in the next turn."""
    nlu, task = state["nlu"], state.get("task")
    if nlu["intent"] == "ask_manual":
        return "answer_manual"
    if nlu["intent"] == "cancel" and task:
        return "cancel"
    if state.get("candidates"):
        return "clarify"
    if not task:
        return "fallback"
    if missing(task):
        return "ask_slot"
    return "confirm_hitl" if needs_confirm(task) else "execute"


# ------------------------------------------------------------------------------------------------- the nodes that speak
def _next_question(task):
    slot = missing(task)[0]
    return STATE_Q[task["intent"]] if slot == "state" else QUESTIONS[slot]


def ask_slot(state):
    return {"reply": _next_question(state["task"]), "awaiting": "", "failures": 0, "path": state["path"] + ["ask_slot"]}


def clarify(state):
    return {"reply": "Which light do you mean: " + " or ".join(state["candidates"]) + "?", "failures": 0, "path": state["path"] + ["clarify"]}


def answer_manual(state):
    """An interruption: answer a question from the manual with the model, keep the task, and carry on with it."""
    from graph_slm import slm
    found = hd.retrieve(state["utterance"], 2)
    if not found:
        answer = "I could not find that in the manual."
    else:
        ctx = "\n".join(f"[{s['title']}] {s['text']}" for s in found)
        answer = slm([{"role": "system", "content": "Answer in one or two short sentences using ONLY the manual excerpts. "
                       "If they do not contain the answer, say you do not know."},
                      {"role": "user", "content": f"Manual excerpts:\n{ctx}\n\nQuestion: {state['utterance']}"}], max_new_tokens=80)
    task = state.get("task")
    if task and missing(task):
        answer += " Back to your request: " + _next_question(task)
    return {"reply": answer, "failures": 0, "path": state["path"] + ["answer_manual"]}


def confirm(state):
    return {"reply": CONFIRM_TEXT, "awaiting": "confirm", "path": state["path"] + ["confirm"]}


def confirm_hitl(state) -> Command[Literal["execute", "cancel"]]:
    """Pause the whole graph until a human answers. Nothing after this line runs until the graph is resumed."""
    answer = interrupt({"question": CONFIRM_TEXT, "task": state["task"]})
    yes = bool(re.fullmatch(r"\W*(yes|yeah|yep|sure|ok|okay|go ahead)\W*", str(answer).lower()))
    return Command(goto="execute" if yes else "cancel", update={"confirmed": yes, "path": state["path"] + ["confirm_hitl"]})


def execute(state):
    task = state["task"]
    args, error = hd.validate_call(task["intent"], task["slots"])
    if error:                                           # belt and braces: the code checks even what it asked the user for
        return {"reply": error, "task": None, "awaiting": "", "path": state["path"] + ["execute"]}
    home, message = hd.execute(state["home"], task["intent"], args)
    return {"home": home, "reply": "Done. " + message, "task": None, "awaiting": "", "confirmed": False, "failures": 0,
            "path": state["path"] + ["execute"]}


def cancel(state):
    return {"reply": "Okay, cancelled.", "task": None, "awaiting": "", "confirmed": False, "path": state["path"] + ["cancel"]}


def fallback(state):
    n = state.get("failures", 0) + 1
    return {"reply": MENU if n >= 2 else "Sorry, I did not understand.", "failures": n, "path": state["path"] + ["fallback"]}


# ------------------------------------------------------------------------------------------------- the graph
def build(understand=slm_understand, hitl=False, checkpointer=None, route_fn=None):
    g = StateGraph(DialogueState)
    nodes = {"understand": understand, "track": track, "ask_slot": ask_slot, "clarify": clarify, "answer_manual": answer_manual,
             "execute": execute, "cancel": cancel, "fallback": fallback}
    nodes.update({"confirm_hitl": confirm_hitl} if hitl else {"confirm": confirm})
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "understand")
    g.add_edge("understand", "track")
    speaking = [n for n in nodes if n not in ("understand", "track", "confirm_hitl")]
    g.add_conditional_edges("track", route_fn or (route_hitl if hitl else route), list(nodes)[2:])
    for n in speaking:
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def converse(app, turns, thread="demo", show=True):
    """Send the turns one by one into the same conversation. Returns (replies, final_state_values)."""
    cfg, replies = {"configurable": {"thread_id": thread}}, []
    for t in turns:
        out = app.invoke({"utterance": t}, cfg)
        replies.append(out["reply"])
        if show:
            task = out.get("task")
            print(f"  you  : {t}\n  home : {out['reply']}\n         path {' > '.join(out['path'])}" + (f" | task {task['intent']} {task['slots']}" if task else ""))
    return replies, app.get_state(cfg).values


# ------------------------------------------------------------------------------------------------- scenarios
# (name, turns, expected replies). An expected reply is a text, or ("has", "words") when the model writes part of it.
SCENARIOS = [
    ("slot filling", ["turn on the light", "the kitchen"], ["Which room?", "Done. The kitchen light is on."]),
    ("everything in one sentence", ["switch off the bedroom light"], ["Done. The bedroom light is off."]),
    ("context: values over turns", ["set the ac", "bedroom", "22"], ["Which room?", "What temperature, from 16 to 30?", "Done. The bedroom AC is set to 22 C."]),
    ("correction: a later value replaces an earlier one", ["set the ac to 24", "actually 20", "kitchen"],
     ["Which room?", "Which room?", "Done. The kitchen AC is set to 20 C."]),
    ("ambiguity: which light?", ["turn on the kitchen light", "turn on the study light", "turn the light off"],
     ["Done. The kitchen light is on.", "Done. The study light is on.", "Which light do you mean: kitchen or study?"]),
    ("only one light is on, so that one", ["turn on the study light", "turn the light off"], ["Done. The study light is on.", "Done. The study light is off."]),
    ("interruption, then back", ["set the bedroom ac", "what does E47 mean on the washer?", "21"],
     ["What temperature, from 16 to 30?", ("has", "door"), "Done. The bedroom AC is set to 21 C."]),
    ("risky needs a yes", ["i am leaving", "yes"], [CONFIRM_TEXT, "Done. Scene 'away' is running."]),
    ("risky can be refused", ["run away mode", "no"], [CONFIRM_TEXT, "Okay, cancelled."]),
    ("two failures give a menu", ["blah blah", "more blah"], ["Sorry, I did not understand.", MENU]),
    ("an impossible value is not accepted", ["set the study ac to 45"], ["What temperature, from 16 to 30?"]),
]


# People do not talk like keyword rules. The same manager, with messages the rules were never written for.
PARAPHRASES = [
    ("it is boiling", ["it is boiling in the bedroom, make it 22 degrees"], ["Done. The bedroom AC is set to 22 C."]),
    ("kill the lamp", ["can you kill the lamp in the study"], ["Done. The study light is off."]),
    ("misheard words", ["turn on the livin room lite"], ["Done. The living room light is on."]),
    ("cooler, please", ["make the kitchen a bit cooler", "20"], ["What temperature, from 16 to 30?", "Done. The kitchen AC is set to 20 C."]),
    ("lights out", ["lights out in the bedroom"], ["Done. The bedroom light is off."]),
]


def run_scenarios(make_app, scenarios=None, show=True):
    """Run every scenario on a fresh graph. Returns (passed, total, failures)."""
    scenarios, failures = scenarios or SCENARIOS, []
    for i, (name, turns, expected) in enumerate(scenarios):
        got, _ = converse(make_app(), turns, thread=f"s{i}", show=False)
        ok = len(got) == len(expected) and all((g == e if isinstance(e, str) else all(w in g.lower() for w in e[1].split())) for g, e in zip(got, expected))
        if show:
            print(("PASS " if ok else "FAIL ") + name)
        if not ok:
            failures.append((name, turns, expected, got))
    return len(scenarios) - len(failures), len(scenarios), failures
