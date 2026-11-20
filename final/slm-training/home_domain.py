"""The Day 2 domain: a smart home (lights, AC, blinds, scenes, a door) plus a small appliance manual to ask questions about.

Everything the model may do goes through the TOOLS below. The model proposes a call; `validate_call` decides if it is allowed;
`execute` changes the home. Provided as a helper so the notebooks can concentrate on LangGraph.
"""
import copy
import math
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
ROOMS = ["living room", "bedroom", "kitchen", "study", "balcony"]
SCENES = ["movie night", "good morning", "away"]
AC_RANGE = (16, 30)


def new_home():
    home = {"locked": True, "scene": "none"}
    for r in ROOMS:
        home[f"light.{r}"] = "off"
        home[f"ac.{r}"] = "off"
        home[f"blinds.{r}"] = "closed"
    return home


# ------------------------------------------------------------------------------------------------- tools
def _tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                              "parameters": {"type": "object", "properties": properties, "required": required}}}


_room = {"type": "string", "enum": ROOMS, "description": "the room"}
TOOLS = [
    _tool("set_light", "Turn a light on or off in a room.", {"room": _room, "state": {"type": "string", "enum": ["on", "off"]}}, ["room", "state"]),
    _tool("set_ac", "Set the air conditioner of a room to a temperature in degrees Celsius (16 to 30).",
          {"room": _room, "temp": {"type": "integer", "description": "16 to 30"}}, ["room", "temp"]),
    _tool("ac_off", "Switch the air conditioner of a room off.", {"room": _room}, ["room"]),
    _tool("set_blinds", "Open or close the blinds in a room.", {"room": _room, "state": {"type": "string", "enum": ["open", "closed"]}}, ["room", "state"]),
    _tool("run_scene", "Run a scene: movie night, good morning or away.", {"scene": {"type": "string", "enum": SCENES}}, ["scene"]),
    _tool("unlock_door", "Unlock the front door.", {}, []),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
RISKY = {"unlock_door"}                      # tools that must never run without a human saying yes


def risky(name, args):
    return name in RISKY or (name == "run_scene" and args.get("scene") == "away")


def validate_call(name, args):
    """Is this tool call allowed? Returns (clean_arguments, None) or (None, an error message for the model)."""
    if name not in TOOL_NAMES:
        return None, f"Unknown tool '{name}'. Available tools: {sorted(TOOL_NAMES)}."
    if not isinstance(args, dict):
        return None, "Arguments must be a JSON object."
    clean = {}
    if name in ("set_light", "set_ac", "ac_off", "set_blinds"):
        room = str(args.get("room", "")).lower()
        if room not in ROOMS:
            return None, f"Unknown room '{args.get('room')}'. Rooms: {ROOMS}."
        clean["room"] = room
    if name in ("set_light", "set_blinds"):
        allowed = ["on", "off"] if name == "set_light" else ["open", "closed"]
        st = str(args.get("state", "")).lower()
        if st not in allowed:
            return None, f"state must be one of {allowed}."
        clean["state"] = st
    if name == "set_ac":
        try:
            t = int(float(args.get("temp")))
        except (TypeError, ValueError):
            return None, f"temp must be a whole number from {AC_RANGE[0]} to {AC_RANGE[1]}."
        if not AC_RANGE[0] <= t <= AC_RANGE[1]:
            return None, f"temp {t} is outside the allowed range {AC_RANGE[0]} to {AC_RANGE[1]}."
        clean["temp"] = t
    if name == "run_scene":
        sc = str(args.get("scene", "")).lower()
        if sc not in SCENES:
            return None, f"Unknown scene '{args.get('scene')}'. Scenes: {SCENES}."
        clean["scene"] = sc
    return clean, None


def execute(home, name, args):
    """Apply a validated call. Returns (new_home, message). Never mutates the home you pass in."""
    home = copy.deepcopy(home)
    if name == "set_light":
        home[f"light.{args['room']}"] = args["state"]
        return home, f"The {args['room']} light is {args['state']}."
    if name == "set_ac":
        home[f"ac.{args['room']}"] = f"{args['temp']} C"
        return home, f"The {args['room']} AC is set to {args['temp']} C."
    if name == "ac_off":
        home[f"ac.{args['room']}"] = "off"
        return home, f"The {args['room']} AC is off."
    if name == "set_blinds":
        home[f"blinds.{args['room']}"] = args["state"]
        return home, f"The {args['room']} blinds are {args['state']}."
    if name == "run_scene":
        sc = args["scene"]
        home["scene"] = sc
        if sc == "movie night":
            home["light.living room"], home["blinds.living room"] = "20%", "closed"
        elif sc == "good morning":
            home["light.bedroom"], home["blinds.bedroom"] = "on", "open"
        else:
            for k in list(home):
                if k.startswith(("light.", "ac.")):
                    home[k] = "off"
                elif k.startswith("blinds."):
                    home[k] = "closed"
            home["locked"] = True
        return home, f"Scene '{sc}' is running."
    if name == "unlock_door":
        home["locked"] = False
        return home, "The front door is unlocked."
    return home, f"Nothing to do for '{name}'."


def describe(home):
    """A short text of the home's state, handy to print."""
    on = [k.split(".", 1)[1] for k, v in home.items() if k.startswith("light.") and v != "off"]
    ac = {k.split(".", 1)[1]: v for k, v in home.items() if k.startswith("ac.") and v != "off"}
    return f"lights on: {on or 'none'} | AC: {ac or 'all off'} | door: {'locked' if home['locked'] else 'UNLOCKED'} | scene: {home['scene']}"


# ------------------------------------------------------------------------------------------------- a small manual to ask about
def manual_sections():
    out, title, buf = [], None, []
    for line in (DATA / "manual.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title:
                out.append({"title": title, "text": " ".join(buf).strip()})
            title, buf = line[3:].strip(), []
        elif title and line.strip():
            buf.append(line.strip())
    if title:
        out.append({"title": title, "text": " ".join(buf).strip()})
    return out


_SEC = None
STOP = set("a an the is are was be to of in on at my me i we you it its and or for with what does do how can should when why who this that "
           "there which will would please".split())


def retrieve(question, k=2):
    """The k manual sections that share the most (rare) words with the question. Simple keyword search: Day 1's RAG, in miniature."""
    global _SEC
    if _SEC is None:
        _SEC = [(s, Counter(re.findall(r"[a-z0-9]+", f"{s['title']} {s['text']}".lower()))) for s in manual_sections()]
    q = set(re.findall(r"[a-z0-9]+", question.lower())) - STOP
    df = Counter(w for _, c in _SEC for w in c)
    scored = sorted(((sum(math.log(1 + len(_SEC) / df[w]) for w in q if w in c), s) for s, c in _SEC), key=lambda x: -x[0])
    return [s for sc, s in scored[:k] if sc > 0]
