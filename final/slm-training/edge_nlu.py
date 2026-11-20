"""Day 3 helper: the NLU data, the golden test set and the scoring.

Day 2's assistant reads a message and returns {"intent": ..., "slots": {...}}  (see slm_understand in home_manager.py).
Day 3 makes that one step better, smaller and faster, so we keep the SAME intents and the SAME slots.

Everything here is plain Python (for loops, if, lists, dictionaries) so you can read it from top to bottom.
Nothing in Days 1 and 2 is changed: this file only reads home_domain.py.
"""
import json
import random
from pathlib import Path

import home_domain

DATA = Path(__file__).resolve().parent / "data"

INTENTS = ["set_light", "set_ac", "set_blinds", "run_scene", "ask_manual", "yes", "no", "cancel", "other"]
ROOMS = home_domain.ROOMS            # living room, bedroom, kitchen, study, balcony
SCENES = home_domain.SCENES          # movie night, good morning, away
TEMPS = [18, 20, 21, 22, 23, 24, 25, 26, 27]

# The short instruction used while training (the model learns the rules, so the prompt can be short).
SHORT_SYSTEM = 'Read the message sent to a smart-home assistant. Reply with JSON only: {"intent": ..., "slots": {...}}.'


# ------------------------------------------------------------------------------------------------ the sentences
# Each list holds different ways to say the same thing. {room}, {temp} and {scene} are filled in below.
LIGHT_ON = [
    "turn on the {room} light", "switch on the {room} light", "{room} light on please", "can you put the light on in the {room}",
    "it is too dark in the {room}", "lights on in the {room}", "I need the {room} light", "please turn the {room} lights on",
    "could you switch on the light in the {room}", "{room} light on",
]
LIGHT_OFF = [
    "turn off the {room} light", "switch off the {room} light", "{room} light off please", "can you put the light off in the {room}",
    "kill the {room} light", "lights off in the {room}", "I do not need the {room} light", "please turn the {room} lights off",
    "could you switch off the light in the {room}", "{room} light off",
]
AC_TEMP = [
    "set the {room} ac to {temp}", "make the {room} {temp} degrees", "{room} ac {temp} please", "I am feeling hot in the {room}, set {temp}",
    "can you cool the {room} to {temp}", "ac in the {room} to {temp} degrees", "put the air conditioner on {temp} in the {room}",
    "set the air conditioner in the {room} to {temp}", "{room} ac to {temp}",
]
AC_NO_TEMP = ["set the ac", "set the {room} ac", "it is boiling in here", "it is too hot in the {room}", "change the ac please"]
BLINDS_OPEN = [
    "open the {room} blinds", "open the blinds in the {room}", "pull up the {room} blinds", "let some light in, open the {room} blinds",
    "{room} blinds open please", "open the curtains in the {room}",
]
BLINDS_CLOSE = [
    "close the {room} blinds", "close the blinds in the {room}", "pull down the {room} blinds", "shut the {room} blinds",
    "{room} blinds closed please", "draw the curtains in the {room}",
]
SCENE_SAYINGS = ["run {scene}", "start {scene} mode", "{scene} please", "activate the {scene} scene", "set the scene to {scene}"]
LEAVING = ["I am leaving", "I am going out now", "run away mode", "lock up, I am leaving", "I am off to work", "I am heading out"]
MANUAL_CODES = ["E21", "E03", "F03", "U1", "CH01", "CH05"]
MANUAL_CODE_SAYINGS = ["what does {code} mean", "my washer shows {code}, what do I do", "what is error {code}", "how do I fix {code}"]
MANUAL_OTHER = [
    "how do I clean the drum", "how do I clean the ac filter", "how much detergent should I use", "how do I switch on the child lock",
    "how long does the quick wash take", "what is the maximum load for cotton", "how do I use the ac remote control",
]
YES_SAYINGS = ["yes", "yeah", "yep", "sure", "okay", "go ahead", "yes please", "sure, do it", "that is fine, go ahead"]
NO_SAYINGS = ["no", "nope", "not really", "no thanks", "I do not want that"]
CANCEL_SAYINGS = ["cancel", "cancel that", "never mind", "stop", "forget it", "leave it"]
OUT_OF_DOMAIN = [
    "what is the capital of France", "tell me a joke", "play some music", "order a pizza", "who won the cricket match",
    "what is the weather today", "how far is the moon", "what time is it in London", "remind me to call mum",
    "tell me a story", "how do I cook rice", "what is two plus two", "book me a cab",
]

# The same ideas in Hinglish (Hindi words written in English letters, mixed with English), the way people really speak.
HI_LIGHT_ON = ["{room} ki light on karo", "{room} mein light jala do", "{room} ki batti jalao", "{room} ki light chalu karo"]
HI_LIGHT_OFF = ["{room} ki light band karo", "{room} ki light off kar do", "{room} ki batti bujhao", "{room} mein light band kar do"]
HI_AC = ["{room} ka ac {temp} par kar do", "{room} ka ac {temp} degree pe chalao", "ac {temp} pe set karo {room} mein"]
HI_BLINDS_OPEN = ["{room} ke parde khol do", "{room} ki blinds open karo", "{room} ke parde kholo"]
HI_BLINDS_CLOSE = ["{room} ke parde band karo", "{room} ki blinds close kar do", "{room} ke parde girao"]
HI_SCENE = ["{scene} laga do", "{scene} mode chalu karo"]
HI_LEAVING = ["main bahar ja raha hoon", "ghar band kar do, main ja raha hoon", "main office ja raha hoon"]
HI_YES = ["haan", "haan kar do", "theek hai kar do", "bilkul"]
HI_NO = ["nahi", "nahi chahiye", "mat karo"]
HI_CANCEL = ["rehne do", "chhod do", "cancel kar do"]
HI_OUT = ["ek joke sunao", "aaj ka mausam kaisa hai", "gaana chalao", "pizza order kar do"]

# What a speech-recognition system gets wrong. Each pair is (correct text, what the recogniser may write instead).
NOISE_PAIRS = [
    ("living room", "livin room"), ("living room", "leaving room"), ("kitchen", "kitchin"), ("bedroom", "bed room"),
    ("blinds", "blends"), ("light", "lite"), (" off", " of"), ("turn on", "tern on"), (" ac ", " a c "), ("ac ", "a c "),
    ("degrees", "degree"), ("balcony", "balconi"), ("study", "studdy"), ("switch", "swich"), ("close", "clothes"),
]


# ------------------------------------------------------------------------------------------------ small helpers
def fill(template, room="", temp="", scene="", code=""):
    """Put the values into a sentence template."""
    text = template.replace("{room}", room)
    text = text.replace("{temp}", str(temp))
    text = text.replace("{scene}", scene)
    text = text.replace("{code}", code)
    return text


def make_noisy(text, rng):
    """Mimic speech-recognition mistakes: each known mistake is applied half of the time."""
    noisy = text.lower()
    for correct, wrong in NOISE_PAIRS:
        if correct in noisy and rng.random() < 0.5:
            noisy = noisy.replace(correct, wrong)
    return noisy


def answer_text(intent, slots):
    """The model's answer, written as JSON text."""
    return json.dumps({"intent": intent, "slots": slots})


def make_example(text, intent, slots):
    """One training example: the chat messages the model must learn to continue."""
    return {
        "text": text,
        "intent": intent,
        "slots": slots,
        "messages": [
            {"role": "system", "content": SHORT_SYSTEM},
            {"role": "user", "content": '"' + text + '"'},
            {"role": "assistant", "content": answer_text(intent, slots)},
        ],
    }


# ------------------------------------------------------------------------------------------------ the training set
def make_training_examples(how_many=1200, seed=7):
    """Build training examples from structured facts: an intent plus slot values, said in many different ways."""
    rng = random.Random(seed)
    examples = []
    seen = set()
    tries = 0
    golden_texts = set()                     # the test messages must never appear in training
    for case in load_golden():
        golden_texts.add(case["text"].lower())
    while len(examples) < how_many and tries < how_many * 20:
        tries = tries + 1
        kind = rng.choice(["light", "light", "light", "ac", "ac", "blinds", "blinds", "scene", "scene", "manual", "manual", "short", "short", "short", "other", "other"])
        room = rng.choice(ROOMS)
        temp = rng.choice(TEMPS)
        scene = rng.choice(SCENES)
        hinglish = rng.random() < 0.25
        text = ""
        intent = "other"
        slots = {}

        if kind == "light":
            state = rng.choice(["on", "off"])
            if hinglish:
                templates = HI_LIGHT_ON if state == "on" else HI_LIGHT_OFF
            else:
                templates = LIGHT_ON if state == "on" else LIGHT_OFF
            text = fill(rng.choice(templates), room=room)
            intent, slots = "set_light", {"room": room, "state": state}
        elif kind == "ac":
            roll = rng.random()
            if roll < 0.8:
                templates = HI_AC if hinglish else AC_TEMP
                text = fill(rng.choice(templates), room=room, temp=temp)
                intent, slots = "set_ac", {"room": room, "temp": temp}
            else:
                text = fill(rng.choice(AC_NO_TEMP), room=room)
                intent = "set_ac"
                slots = {"room": room} if "{room}" in text or room in text else {}
        elif kind == "blinds":
            state = rng.choice(["open", "closed"])
            if hinglish:
                templates = HI_BLINDS_OPEN if state == "open" else HI_BLINDS_CLOSE
            else:
                templates = BLINDS_OPEN if state == "open" else BLINDS_CLOSE
            text = fill(rng.choice(templates), room=room)
            intent, slots = "set_blinds", {"room": room, "state": state}
        elif kind == "scene":
            if rng.random() < 0.4:
                text = rng.choice(HI_LEAVING if hinglish else LEAVING)
                intent, slots = "run_scene", {"scene": "away"}
            else:
                text = fill(rng.choice(HI_SCENE if hinglish else SCENE_SAYINGS), scene=scene)
                intent, slots = "run_scene", {"scene": scene}
        elif kind == "manual":
            if rng.random() < 0.6:
                text = fill(rng.choice(MANUAL_CODE_SAYINGS), code=rng.choice(MANUAL_CODES))
            else:
                text = rng.choice(MANUAL_OTHER)
            intent, slots = "ask_manual", {}
        elif kind == "short":
            roll = rng.random()
            if roll < 0.25:
                text, intent, slots = fill(rng.choice(["the {room}", "{room}", "in the {room}"]), room=room), "other", {"room": room}
            elif roll < 0.45:
                text, intent, slots = rng.choice(["{temp}", "make it {temp}", "actually {temp}"]).replace("{temp}", str(temp)), "other", {"temp": temp}
            elif roll < 0.65:
                text, intent, slots = rng.choice(HI_YES if hinglish else YES_SAYINGS), "yes", {}
            elif roll < 0.82:
                text, intent, slots = rng.choice(HI_NO if hinglish else NO_SAYINGS), "no", {}
            else:
                text, intent, slots = rng.choice(HI_CANCEL if hinglish else CANCEL_SAYINGS), "cancel", {}
        else:
            text = rng.choice(HI_OUT if hinglish else OUT_OF_DOMAIN)
            intent, slots = "other", {}

        # One in four messages is written the way a speech recogniser might write it.
        if rng.random() < 0.25:
            text = make_noisy(text, rng)
        if text.lower() in golden_texts:
            continue
        key = text + "|" + answer_text(intent, slots)
        if key in seen and kind != "short":
            continue                         # no exact duplicates (short replies like "yes" naturally repeat)
        seen.add(key)
        examples.append(make_example(text, intent, slots))
    return examples


# ------------------------------------------------------------------------------------------------ replay data
# General questions with normal answers. They are mixed into training so the model does not forget how to chat.
REPLAY = [
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is two plus three?", "Two plus three is five."),
    ("Name the three states of matter.", "Solid, liquid and gas."),
    ("What do plants need to make food?", "Plants need sunlight, water and carbon dioxide."),
    ("How many days are there in a week?", "There are seven days in a week."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("Which planet is closest to the Sun?", "Mercury is the closest planet to the Sun."),
    ("What colour do you get by mixing blue and yellow?", "Blue and yellow make green."),
    ("How many minutes are there in an hour?", "There are sixty minutes in an hour."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("Write one sentence about the monsoon.", "The monsoon brings heavy rain to most parts of India from June to September."),
    ("What is the freezing point of water in Celsius?", "Water freezes at zero degrees Celsius."),
    ("Who wrote the national anthem of India?", "Rabindranath Tagore wrote the national anthem of India."),
    ("What is the plural of child?", "The plural of child is children."),
    ("How many sides does a triangle have?", "A triangle has three sides."),
    ("What do we call a baby cat?", "A baby cat is called a kitten."),
    ("Which gas do we breathe in to live?", "We breathe in oxygen."),
    ("What is ten times ten?", "Ten times ten is one hundred."),
    ("Name two primary colours.", "Red and blue are two primary colours."),
    ("What is the first month of the year?", "The first month of the year is January."),
]


def make_replay_examples():
    """Replay examples: a different system prompt and a plain-language answer (not JSON)."""
    examples = []
    for question, answer in REPLAY:
        examples.append({
            "text": question,
            "intent": "replay",
            "slots": {},
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        })
    return examples


# ------------------------------------------------------------------------------------------------ reading a model answer
def read_json(text):
    """Find the first JSON object in a reply. Returns a dictionary, or None when there is none or it is broken."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def read_intent(text):
    """The intent in a reply, or 'unreadable' when the reply is not valid JSON or the intent is not one we know."""
    obj = read_json(text)
    if obj is None:
        return "unreadable"
    intent = obj.get("intent")
    if intent in INTENTS:
        return intent
    return "unreadable"


def read_slots(text):
    """The slots in a reply, cleaned: values are lowercase text, and temp is a whole number."""
    obj = read_json(text)
    if obj is None:
        return {}
    raw_slots = obj.get("slots")
    if not isinstance(raw_slots, dict):
        return {}
    slots = {}
    for name in raw_slots:
        value = raw_slots[name]
        if name == "temp":
            try:
                slots[name] = int(float(value))
            except (TypeError, ValueError):
                slots[name] = value
        else:
            slots[name] = str(value).lower()
    return slots


def read_compact(text):
    """For the short answer format  intent|room|state|temp|scene  (a dash means 'not said'). Returns (intent, slots)."""
    line = text.strip().split("\n")[0]
    parts = line.split("|")
    while len(parts) < 5:
        parts.append("-")
    intent = parts[0].strip().lower()
    if intent not in INTENTS:
        return "unreadable", {}
    slots = {}
    names = ["room", "state", "temp", "scene"]
    for position in range(4):
        value = parts[position + 1].strip().lower()
        if value != "-" and value != "":
            if names[position] == "temp":
                try:
                    value = int(float(value))
                except ValueError:
                    pass
            slots[names[position]] = value
    return intent, slots


def compact_to_json(reply):
    """Turn a short-format answer (intent|room|state|temp|scene) into JSON text, so one scorer can mark every format."""
    intent, slots = read_compact(reply)
    if intent == "unreadable":
        return reply                         # leave it as it is: it will count as not valid
    return answer_text(intent, slots)


# ------------------------------------------------------------------------------------------------ the golden set
def load_golden():
    """The golden test cases: a list of dictionaries with text, intent, slots and kind. Never used for training."""
    with open(DATA / "edge_golden.json", encoding="utf-8") as f:
        return json.load(f)


def score_one(reply, case):
    """Score one reply against one golden case. Returns a dictionary of True/False results."""
    intent = read_intent(reply)
    slots = read_slots(reply)
    expected = case["slots"]
    slots_ok = True
    for name in expected:
        if slots.get(name) != expected[name]:
            slots_ok = False
    for name in slots:
        if name not in expected:
            slots_ok = False                 # an invented slot counts as wrong
    valid = read_json(reply) is not None
    return {
        "valid_json": valid,
        "intent_ok": intent == case["intent"],
        "slots_ok": slots_ok,
        "exact_ok": intent == case["intent"] and slots_ok,
    }


def score_all(replies, cases):
    """Score many replies. Returns percentages and the list of cases that were not exactly right."""
    counts = {"valid_json": 0, "intent_ok": 0, "slots_ok": 0, "exact_ok": 0}
    wrong = []
    for index in range(len(cases)):
        result = score_one(replies[index], cases[index])
        for name in counts:
            if result[name]:
                counts[name] = counts[name] + 1
        if not result["exact_ok"]:
            wrong.append({"text": cases[index]["text"], "expected": cases[index]["intent"], "got": read_intent(replies[index]),
                          "expected_slots": cases[index]["slots"], "got_slots": read_slots(replies[index]),
                          "reply": replies[index], "kind": cases[index]["kind"]})
    total = len(cases)
    summary = {}
    for name in counts:
        summary[name] = round(100 * counts[name] / total, 1)
    summary["wrong"] = wrong
    return summary


def describe_mistake(mistake):
    """One readable line about a wrong reply: the kind, the message, what was expected and what the model said."""
    return (mistake["kind"] + " | " + mistake["text"] + " | expected: " + mistake["expected"] + " " + str(mistake["expected_slots"])
            + " | got: " + mistake["got"] + " " + str(mistake["got_slots"]))


def accuracy_by_kind(replies, cases):
    """Exact accuracy for each kind of case (clean, noisy, hinglish, out_of_domain, ...). Returns a dictionary."""
    right = {}
    total = {}
    for index in range(len(cases)):
        kind = cases[index]["kind"]
        total[kind] = total.get(kind, 0) + 1
        if score_one(replies[index], cases[index])["exact_ok"]:
            right[kind] = right.get(kind, 0) + 1
    result = {}
    for kind in total:
        result[kind] = round(100 * right.get(kind, 0) / total[kind], 1)
    return result


# ------------------------------------------------------------------------------------------------ your own messages
OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"
MY_MESSAGES_FILE = OUTPUT_FOLDER / "my_messages.json"
STATES = ["on", "off", "open", "closed"]
SLOT_NAMES = ["room", "state", "temp", "scene"]


def prepare_my_messages(rows, kind="mine"):
    """Check the messages you wrote. Empty rows are skipped, problems are explained in plain words.

    Each row is a dictionary: {"text": "...", "intent": "...", "slots": {...}}. Returns the clean test cases.
    """
    cases = []
    for number in range(len(rows)):
        row = rows[number]
        text = str(row.get("text", "")).strip()
        if text == "":
            continue                                     # an empty row is simply skipped
        intent = row.get("intent", "")
        slots = row.get("slots", {})
        problem = ""
        if intent not in INTENTS:
            problem = "intent must be one of: " + ", ".join(INTENTS)
        elif not isinstance(slots, dict):
            problem = 'slots must be a dictionary such as {"room": "kitchen"}'
        else:
            for name in slots:
                value = slots[name]
                if name not in SLOT_NAMES:
                    problem = "the slot name '" + str(name) + "' is not allowed. Use: " + ", ".join(SLOT_NAMES)
                elif name == "room" and value not in ROOMS:
                    problem = "room must be one of: " + ", ".join(ROOMS)
                elif name == "state" and value not in STATES:
                    problem = "state must be one of: " + ", ".join(STATES)
                elif name == "scene" and value not in SCENES:
                    problem = "scene must be one of: " + ", ".join(SCENES)
                elif name == "temp" and not isinstance(value, int):
                    problem = "temp must be a whole number such as 22"
        if problem != "":
            print("row " + str(number + 1) + " skipped:", problem)
        else:
            cases.append({"text": text, "intent": intent, "slots": slots, "kind": kind})
    print(len(cases), "message(s) ready")
    return cases


def save_my_messages(cases):
    """Save your messages so that Lab 17 can use them."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    with open(MY_MESSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1, ensure_ascii=False)


def load_my_messages():
    """The messages you saved in Lab 16. An empty list if there are none yet."""
    if MY_MESSAGES_FILE.exists():
        with open(MY_MESSAGES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def show_results(cases, replies):
    """One line per message: OK or XX, and for a wrong one what was expected and what the model said."""
    for index in range(len(cases)):
        right = score_one(replies[index], cases[index])["exact_ok"]
        mark = "OK " if right else "XX "
        print(mark + cases[index]["text"])
        if not right:
            print("      expected:", cases[index]["intent"], cases[index]["slots"])
            print("      model   :", read_intent(replies[index]), read_slots(replies[index]))
