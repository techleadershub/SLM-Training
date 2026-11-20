"""Extra lab X3: how the TV assistant READS language. The model only reads; it never acts.

This file holds three things.
1. The training data: TV commands built from facts (an intent plus its slots), said in many ways, in English, in Hinglish and
   with speech-recognition mistakes. It is the same idea as edge_nlu.py of Lab 17, for a TV.
2. A keyword-rules reader: the baseline, and the reader you can run on a laptop with no GPU.
3. The model reader: load the small model (with its adapter), read one message, say how sure it is, and the helper
   train_tv_adapter() that trains the adapter with the SAME recipe as Lab 17.

Plain Python only. torch and transformers are imported inside the functions that need them, so the rules reader works anywhere.
"""
import json
import random
from pathlib import Path

import edge_nlu
import tv_domain as td

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT_FOLDER = HERE / "outputs"
OWN_ADAPTER = OUTPUT_FOLDER / "tv_adapter"            # the adapter the participant trains
PREPARED_ADAPTER = DATA / "tv_adapter_prepared"       # the adapter the trainer prepared (used when the own one is missing)

INTENTS = [
    "power", "set_volume", "volume_up", "volume_down", "mute", "set_channel", "launch_app", "set_input", "playback",
    "search_content", "same_again", "choose", "subtitles", "sleep_timer", "settings", "factory_reset", "delete_recordings",
    "rent_movie", "adult_content", "ask_time", "ask_status", "undo", "help", "yes", "no", "cancel", "out_of_scope", "other",
]
SLOT_NAMES = {
    "power": ["state"], "set_volume": ["level"], "volume_up": ["amount"], "volume_down": ["amount"], "mute": ["state"],
    "set_channel": ["channel"], "launch_app": ["app"], "set_input": ["input"], "playback": ["action"],
    "search_content": ["query", "app"], "same_again": ["app"], "choose": ["index"], "subtitles": ["state"],
    "sleep_timer": ["minutes", "state"], "settings": ["section"], "rent_movie": ["title", "mode"], "help": ["topic"],
    "other": ["number", "title"],
}
TV_SYSTEM = 'Read the message sent to a TV assistant. Reply with JSON only: {"intent": ..., "slots": {...}}.'
TOPICS = ["volume", "channel", "app", "input", "playback", "subtitles", "sleep"]


# ------------------------------------------------------------------------------------------------ numbers in words
UNIT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
              "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ONE_NEEDS_A_HINT = ["hdmi", "channel", "to", "number", "volume", "option", "at", "av", "input", "level", "by"]


def number_words(number):
    """25 becomes 'twenty five'. Works from 0 to 100."""
    if number == 100:
        return "one hundred"
    if number < 20:
        for word in UNIT_WORDS:
            if UNIT_WORDS[word] == number:
                return word
    tens_part = (number // 10) * 10
    rest = number % 10
    text = ""
    for word in TENS_WORDS:
        if TENS_WORDS[word] == tens_part:
            text = word
    if rest > 0:
        for word in UNIT_WORDS:
            if UNIT_WORDS[word] == rest:
                text = text + " " + word
    return text


def clean_text(text):
    """Lower case, punctuation turned into spaces (apostrophes dropped), extra spaces removed."""
    out = ""
    for letter in text.lower():
        if letter == "'":
            continue
        if letter.isalnum() or letter == " ":
            out = out + letter
        else:
            out = out + " "
    return " ".join(out.split())


def convert_number_words(text, every_one=False):
    """'volume twenty five' becomes 'volume 25'. A lone 'one' is changed only after a word such as hdmi or channel
    (so 'the second one' stays as it is), unless every_one is True (used for a spoken PIN)."""
    tokens = text.split()
    out = []
    index = 0
    while index < len(tokens):
        word = tokens[index]
        nxt = ""
        if index + 1 < len(tokens):
            nxt = tokens[index + 1]
        if word in TENS_WORDS and nxt in UNIT_WORDS and 1 <= UNIT_WORDS[nxt] <= 9:
            out.append(str(TENS_WORDS[word] + UNIT_WORDS[nxt]))
            index = index + 2
        elif word in UNIT_WORDS and nxt == "hundred":
            out.append(str(UNIT_WORDS[word] * 100))
            index = index + 2
        elif word in TENS_WORDS:
            out.append(str(TENS_WORDS[word]))
            index = index + 1
        elif word in UNIT_WORDS:
            if word == "one" and not every_one:
                before = ""
                if len(out) > 0:
                    before = out[-1]
                if before in ONE_NEEDS_A_HINT:
                    out.append("1")
                else:
                    out.append("one")
            else:
                out.append(str(UNIT_WORDS[word]))
            index = index + 1
        elif word == "hundred":
            out.append("100")
            index = index + 1
        else:
            out.append(word)
            index = index + 1
    return " ".join(out)


def parse_pin(text):
    """The digits in a spoken PIN: '1234', '1 2 3 4' and 'one two three four' all give '1234'. Empty text when there are none."""
    converted = convert_number_words(clean_text(text), every_one=True)
    digits = ""
    for letter in converted:
        if letter.isdigit():
            digits = digits + letter
    return digits


# ------------------------------------------------------------------------------------------------ cleaning what a reader returns
APP_ALIASES = {
    "netflix": "netflix", "netflicks": "netflix", "youtube": "youtube", "utube": "youtube", "yt": "youtube",
    "prime": "prime video", "primevideo": "prime video", "primevedio": "prime video", "amazonprime": "prime video",
    "amazonprimevideo": "prime video", "amazon": "prime video", "hotstar": "hotstar", "disneyhotstar": "hotstar",
    "disneyplushotstar": "hotstar",
}
INPUT_ALIASES = {"hdmi1": "hdmi 1", "hdmi2": "hdmi 2", "hdmione": "hdmi 1", "hdmitwo": "hdmi 2", "hdmito": "hdmi 2", "av": "av",
                 "avinput": "av", "hdmi": ""}
ACTION_ALIASES = {"play": "play", "resume": "play", "continue": "play", "pause": "pause", "stop": "pause", "next": "next",
                  "skip": "next"}


def squeeze(text):
    """Only letters and digits, lower case: 'Prime Video' becomes 'primevideo'."""
    out = ""
    for letter in str(text).lower():
        if letter.isalnum():
            out = out + letter
    return out


def to_int(value):
    """A whole number from a value, or None."""
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def clean_one_slot(name, value):
    """Make one slot value normal: an alias becomes the real name, a number becomes an int. Returns None for a value to drop."""
    if name in ["level", "amount", "minutes", "index", "number"]:
        return to_int(value)
    if name == "channel":
        number = to_int(value)
        if number is not None and str(value).strip().lstrip("-").isdigit():
            return number
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        text = " ".join(str(value).lower().split())
        if text == "":
            return None
        return text
    if name == "app":
        key = squeeze(value)
        if key == "":
            return None
        return APP_ALIASES.get(key, str(value).lower().strip())
    if name == "input":
        key = squeeze(value)
        if key in INPUT_ALIASES:
            if INPUT_ALIASES[key] == "":
                return None
            return INPUT_ALIASES[key]
        return str(value).lower().strip()
    if name == "action":
        return ACTION_ALIASES.get(str(value).lower().strip())
    if name == "state":
        text = str(value).lower().strip()
        if text in ["on", "off"]:
            return text
        return None
    if name == "mode":
        text = str(value).lower().strip()
        if text in ["rent", "buy"]:
            return text
        return None
    if name == "section":
        text = str(value).lower().strip()
        if text in td.SETTINGS_SECTIONS:
            return text
        return None
    text = " ".join(str(value).lower().split())
    if text == "":
        return None
    return text


def clean_nlu(intent, slots):
    """The model proposes, the code cleans. Unknown intents become 'other'; invented slots and empty values are dropped.

    Returns (intent, slots, dropped) where dropped is the list of slot names that were thrown away.
    """
    if intent not in INTENTS:
        return "other", {}, []
    allowed = SLOT_NAMES.get(intent, [])
    clean = {}
    dropped = []
    if not isinstance(slots, dict):
        return intent, {}, []
    for name in slots:
        if name not in allowed:
            dropped.append(name)
            continue
        value = clean_one_slot(name, slots[name])
        if value is None:
            dropped.append(name)
        else:
            clean[name] = value
    return intent, clean, dropped


# ------------------------------------------------------------------------------------------------ the rules reader (the baseline)
YES_WORDS = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "proceed", "do", "go", "please", "fine"]
NO_WORDS = ["no", "nope", "nah", "not"]
COMMAND_WORDS = ["volume", "channel", "netflix", "youtube", "hotstar", "prime", "hdmi", "av", "mute", "play", "pause", "subtitles",
                 "sleep", "settings", "tv", "louder", "quieter", "next", "meant", "number", "up", "down"]
OUT_OF_SCOPE_WORDS = ["joke", "weather", "pizza", "cab", "capital", "alarm", "recipe", "story", "remind", "taxi", "call", "light",
                      "lights", "fan", "ac", "who", "cook", "calculate", "plus", "score", "order", "message", "moon", "food"]
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}


def has_number(words):
    """The first whole number in a list of words, or None."""
    for word in words:
        if word.isdigit():
            return int(word)
    return None


def find_app(text):
    """The app named in a text, or ''. Spaces are ignored, so 'prime video' and 'primevideo' both work."""
    squeezed = squeeze(text)
    for alias in ["netflix", "youtube", "primevideo", "prime", "hotstar"]:
        if alias in squeezed:
            return APP_ALIASES[alias]
    return ""


def rules_confirmations(text, words):
    """yes, no, cancel, undo: short replies. Returns an answer dictionary or None."""
    if text in ["cancel", "cancel that", "never mind", "nevermind", "forget it", "leave it", "stop", "stop that", "abort"]:
        return {"intent": "cancel", "slots": {}}
    if text.startswith("undo") or text in ["go back", "revert", "revert that", "put it back", "take that back"]:
        return {"intent": "undo", "slots": {}}
    if text in ["forget about it", "forget that"]:
        return {"intent": "cancel", "slots": {}}
    has_command = False
    for word in words:
        if word in COMMAND_WORDS or word.isdigit():
            has_command = True
    if len(words) <= 4 and words[0] in YES_WORDS and not has_command and text != "go back":
        return {"intent": "yes", "slots": {}}
    if len(words) <= 3 and words[0] in NO_WORDS and not has_command:
        return {"intent": "no", "slots": {}}
    return None


def rules_topic(words):
    """Which topic a short message is about (for help and the menu). Returns a topic name or ''."""
    for word in words:
        if word in ["volume", "sound", "loud"]:
            return "volume"
        if word in ["channel", "channels"]:
            return "channel"
        if word in ["app", "apps", "netflix", "youtube", "hotstar", "prime"]:
            return "app"
        if word in ["input", "inputs", "hdmi"]:
            return "input"
        if word in ["play", "pause", "playback", "next"]:
            return "playback"
        if word in ["subtitle", "subtitles", "captions"]:
            return "subtitles"
        if word in ["sleep", "timer"]:
            return "sleep"
    return ""


def rules_info(text, words):
    """help, time, status. Returns an answer dictionary or None."""
    if text.startswith("help") or "you can do" in text or "what can you do" in text or "what can i say" in text or text.startswith("how do i") or "need help" in text:
        slots = {}
        topic = rules_topic(words)
        if topic != "":
            slots["topic"] = topic
        return {"intent": "help", "slots": slots}
    if "what time" in text or "the time" in text or text == "time please":
        return {"intent": "ask_time", "slots": {}}
    if "what is playing" in text or "what am i watching" in text or "what is on" in text or "which channel" in text:
        return {"intent": "ask_status", "slots": {}}
    if "what is the volume" in text or "which app is open" in text:
        return {"intent": "ask_status", "slots": {}}
    return None


def rules_risky(text, words):
    """factory reset, delete recordings, rent or buy, adult content."""
    if "factory" in words or text == "reset the tv" or text == "reset tv":
        return {"intent": "factory_reset", "slots": {}}
    for verb in ["delete", "erase", "remove", "clear"]:
        if verb in words and ("recording" in text or "recorded" in text):
            return {"intent": "delete_recordings", "slots": {}}
    if "adult" in words or "18" in words:
        return {"intent": "adult_content", "slots": {}}
    for verb in ["rent", "buy", "purchase"]:
        if verb in words:
            slots = {"mode": "rent"}
            if verb != "rent":
                slots["mode"] = "buy"
            for item in td.STORE:
                if item in text:
                    slots["title"] = item
            if "title" not in slots:
                rest = []
                for word in words:
                    if word not in td.FILLER_WORDS and word not in ["rent", "buy", "purchase", "movie", "the", "i", "can"]:
                        rest.append(word)
                if len(rest) > 0:
                    slots["title"] = " ".join(rest)
            return {"intent": "rent_movie", "slots": slots}
    return None


def rules_settings(text, words):
    """sleep timer, subtitles, settings."""
    if "sleep" in words or "timer" in words or (("off" in words or "switch" in words) and ("minutes" in words or "hour" in words)):
        slots = {}
        if "off" in words and ("sleep" in words or "timer" in words) and has_number(words) is None:
            slots["state"] = "off"
        elif "cancel" in words or "remove" in words or "stop" in words:
            slots["state"] = "off"
        elif "half" in words and "hour" in words:
            slots["minutes"] = 30
        elif "hour" in words:
            slots["minutes"] = 60
        elif has_number(words) is not None:
            slots["minutes"] = has_number(words)
        return {"intent": "sleep_timer", "slots": slots}
    if "subtitle" in words or "subtitles" in words or "captions" in words:
        state = "on"
        for word in ["off", "hide", "disable", "remove", "no"]:
            if word in words:
                state = "off"
        return {"intent": "subtitles", "slots": {"state": state}}
    if "settings" in words or "setting" in words:
        slots = {}
        for section in td.SETTINGS_SECTIONS:
            if section in words:
                slots["section"] = section
        return {"intent": "settings", "slots": slots}
    return None


def rules_sound(text, words):
    """volume up, down, set, mute."""
    if "unmute" in words or "sound on" in text or "sound back" in text:
        return {"intent": "mute", "slots": {"state": "off"}}
    if "mute" in words or "sound off" in text or "silence" in words:
        return {"intent": "mute", "slots": {"state": "on"}}
    number = has_number(words)
    up = ["louder", "up", "increase", "raise"]
    down = ["quieter", "down", "decrease", "lower", "reduce", "softer"]
    context = ["volume", "louder", "quieter", "softer", "it", "loud", "sound", "lower", "raise", "increase", "decrease", "reduce"]
    found_context = False
    for word in context:
        if word in words:
            found_context = True
    if found_context:
        for word in down:
            if word in words:
                return volume_answer("volume_down", number)
        for word in up:
            if word in words:
                return volume_answer("volume_up", number)
        if "loud" in words:
            return volume_answer("volume_down", number)
        if "volume" in words:
            slots = {}
            if number is not None:
                slots["level"] = number
            return {"intent": "set_volume", "slots": slots}
    return None


def volume_answer(intent, number):
    """The volume_up or volume_down answer, with the amount when there is one."""
    slots = {}
    if number is not None:
        slots["amount"] = number
    return {"intent": intent, "slots": slots}


def has_command_word(words):
    """True when a list of words contains a word that names a TV command."""
    for word in words:
        if word in COMMAND_WORDS:
            return True
    return False


def rules_media(text, words):
    """same on another app, choose from a list, input, app, channel, play, search."""
    app = find_app(text)
    if app != "" and ("same" in words or text.startswith("play it on") or text.startswith("try that on") or text.startswith("watch it on")):
        return {"intent": "same_again", "slots": {"app": app}}
    for word in ORDINALS:
        if word in words and len(words) <= 6 and not has_command_word(words):
            return {"intent": "choose", "slots": {"index": ORDINALS[word]}}
    if ("number" in words or "option" in words) and has_number(words) is not None and len(words) <= 3:
        return {"intent": "choose", "slots": {"index": has_number(words)}}
    if "hdmi" in words or "av" in words or "input" in words:
        return rules_input(words)
    return None


def rules_input(words):
    """The input the user named."""
    number = has_number(words)
    if "hdmi" in words and number in [1, 2]:
        return {"intent": "set_input", "slots": {"input": "hdmi " + str(number)}}
    if "av" in words:
        return {"intent": "set_input", "slots": {"input": "av"}}
    return {"intent": "set_input", "slots": {}}


def rules_watch(text, words):
    """app, channel, playback, search, power. Returns an answer dictionary or None."""
    app = find_app(text)
    if "app" in words and app == "" and words[0] in ["open", "launch", "start"]:
        return {"intent": "launch_app", "slots": {}}
    if "channel" in words or "channels" in words:
        return rules_channel(text, words)
    found_name = channel_name_in(text)
    if found_name != "":
        return {"intent": "set_channel", "slots": {"channel": found_name}}
    if app != "":
        rest = []
        for word in words:
            if word not in td.FILLER_WORDS and word not in ["open", "launch", "start", "go", "app", "netflix", "youtube", "hotstar", "prime", "video", "amazon", "turn", "switch", "run", "take", "on", "in", "up"]:
                rest.append(word)
        if len(rest) > 0 and ("play" in words or "watch" in words):
            return {"intent": "search_content", "slots": {"query": " ".join(rest), "app": app}}
        return {"intent": "launch_app", "slots": {"app": app}}
    return rules_play(text, words)


def channel_name_in(text):
    """A channel name said in a text ('star sports' fits 'Star Sports 1'), or ''."""
    for number in td.CHANNELS:
        name = td.CHANNELS[number].lower()
        short = name.rstrip("0123456789").strip()
        if name in text or short in text:
            return short
    return ""


def rules_channel(text, words):
    """The channel the user named: a number, a name or a kind ('sports')."""
    if "next" in words or ("channel" in words and "up" in words):
        return {"intent": "playback", "slots": {"action": "next"}}
    number = has_number(words)
    if number is not None:
        return {"intent": "set_channel", "slots": {"channel": number}}
    found_name = channel_name_in(text)
    if found_name != "":
        return {"intent": "set_channel", "slots": {"channel": found_name}}
    for index in range(len(words)):
        if words[index] in ["channel", "channels"] and index > 0:
            before = words[index - 1]
            if before in ["movie", "film"]:
                before = "movies"
            if before not in ["the", "a", "any", "different", "this", "that", "my", "another", "new"]:
                return {"intent": "set_channel", "slots": {"channel": before}}
    return {"intent": "set_channel", "slots": {}}


def rules_play(text, words):
    """pause, play, next, search, then power. Returns an answer dictionary or None."""
    if words[0] in ["pause", "hold"] or "pause" in words:
        return {"intent": "playback", "slots": {"action": "pause"}}
    if text in ["play", "resume", "continue", "play it", "resume playing", "carry on"]:
        return {"intent": "playback", "slots": {"action": "play"}}
    if text in ["next", "next episode", "skip", "next one", "play the next one", "skip this"] or ("next" in words and "episode" in words):
        return {"intent": "playback", "slots": {"action": "next"}}
    if words[0] in ["play", "watch", "find", "show", "search"] or "want to watch" in text:
        rest = []
        for word in words:
            if word not in td.FILLER_WORDS and word not in ["watch", "search", "find", "show", "play", "want", "to"]:
                rest.append(word)
        slots = {}
        if len(rest) > 0:
            slots["query"] = " ".join(rest)
        return {"intent": "search_content", "slots": slots}
    return rules_power(text, words)


def rules_power(text, words):
    """Power on or off, with 'tv', or 'turn it off'."""
    if "on" in words or "off" in words:
        state = "on"
        if "off" in words:
            state = "off"
        if "tv" in words or "television" in words or "power" in words or "it" in words or len(words) <= 3:
            return {"intent": "power", "slots": {"state": state}}
    if "shut" in words and "down" in words:
        return {"intent": "power", "slots": {"state": "off"}}
    return None


def rules_read(text):
    """Read one message with keyword rules. Returns {'intent', 'slots', 'confidence', 'raw'}. Same shape as the model reader."""
    text = convert_number_words(clean_text(text))
    words = text.split()
    answer = rules_answer(text, words)
    clean_intent, clean_slots, dropped = clean_nlu(answer["intent"], answer["slots"])
    return {"intent": clean_intent, "slots": clean_slots, "dropped": dropped, "confidence": 1.0, "raw": json.dumps(answer),
            "valid_json": True}


def rules_answer(text, words):
    """The keyword rules, in order. The first rule that fits decides."""
    if len(words) == 0:
        return {"intent": "other", "slots": {}}
    for rule in [rules_confirmations, rules_info, rules_risky, rules_settings, rules_sound, rules_media, rules_watch]:
        found = rule(text, words)
        if found is not None:
            return found
    for word in OUT_OF_SCOPE_WORDS:
        if word in words:
            return {"intent": "out_of_scope", "slots": {}}
    number = has_number(words)
    if number is not None and len(words) <= 4:
        return {"intent": "other", "slots": {"number": number}}
    for item in td.CATALOG:
        if item["title"] == text:
            return {"intent": "other", "slots": {"title": text}}
    for item in td.STORE:
        if item == text:
            return {"intent": "other", "slots": {"title": text}}
    return {"intent": "other", "slots": {}}


# ------------------------------------------------------------------------------------------------ the model reader
def read_model_answer(raw):
    """Turn the model's text into the same dictionary the rules reader returns."""
    obj = edge_nlu.read_json(raw)
    if obj is None:
        return {"intent": "other", "slots": {}, "dropped": [], "valid_json": False, "raw": raw}
    clean_intent, clean_slots, dropped = clean_nlu(str(obj.get("intent")), obj.get("slots"))
    return {"intent": clean_intent, "slots": clean_slots, "dropped": dropped, "valid_json": True, "raw": raw}


def find_adapter():
    """The adapter folder to use: your own (outputs/tv_adapter) if it exists, else the prepared one, else None."""
    if (OWN_ADAPTER / "adapter_config.json").exists():
        return str(OWN_ADAPTER)
    if (PREPARED_ADAPTER / "adapter_config.json").exists():
        return str(PREPARED_ADAPTER)
    return None


def load_reader(adapter=None):
    """Load the small model with the TV adapter. Returns a reader dictionary. Needs a GPU."""
    import edge_lab
    if adapter is None:
        adapter = find_adapter()
    if adapter is None:
        raise RuntimeError("No TV adapter found. Run train_tv_adapter() first, or ask the trainer for data/tv_adapter_prepared.")
    tok, model = edge_lab.load_edge_model(adapter=adapter)
    return {"mode": "model", "tok": tok, "model": model, "adapter": adapter, "calls": 0}


def read_with_model(reader, text, with_confidence=True):
    """One message through the model. The same dictionary as rules_read, with the model's own confidence."""
    import edge_lab
    prompt = edge_lab.make_prompt_text(reader["tok"], TV_SYSTEM, text)
    result = edge_lab.ask_many(reader["tok"], reader["model"], [prompt], max_new_tokens=60, batch_size=1)
    reader["calls"] = reader["calls"] + 1
    answer = read_model_answer(result["replies"][0])
    answer["confidence"] = None
    if with_confidence:
        probabilities = label_probabilities(reader["tok"], reader["model"], text)
        answer["confidence"] = round(probabilities.get(answer["intent"], 0.0), 3)
    return answer


def label_probabilities(tok, model, text):
    """How likely is each of the intents, as a dictionary (they add up to 1). The idea of Lab 20, for the TV intents."""
    import math
    import torch
    import edge_lab
    prompt = edge_lab.make_prompt_text(tok, TV_SYSTEM, text) + '{"intent": "'
    prefix_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    label_ids_list = []
    sequences = []
    for label in INTENTS:
        label_ids = tok(label, add_special_tokens=False)["input_ids"]
        label_ids_list.append(label_ids)
        sequences.append(prefix_ids + label_ids)
    longest = 0
    for sequence in sequences:
        if len(sequence) > longest:
            longest = len(sequence)
    input_ids = []
    attention = []
    for sequence in sequences:
        padding = longest - len(sequence)
        input_ids.append(sequence + [tok.pad_token_id] * padding)
        attention.append([1] * len(sequence) + [0] * padding)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(input_ids).to(model.device), attention_mask=torch.tensor(attention).to(model.device)).logits
    scores = []
    for index in range(len(label_ids_list)):
        total = 0.0
        for position in range(len(label_ids_list[index])):
            row = logits[index, len(prefix_ids) + position - 1].float()
            total = total + float(torch.log_softmax(row, dim=-1)[label_ids_list[index][position]])
        scores.append(total)
    biggest = max(scores)
    weights = []
    for score in scores:
        weights.append(math.exp(score - biggest))
    weight_sum = sum(weights)
    probabilities = {}
    for index in range(len(INTENTS)):
        probabilities[INTENTS[index]] = weights[index] / weight_sum
    return probabilities


def read_text(reader, text):
    """The one door: read a message with whatever reader this is (rules or model)."""
    if reader["mode"] == "model":
        return read_with_model(reader, text, reader.get("confidence", True))
    return rules_read(text)


def normalise_text(text):
    """What the reader is given: lower case, without a final full stop or question mark (speech recognisers add them)."""
    out = str(text).strip().lower()
    while out != "" and out[-1] in ".?!;:":
        out = out[:-1].rstrip()
    return out


def make_rules_reader():
    """The keyword-rules reader (no model, no GPU)."""
    return {"mode": "rules", "calls": 0}


# ------------------------------------------------------------------------------------------------ the golden set
def load_golden():
    """The TV golden test cases: text, intent, slots, kind. Never used for training."""
    with open(DATA / "tv_golden.json", encoding="utf-8") as f:
        return json.load(f)


def score_case(answer, case):
    """Does one reader answer match one golden case exactly (intent and every slot)? Returns True or False."""
    if answer["intent"] != case["intent"]:
        return False
    if len(answer["slots"]) != len(case["slots"]):
        return False
    for name in case["slots"]:
        if answer["slots"].get(name) != case["slots"][name]:
            return False
    return True


def score_reader(reader, cases):
    """Run a reader over test cases. Returns a dictionary: exact (percent), by_kind (percent per kind), wrong (list)."""
    right = 0
    kinds_total = {}
    kinds_right = {}
    wrong = []
    for case in cases:
        answer = read_text(reader, case["text"])
        kind = case["kind"]
        kinds_total[kind] = kinds_total.get(kind, 0) + 1
        if score_case(answer, case):
            right = right + 1
            kinds_right[kind] = kinds_right.get(kind, 0) + 1
        else:
            wrong.append({"text": case["text"], "kind": kind, "expected": case["intent"], "expected_slots": case["slots"],
                          "got": answer["intent"], "got_slots": answer["slots"]})
    by_kind = {}
    for kind in kinds_total:
        by_kind[kind] = round(100 * kinds_right.get(kind, 0) / kinds_total[kind], 1)
    return {"exact": round(100 * right / len(cases), 1), "by_kind": by_kind, "wrong": wrong}


def describe_mistake(mistake):
    """One readable line about a wrong answer."""
    return (mistake["kind"] + " | " + mistake["text"] + " | expected: " + mistake["expected"] + " " + str(mistake["expected_slots"])
            + " | got: " + mistake["got"] + " " + str(mistake["got_slots"]))


# ------------------------------------------------------------------------------------------------ the training sentences
LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]
CHANNEL_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 35, 52, 101, 121, 205]
MINUTES = [10, 15, 20, 30, 45, 60, 90, 120]
AMOUNTS = [2, 5, 10, 15, 20]
GENRES = ["funny", "comedy", "thriller", "kids", "cricket", "documentary", "drama"]
TITLES = []
for _item in td.CATALOG:
    TITLES.append(_item["title"])
STORE_TITLES = list(td.STORE)
OTHER_TITLES = ["golden hour", "desert wind", "city lights", "the last train"]
CHANNEL_NAMES = []
for _number in td.CHANNELS:
    CHANNEL_NAMES.append(td.CHANNELS[_number].lower())
CHANNEL_KINDS = ["sports", "news", "movies", "kids", "music"]
ORDINAL_WORDS = ["first", "second", "third"]
SECTIONS = td.SETTINGS_SECTIONS

POWER_ON = ["turn on tv", "tv on now", "switch the tv on please", "start up the tv", "turn on the tv", "switch on the tv", "tv on please", "power on", "please turn the tv on", "switch the television on",
            "turn on the television", "start the tv", "tv on", "turn the tv on"]
POWER_OFF = ["turn off tv", "tv off now", "switch the tv off please", "close the tv", "turn off the tv", "switch off the tv", "tv off", "power off", "please turn the tv off", "switch the television off",
             "shut down the tv", "turn it off", "turn the tv off", "switch it off"]
SET_VOLUME = ["set the volume to {n}", "volume {n}", "make the volume {n}", "put the volume at {n}", "volume to {n} please",
              "set volume {n}", "change the volume to {n}", "I want the volume at {n}", "volume level {n}", "set the volume at {n}"]
SET_VOLUME_ASK = ["set the volume", "change the volume", "I want to change the volume", "adjust the volume", "volume please"]
VOLUME_UP = ["turn it up", "louder please", "volume up", "turn the volume up", "increase the volume", "a bit louder",
             "raise the volume", "make it louder", "up the volume", "turn up the volume", "it is too quiet", "I cannot hear"]
VOLUME_UP_BY = ["turn it up by {n}", "volume up by {n}", "increase the volume by {n}", "raise the volume by {n}", "louder by {n}"]
VOLUME_DOWN = ["turn it down", "quieter please", "volume down", "turn the volume down", "decrease the volume", "lower the volume",
               "make it quieter", "reduce the volume", "it is too loud", "bring the volume down", "a bit quieter", "turn down the volume"]
VOLUME_DOWN_BY = ["turn it down by {n}", "volume down by {n}", "decrease the volume by {n}", "lower the volume by {n}", "quieter by {n}"]
MUTE_ON = ["mute", "mute it", "mute the tv", "mute the sound", "silence the tv", "sound off", "mute please", "please mute"]
MUTE_OFF = ["unmute", "unmute it", "sound on", "unmute the tv", "bring the sound back", "unmute please"]
CHANNEL_NUM = ["go to channel {n}", "channel {n}", "switch to channel {n}", "put on channel {n}", "change the channel to {n}",
               "take me to channel {n}", "tune to channel {n}", "channel number {n}", "no, channel {n}", "I meant channel {n}",
               "not that, channel {n}", "sorry, channel {n}", "go to {n}"]
CHANNEL_NAME = ["{name}", "put on {name}", "go to {name}", "switch to {name}", "show me {name}", "{name} please", "change to {name}", "I want {name}"]
CHANNEL_KIND = ["put on a {name} channel", "show me the {name} channel", "any {name} channel", "go to the {name} channel", "{name} channel please"]
CHANNEL_ASK = ["change the channel", "go to a different channel", "switch the channel", "I want to change the channel"]
APP_OPEN = ["{app}", "open {app}", "launch {app}", "start {app}", "go to {app}", "{app} please", "switch to {app}", "open the {app} app",
            "take me to {app}", "put on {app}", "run {app}"]
APP_ASK = ["open an app", "launch an app", "I want to open an app", "start an app"]
INPUT_SWITCH = ["{input}", "switch to {input}", "change the input to {input}", "go to {input}", "use {input}", "{input} please", "select {input}",
                "input {input}", "change to {input}", "put it on {input}"]
INPUT_ASK = ["change the input", "switch the input", "I want to change the input", "change the source"]
PAUSE = ["pause", "pause it", "pause the movie", "hold on, pause", "pause the video", "please pause", "pause this"]
PLAY = ["play", "resume", "continue", "play it", "resume playing", "carry on", "please play", "unpause"]
NEXT = ["next", "next episode", "skip", "play the next one", "next one", "skip this", "next please"]
SEARCH_GENRE = ["play something {genre}", "I want to watch something {genre}", "find a {genre} show", "show me {genre} movies",
                "put on a {genre} movie", "play a {genre} show", "something {genre} please", "I feel like watching something {genre}"]
SEARCH_TITLE = ["play {title}", "watch {title}", "I want to watch {title}", "search for {title}", "find {title}", "put on {title}",
                "can you play {title}"]
SEARCH_APP = ["play {title} on {app}", "watch {title} on {app}", "{title} on {app}", "play something {genre} on {app}",
              "find {genre} shows on {app}"]
SAME_AGAIN = ["the same on {app}", "same thing on {app}", "try the same on {app}", "play it on {app}", "try that on {app}",
              "same on {app}", "do the same on {app}", "try {app} instead"]
CHOOSE = ["the {ord} one", "{ord} one", "number {n}", "option {n}", "the {ord}", "{ord} please", "I will take the {ord} one",
          "choose {n}", "go with the {ord}", "the {ord} one please", "{n}rd one"]
SUBTITLES_ON = ["subtitles on", "turn on the subtitles", "show subtitles", "enable subtitles", "I need subtitles", "captions on",
                "switch on subtitles", "put the subtitles on"]
SUBTITLES_OFF = ["subtitles off", "turn off the subtitles", "hide subtitles", "disable subtitles", "no subtitles please", "captions off",
                 "switch off subtitles", "remove the subtitles"]
SLEEP_SET = ["set a sleep timer for {n} minutes", "sleep timer {n} minutes", "turn off the tv in {n} minutes",
             "switch off after {n} minutes", "sleep in {n} minutes", "set the timer for {n} minutes", "put a {n} minute sleep timer",
             "turn the tv off in {n} minutes"]
SLEEP_OFF = ["the sleep timer off please", "sleep timer please off", "switch off the sleep timer", "cancel the sleep timer", "sleep timer off", "remove the timer", "stop the sleep timer", "no sleep timer"]
SLEEP_ASK = ["set a sleep timer", "sleep timer please", "I want a sleep timer", "put on a sleep timer"]
SETTINGS = ["open settings", "open the settings", "go to settings", "settings please", "open {section} settings",
            "show me the {section} settings", "I want to change the {section} settings", "{section} settings please"]
FACTORY = ["factory reset", "reset the tv to factory settings", "do a factory reset", "erase everything and start again",
           "reset the tv", "restore the factory settings", "reset everything on the tv"]
DELETE = ["delete all recordings", "delete my recordings", "erase the recordings", "remove all recordings", "clear my recordings",
          "delete the recorded shows", "wipe all recordings"]
RENT = ["rent {title}", "I want to rent {title}", "rent the movie {title}", "can I rent {title}", "please rent {title}"]
BUY = ["buy {title}", "I want to buy {title}", "buy the movie {title}", "can I buy {title}", "purchase {title}"]
RENT_ASK = ["I want to rent a movie", "rent a movie", "rent something"]
BUY_ASK = ["I want to buy a movie", "buy a movie"]
ADULT = ["show adult content", "open the adult zone", "adult channels please", "I want to watch adult content", "play adult movies",
         "adult zone please", "switch to the adult channel"]
ASK_TIME = ["what time is it", "what is the time", "tell me the time", "time please", "what is the time now", "do you know the time",
            "what time is it now"]
ASK_STATUS = ["what is playing", "what am I watching", "what is the volume", "which channel is this", "what is on now",
              "tell me what is on", "what is the current volume", "which app is open"]
UNDO = ["undo", "undo that", "undo it", "go back", "revert that", "put it back", "that was a mistake, undo", "take that back"]
HELP = ["help", "what can you do", "how do I use you", "I need help", "what can I say", "help me", "can you help me"]
HELP_TOPIC = ["how do I change the {topic}", "help with the {topic}", "I need help with {topic}", "what can I say about {topic}"]
YES_SAYINGS = ["yes", "yeah", "yep", "sure", "okay", "go ahead", "yes please", "sure, do it", "that is fine, go ahead", "yes, do it"]
NO_SAYINGS = ["no, I changed my mind about it", "no I have changed my mind", "no",  "nope", "not really", "no thanks", "I do not want that", "no, do not", "no stop"]
CANCEL_SAYINGS = ["cancel", "cancel that", "never mind", "stop", "forget it", "leave it", "stop that"]
OUT_OF_SCOPE = ["tell me a joke", "what is the weather today", "order a pizza", "book me a cab", "who won the cricket match",
                "set an alarm for six", "call my mother", "what is the capital of France", "how do I cook rice",
                "remind me to buy milk", "tell me a story", "what is two plus two", "turn on the kitchen light", "switch on the ac",
                "how far is the moon", "what is the weather like in Delhi"]
NOT_UNDERSTOOD = ["blah blah", "hmm", "uh", "asdf", "the thing", "um okay", "what", "huh", "more blah", "nothing", "that", "maybe",
                  "perhaps", "I do not know", "not sure", "hmm maybe"]
BARE_NUMBER = ["{n}", "{n} minutes", "make it {n}", "actually {n}", "{n} please", "I meant {n}", "no, {n}", "set it to {n}", "no, I said {n}"]

# The same ideas in Hinglish (Hindi words in English letters, mixed with English).
HI_POWER_ON = ["tv ko chalu karo", "tv chalu kar dijiye", "tv on kar dena", "please tv chalu karo", "tv chalu karo", "tv on kar do", "tv chalao", "tv on karo"]
HI_POWER_OFF = ["tv ko band karo", "tv band kar dijiye", "tv off kar dena", "please tv band karo", "tv band karo", "tv off kar do", "tv band kar do", "tv off karo"]
HI_VOLUME_SET = ["volume {n} kar do", "awaaz {n} kar do", "volume {n} pe set karo", "awaaz {n} par rakho"]
HI_VOLUME_UP = ["volume badhao", "awaaz badhao", "awaaz thodi badhao", "volume zyada karo", "awaaz tez karo"]
HI_VOLUME_DOWN = ["volume kam karo", "awaaz kam karo", "awaaz dheere karo", "volume ghatao", "awaaz thodi kam karo"]
HI_MUTE_ON = ["mute kar do", "awaaz band karo", "mute karo"]
HI_MUTE_OFF = ["unmute karo", "awaaz chalu karo", "unmute kar do"]
HI_CHANNEL_NUM = ["channel {n} lagao", "{n} number channel lagao", "channel badlo, {n} par", "channel {n} pe jao"]
HI_CHANNEL_NAME = ["{name} lagao", "{name} chalao", "{name} pe jao"]
HI_APP = ["{app} chalao", "{app} kholo", "{app} lagao", "{app} on karo"]
HI_INPUT = ["{input} pe lagao", "{input} par switch karo", "{input} lagao", "input {input} karo"]
HI_PAUSE = ["pause karo", "ruko, pause karo", "ek minute, pause karo"]
HI_PLAY = ["play karo", "chalu karo", "phir se chalao"]
HI_NEXT = ["agla episode lagao", "next karo", "aage badhao"]
HI_SEARCH = ["koi {genre} dikhao", "kuch {genre} lagao", "{title} lagao", "{title} dikhao", "{title} chalao"]
HI_SUBTITLES_ON = ["subtitles on karo", "subtitle chalu karo", "subtitles lagao"]
HI_SUBTITLES_OFF = ["subtitles band karo", "subtitle hata do", "subtitles off karo"]
HI_SLEEP = ["{n} minute baad tv band kar do", "sleep timer {n} minute ka lagao", "{n} minute ka timer lagao"]
HI_HELP = ["madad karo", "help chahiye", "kya kya kar sakte ho"]
HI_YES = ["haan", "haan kar do", "theek hai", "bilkul", "ji haan"]
HI_NO = ["nahi", "nahi chahiye", "mat karo", "nahin"]
HI_CANCEL = ["rehne do", "chhod do", "cancel kar do", "koi baat nahi"]
HI_TIME = ["time kya hua hai", "kitne baje hain", "abhi time kya hai"]
HI_UNDO = ["wapas karo", "pehle jaisa kar do", "undo karo"]
HI_STATUS = ["kya chal raha hai", "volume kitna hai"]
HI_CHOOSE = ["doosra wala", "pehla wala", "teesra wala", "number {n} wala"]
HI_FACTORY = ["factory reset kar do", "tv reset kar do"]
HI_DELETE = ["saari recordings delete kar do", "recordings hata do"]
HI_RENT = ["{title} rent karo", "{title} rent par lo"]
HI_BUY = ["{title} kharido", "{title} buy karo"]
HI_OUT = ["ek joke sunao", "mausam kaisa hai", "pizza order karo", "cab book karo", "kal ka mausam batao"]
HI_ADULT = ["adult content dikhao", "adult zone kholo"]

# What a speech recogniser gets wrong. Each pair is (what was said, what the recogniser may write instead).
NOISE_PAIRS = [
    ("volume", "volum"), ("volume", "wolume"), ("channel", "chanel"), ("channel", "channal"), ("netflix", "net flix"),
    ("netflix", "netflicks"), ("youtube", "you tube"), ("hotstar", "hot star"), ("prime video", "prime vedio"), ("hdmi 1", "hdmi one"),
    ("hdmi 2", "hdmi to"), ("hdmi", "h d m i"), ("mute", "moot"), ("subtitles", "sub titles"), ("pause", "paws"),
    ("turn on", "tern on"), ("switch", "swich"), ("settings", "setting"), ("recordings", "recording"), ("sleep timer", "sleep time"),
    ("minutes", "minits"), ("factory", "factri"), ("louder", "lauder"), ("the tv", "the tee vee"), ("turn it up", "turn it app"),
    ("quieter", "quiet er"), ("adult", "a dult"),
]
GENRE_HINDI = ["funny", "comedy", "thriller", "kids", "drama"]


def make_noisy(text, rng):
    """Mimic speech-recognition mistakes: each known mistake is applied half of the time."""
    noisy = text.lower()
    for correct, wrong in NOISE_PAIRS:
        if correct in noisy and rng.random() < 0.5:
            noisy = noisy.replace(correct, wrong)
    return noisy


def fill_text(template, values):
    """Put values into {name} places of a sentence template. values is a dictionary of name -> text."""
    text = template
    for name in values:
        text = text.replace("{" + name + "}", str(values[name]))
    return text


def spoken_number(number, rng, words_share=0.25):
    """The number as digits, or (a quarter of the time, if it is small) as words, the way speech recognition may write it."""
    if number <= 100 and rng.random() < words_share:
        return number_words(number)
    return str(number)


def pick(rng, templates, values, intent, slots):
    """Fill one random template. Returns (text, intent, slots)."""
    return fill_text(rng.choice(templates), values), intent, slots


def make_kind_example(kind, rng, hinglish):
    """One (text, intent, slots) for one kind of example. Hinglish is True for about a quarter of the examples."""
    n_level = rng.choice(LEVELS)
    n_chan = rng.choice(CHANNEL_NUMBERS)
    n_min = rng.choice(MINUTES)
    n_amount = rng.choice(AMOUNTS)
    app = rng.choice(td.APPS)
    inp = rng.choice(td.INPUTS)
    genre = rng.choice(GENRES)
    title = rng.choice(TITLES + OTHER_TITLES)
    name = rng.choice(CHANNEL_NAMES)
    if kind == "power":
        state = rng.choice(["on", "off"])
        if hinglish:
            return pick(rng, HI_POWER_ON if state == "on" else HI_POWER_OFF, {}, "power", {"state": state})
        return pick(rng, POWER_ON if state == "on" else POWER_OFF, {}, "power", {"state": state})
    if kind == "set_volume":
        if rng.random() < 0.15:
            return pick(rng, SET_VOLUME_ASK, {}, "set_volume", {})
        if hinglish:
            return pick(rng, HI_VOLUME_SET, {"n": n_level}, "set_volume", {"level": n_level})
        return pick(rng, SET_VOLUME, {"n": spoken_number(n_level, rng)}, "set_volume", {"level": n_level})
    if kind == "volume_up" or kind == "volume_down":
        up = kind == "volume_up"
        if hinglish:
            return pick(rng, HI_VOLUME_UP if up else HI_VOLUME_DOWN, {}, kind, {})
        if rng.random() < 0.3:
            return pick(rng, VOLUME_UP_BY if up else VOLUME_DOWN_BY, {"n": spoken_number(n_amount, rng)}, kind, {"amount": n_amount})
        return pick(rng, VOLUME_UP if up else VOLUME_DOWN, {}, kind, {})
    if kind == "mute":
        state = rng.choice(["on", "off"])
        if hinglish:
            return pick(rng, HI_MUTE_ON if state == "on" else HI_MUTE_OFF, {}, "mute", {"state": state})
        return pick(rng, MUTE_ON if state == "on" else MUTE_OFF, {}, "mute", {"state": state})
    if kind == "channel":
        return make_channel_example(rng, hinglish, n_chan, name)
    if kind == "app":
        if rng.random() < 0.1 and not hinglish:
            return pick(rng, APP_ASK, {}, "launch_app", {})
        return pick(rng, HI_APP if hinglish else APP_OPEN, {"app": noisy_app_name(app, rng)}, "launch_app", {"app": app})
    if kind == "input":
        if rng.random() < 0.1 and not hinglish:
            return pick(rng, INPUT_ASK, {}, "set_input", {})
        return pick(rng, HI_INPUT if hinglish else INPUT_SWITCH, {"input": inp}, "set_input", {"input": inp})
    if kind == "playback":
        action = rng.choice(["pause", "play", "next"])
        lists = {"pause": [PAUSE, HI_PAUSE], "play": [PLAY, HI_PLAY], "next": [NEXT, HI_NEXT]}
        if hinglish:
            return pick(rng, lists[action][1], {}, "playback", {"action": action})
        return pick(rng, lists[action][0], {}, "playback", {"action": action})
    if kind == "search":
        return make_search_example(rng, hinglish, genre, title, app)
    return make_more_example(kind, rng, hinglish, n_level, n_min, app, title)


def noisy_app_name(app, rng):
    """The app name as it may be said: 'prime video' is often just 'prime'."""
    if app == "prime video" and rng.random() < 0.3:
        return "prime"
    return app


def make_channel_example(rng, hinglish, number, name):
    """A channel example: by number, by name, by kind ('sports') or with no value."""
    roll = rng.random()
    if roll < 0.5:
        if hinglish:
            return pick(rng, HI_CHANNEL_NUM, {"n": number}, "set_channel", {"channel": number})
        return pick(rng, CHANNEL_NUM, {"n": spoken_number(number, rng)}, "set_channel", {"channel": number})
    if roll < 0.8:
        if hinglish:
            return pick(rng, HI_CHANNEL_NAME, {"name": name}, "set_channel", {"channel": name})
        return pick(rng, CHANNEL_NAME, {"name": name}, "set_channel", {"channel": name})
    if roll < 0.93 and not hinglish:
        kind_word = rng.choice(CHANNEL_KINDS)
        return pick(rng, CHANNEL_KIND, {"name": kind_word}, "set_channel", {"channel": kind_word})
    return pick(rng, CHANNEL_ASK, {}, "set_channel", {})


def make_search_example(rng, hinglish, genre, title, app):
    """Search for something to watch: a kind of show, a title, with or without an app."""
    roll = rng.random()
    if hinglish:
        if roll < 0.5:
            return pick(rng, HI_SEARCH[:2], {"genre": genre}, "search_content", {"query": genre})
        return pick(rng, HI_SEARCH[2:], {"title": title}, "search_content", {"query": title})
    if roll < 0.35:
        return pick(rng, SEARCH_GENRE, {"genre": genre}, "search_content", {"query": genre})
    if roll < 0.65:
        return pick(rng, SEARCH_TITLE, {"title": title}, "search_content", {"query": title})
    if rng.random() < 0.5:
        return pick(rng, SEARCH_APP[:3], {"title": title, "app": app}, "search_content", {"query": title, "app": app})
    return pick(rng, SEARCH_APP[3:], {"genre": genre, "app": app}, "search_content", {"query": genre, "app": app})


def make_more_example(kind, rng, hinglish, n_level, n_min, app, title):
    """The remaining kinds: same_again, choose, subtitles, sleep, settings, risky actions, info, answers."""
    if kind == "same_again":
        return pick(rng, SAME_AGAIN, {"app": app}, "same_again", {"app": app})
    if kind == "choose":
        index = rng.choice([1, 2, 3])
        if hinglish:
            return pick(rng, HI_CHOOSE, {"n": index}, "choose", {"index": index})
        template = rng.choice(CHOOSE)
        if "{n}rd" in template:
            template = "number {n}"
        if "{ord}" not in template:
            index = rng.choice([1, 2, 3, 4, 5])             # a number can go higher than the words we have
        return fill_text(template, {"ord": ORDINAL_WORDS[(index - 1) % 3], "n": index}), "choose", {"index": index}
    if kind == "subtitles":
        state = rng.choice(["on", "off"])
        if hinglish:
            return pick(rng, HI_SUBTITLES_ON if state == "on" else HI_SUBTITLES_OFF, {}, "subtitles", {"state": state})
        return pick(rng, SUBTITLES_ON if state == "on" else SUBTITLES_OFF, {}, "subtitles", {"state": state})
    if kind == "sleep":
        return make_sleep_example(rng, hinglish, n_min)
    if kind == "settings":
        if rng.random() < 0.5:
            return pick(rng, SETTINGS[:4], {}, "settings", {})
        section = rng.choice(SECTIONS)
        return pick(rng, SETTINGS[4:], {"section": section}, "settings", {"section": section})
    if kind == "risky":
        return make_risky_example(rng, hinglish, title)
    return make_talk_example(kind, rng, hinglish, n_level)


def make_sleep_example(rng, hinglish, minutes):
    """A sleep timer example: set, cancel, or ask."""
    roll = rng.random()
    if hinglish:
        return pick(rng, HI_SLEEP, {"n": minutes}, "sleep_timer", {"minutes": minutes})
    if roll < 0.6:
        return pick(rng, SLEEP_SET, {"n": spoken_number(minutes, rng)}, "sleep_timer", {"minutes": minutes})
    if roll < 0.7:
        return rng.choice(["turn the tv off in half an hour", "sleep timer for half an hour"]), "sleep_timer", {"minutes": 30}
    if roll < 0.8:
        return rng.choice(["turn the tv off in an hour", "sleep timer for one hour"]), "sleep_timer", {"minutes": 60}
    if roll < 0.9:
        return pick(rng, SLEEP_OFF, {}, "sleep_timer", {"state": "off"})
    return pick(rng, SLEEP_ASK, {}, "sleep_timer", {})


def make_risky_example(rng, hinglish, title):
    """Factory reset, delete recordings, rent, buy, adult content."""
    which = rng.choice(["factory", "delete", "rent", "rent", "buy", "adult"])
    store_title = rng.choice(STORE_TITLES + OTHER_TITLES)
    if which == "factory":
        return pick(rng, HI_FACTORY if hinglish else FACTORY, {}, "factory_reset", {})
    if which == "delete":
        return pick(rng, HI_DELETE if hinglish else DELETE, {}, "delete_recordings", {})
    if which == "adult":
        return pick(rng, HI_ADULT if hinglish else ADULT, {}, "adult_content", {})
    if which == "rent":
        if hinglish:
            return pick(rng, HI_RENT, {"title": store_title}, "rent_movie", {"title": store_title, "mode": "rent"})
        if rng.random() < 0.12:
            return pick(rng, RENT_ASK, {}, "rent_movie", {"mode": "rent"})
        return pick(rng, RENT, {"title": store_title}, "rent_movie", {"title": store_title, "mode": "rent"})
    if hinglish:
        return pick(rng, HI_BUY, {"title": store_title}, "rent_movie", {"title": store_title, "mode": "buy"})
    if rng.random() < 0.12:
        return pick(rng, BUY_ASK, {}, "rent_movie", {"mode": "buy"})
    return pick(rng, BUY, {"title": store_title}, "rent_movie", {"title": store_title, "mode": "buy"})


def make_talk_example(kind, rng, hinglish, number):
    """Information questions, help, undo, yes, no, cancel, out of scope, bare numbers and noise."""
    if kind == "info":
        which = rng.choice(["time", "time", "status", "undo", "help"])
        if which == "time":
            return pick(rng, HI_TIME if hinglish else ASK_TIME, {}, "ask_time", {})
        if which == "status":
            return pick(rng, HI_STATUS if hinglish else ASK_STATUS, {}, "ask_status", {})
        if which == "undo":
            return pick(rng, HI_UNDO if hinglish else UNDO, {}, "undo", {})
        if hinglish:
            return pick(rng, HI_HELP, {}, "help", {})
        if rng.random() < 0.4:
            topic = rng.choice(TOPICS)
            return pick(rng, HELP_TOPIC, {"topic": topic}, "help", {"topic": topic})
        return pick(rng, HELP, {}, "help", {})
    if kind == "answer":
        roll = rng.random()
        if roll < 0.4:
            return pick(rng, HI_YES if hinglish else YES_SAYINGS, {}, "yes", {})
        if roll < 0.65:
            return pick(rng, HI_NO if hinglish else NO_SAYINGS, {}, "no", {})
        return pick(rng, HI_CANCEL if hinglish else CANCEL_SAYINGS, {}, "cancel", {})
    if kind == "out":
        return pick(rng, HI_OUT if hinglish else OUT_OF_SCOPE, {}, "out_of_scope", {})
    roll = rng.random()
    if roll < 0.55:
        number = rng.choice(LEVELS + CHANNEL_NUMBERS + MINUTES + [1, 2, 3, 1, 2, 3])
        return pick(rng, BARE_NUMBER, {"n": spoken_number(number, rng)}, "other", {"number": number})
    if roll < 0.7:
        bare_title = rng.choice(STORE_TITLES + TITLES)
        return bare_title, "other", {"title": bare_title}
    return rng.choice(NOT_UNDERSTOOD), "other", {}


KIND_WEIGHTS = [
    ("power", 8), ("set_volume", 3), ("volume_up", 4), ("volume_down", 3), ("mute", 3), ("channel", 5), ("app", 3), ("input", 3),
    ("playback", 4), ("search", 4), ("same_again", 2), ("choose", 3), ("subtitles", 2), ("sleep", 3), ("settings", 3), ("risky", 8),
    ("info", 5), ("answer", 7), ("out", 3), ("other", 3),
]


def make_training_examples(how_many=1400, seed=11):
    """Build training examples from facts: an intent plus slots, said in many ways. Same format as edge_nlu.make_example."""
    rng = random.Random(seed)
    kinds = []
    for kind, weight in KIND_WEIGHTS:
        for copy_number in range(weight):
            kinds.append(kind)
    golden_texts = []
    try:
        for case in load_golden():
            golden_texts.append(case["text"].lower())
    except FileNotFoundError:
        pass
    examples = []
    seen = []
    tries = 0
    while len(examples) < how_many and tries < how_many * 30:
        tries = tries + 1
        kind = rng.choice(kinds)
        hinglish = rng.random() < 0.22
        text, intent, slots = make_kind_example(kind, rng, hinglish)
        if rng.random() < 0.35:
            text = make_noisy(text, rng)
        if text.lower() in golden_texts:
            continue                                   # a test message must never be in training
        key = text + "|" + json.dumps(slots) + intent
        if key in seen and kind not in ["answer", "other", "info", "risky", "mute", "playback", "power"]:
            continue
        seen.append(key)
        examples.append(make_example(text, intent, slots))
    return examples


def make_example(text, intent, slots):
    """One training example: the chat the model must learn to continue (same layout as Lab 17, with the TV instruction)."""
    answer = json.dumps({"intent": intent, "slots": slots})
    return {
        "text": text, "intent": intent, "slots": slots,
        "messages": [
            {"role": "system", "content": TV_SYSTEM},
            {"role": "user", "content": '"' + text + '"'},
            {"role": "assistant", "content": answer},
        ],
    }


def count_by_intent(examples):
    """How many training examples each intent has: a dictionary."""
    counts = {}
    for example in examples:
        counts[example["intent"]] = counts.get(example["intent"], 0) + 1
    return counts


# ------------------------------------------------------------------------------------------------ training
def train_tv_adapter(folder_name="tv_adapter", how_many=1400, epochs=2, learning_rate=2e-4, extra_examples=None):
    """Train the TV adapter with the SAME recipe as Lab 17: 4-bit model, LoRA r=16, replay data, 2 epochs, answer-only loss.

    extra_examples is an optional list of examples from make_example() (for the TODO of the lab).
    Returns a dictionary: folder, seconds, peak_gb, examples, steps, losses. Takes about one minute on a 24 GB GPU.
    """
    import gc
    import torch
    import edge_lab
    import edge_train
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    edge_train.check_gpu()
    main_examples = make_training_examples(how_many)
    if extra_examples is not None:
        for example in extra_examples:
            for copy_number in range(5):
                main_examples.append(example)
    train, validation = edge_train.mix_and_split(main_examples, edge_nlu.make_replay_examples())
    tok = AutoTokenizer.from_pretrained(edge_lab.MODEL_ID)
    four_bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(edge_lab.MODEL_ID, quantization_config=four_bit, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    rows = edge_train.tokenise(tok, train)
    result = edge_train.train_run(model, tok, rows, batch_size=8, epochs=epochs, learning_rate=learning_rate,
                                  checkpointing=True, run_name="tv")
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    folder = str(OUTPUT_FOLDER / folder_name)
    model.save_pretrained(folder)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"folder": folder, "seconds": result["seconds"], "peak_gb": result["peak_gb"], "examples": len(train),
            "steps": result["steps"], "losses": result["losses"]}
