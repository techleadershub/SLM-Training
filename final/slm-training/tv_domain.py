"""Extra lab X3: the domain of a smart TV. Plain Python, no model, no GPU.

This file is the TV itself: what it has (channels, apps, inputs, a small catalogue), what state it keeps (a plain dictionary),
which values are allowed, which actions are risky, and what each action does. The dialogue manager (tv_manager.py) and the
language reader (tv_nlu.py) both use it. Nothing in Days 1 to 3 is changed: this is a new file.

The idea is the same as home_domain.py of Day 2: the model may PROPOSE an action, but only this file decides if it is allowed
and only this file changes the TV.
"""
import copy

# ------------------------------------------------------------------------------------------------ what the TV has
CHANNELS = {
    1: "DD National", 2: "Star Plus", 3: "Sony Entertainment", 4: "Zee TV", 5: "Star Sports 1", 6: "Sony Sports",
    7: "Aaj Tak", 8: "Star Movies", 9: "Zee Cinema", 10: "Cartoon Network", 11: "Pogo", 12: "MTV", 13: "NDTV", 14: "Discovery",
}
CHANNEL_TAGS = {
    5: ["sports", "cricket"], 6: ["sports", "cricket"], 7: ["news"], 8: ["movies"], 9: ["movies"], 10: ["kids", "cartoons"],
    11: ["kids", "cartoons"], 12: ["music"], 13: ["news"], 14: ["documentary"],
}
CHANNEL_MAX = 999

APPS = ["netflix", "youtube", "prime video", "hotstar"]
APP_NAMES = {"netflix": "Netflix", "youtube": "YouTube", "prime video": "Prime Video", "hotstar": "Hotstar"}
INPUTS = ["hdmi 1", "hdmi 2", "av"]
INPUT_NAMES = {"hdmi 1": "HDMI 1", "hdmi 2": "HDMI 2", "av": "AV"}
SETTINGS_SECTIONS = ["picture", "sound", "network", "accessibility"]

VOLUME_RANGE = (0, 100)
SLEEP_RANGE = (1, 180)
VOLUME_STEP = 5                      # "turn it up" with no amount
PIN = "1234"                         # a teaching value. A real TV keeps the PIN safely and never prints it.
PIN_TRIES = 3

# Each title: genre, kind and the apps that have it.
CATALOG = [
    {"title": "laugh riot", "genre": "comedy", "apps": ["netflix"]},
    {"title": "office hours", "genre": "comedy", "apps": ["netflix", "prime video"]},
    {"title": "funny farm", "genre": "comedy", "apps": ["youtube"]},
    {"title": "mumbai nights", "genre": "thriller", "apps": ["prime video"]},
    {"title": "midnight mystery", "genre": "thriller", "apps": ["netflix", "hotstar"]},
    {"title": "cricket classics", "genre": "sports", "apps": ["hotstar"]},
    {"title": "chhota chef", "genre": "kids", "apps": ["youtube", "hotstar"]},
    {"title": "cosmos explained", "genre": "documentary", "apps": ["youtube", "netflix"]},
    {"title": "river of dreams", "genre": "drama", "apps": ["hotstar"]},
    {"title": "the long road", "genre": "drama", "apps": ["netflix"]},
]
GENRE_WORDS = {
    "funny": "comedy", "comedy": "comedy", "comedies": "comedy", "laugh": "comedy", "humour": "comedy",
    "thriller": "thriller", "thrillers": "thriller", "suspense": "thriller", "scary": "thriller", "mystery": "thriller",
    "cricket": "sports", "sports": "sports", "sport": "sports", "kids": "kids", "children": "kids", "cartoon": "kids",
    "cartoons": "kids", "documentary": "documentary", "documentaries": "documentary", "science": "documentary",
    "drama": "drama", "emotional": "drama", "sad": "drama",
}
FILLER_WORDS = ["something", "some", "a", "an", "the", "show", "shows", "movie", "movies", "film", "films", "anything", "to",
                "watch", "play", "on", "please", "me", "for", "i", "want", "like", "can", "you", "put", "find", "search"]

# The movie store: the price in rupees. A teaching table (not real prices).
STORE = {
    "mumbai nights": {"rent": 149, "buy": 499}, "river of dreams": {"rent": 129, "buy": 449},
    "laugh riot": {"rent": 99, "buy": 349}, "cosmos explained": {"rent": 79, "buy": 299},
}

# Which actions need more than a plain command (the permission tiers).
TIER_CONFIRM = ["factory_reset", "delete_recordings", "rent_movie"]
TIER_PIN = ["adult_content"]
IRREVERSIBLE = ["factory_reset", "delete_recordings", "rent_movie"]      # "undo" must refuse these


def new_tv(overrides=None):
    """A fresh TV: on, live channel 1, volume 20. `overrides` is a dictionary of values to change."""
    tv = {
        "power": "on", "volume": 20, "muted": False, "input": "tv", "channel": 1, "app": "", "playing": True,
        "now_playing": "DD National", "episode": 1, "subtitles": False, "sleep_minutes": 0, "settings": "",
        "recordings": ["Match highlights", "News special", "Cooking show"], "rented": [],
        "adult_unlocked": False, "pin_locked": False, "clock": "20:30",
    }
    if overrides is not None:
        for name in overrides:
            tv[name] = overrides[name]
    return tv


# ------------------------------------------------------------------------------------------------ small helpers
def pretty_title(title):
    """'laugh riot' becomes 'Laugh Riot'."""
    return title.title()


def channel_label(number):
    """'Channel 7, Aaj Tak' or 'Channel 21' when we have no name for it."""
    if number in CHANNELS:
        return "Channel " + str(number) + ", " + CHANNELS[number]
    return "Channel " + str(number)


def say_time(clock):
    """'20:30' becomes '8:30 pm'."""
    hours, minutes = clock.split(":")
    hour = int(hours)
    suffix = "am"
    if hour >= 12:
        suffix = "pm"
    hour = hour % 12
    if hour == 0:
        hour = 12
    return str(hour) + ":" + minutes + " " + suffix


def describe(tv):
    """One or two sentences about what the TV is doing now."""
    if tv["power"] == "off":
        return "The TV is off."
    text = "The TV is on. "
    if tv["input"] == "tv":
        text = text + channel_label(tv["channel"]) + ". "
    elif tv["app"] != "":
        text = text + APP_NAMES.get(tv["app"], tv["app"]) + " is open"
        if tv["now_playing"] != "":
            text = text + ", " + tv["now_playing"]
        text = text + ". "
    else:
        text = text + INPUT_NAMES.get(tv["input"], tv["input"]) + ". "
    if tv["muted"]:
        text = text + "Muted."
    else:
        text = text + "Volume " + str(tv["volume"]) + "."
    return text


def words_of(text):
    """Lower-case words of a text, as a list."""
    cleaned = ""
    for letter in text.lower():
        if letter.isalnum():
            cleaned = cleaned + letter
        else:
            cleaned = cleaned + " "
    return cleaned.split()


# ------------------------------------------------------------------------------------------------ finding things
def find_channel(value):
    """What does the user mean by this channel? Returns (status, data).

    ('ok', number) one channel. ('many', [numbers]) more than one fits. ('none', None) no such channel. ('range', None) a number we do not have.
    """
    if isinstance(value, bool):
        return "none", None
    if isinstance(value, (int, float)):
        number = int(value)
        if number < 1 or number > CHANNEL_MAX:
            return "range", None
        return "ok", number
    query = []
    for word in words_of(str(value)):
        if word not in ["channel", "the", "a", "on", "to", "please", "put"]:
            query.append(word)
    if len(query) == 0:
        return "none", None
    if len(query) == 1 and query[0].isdigit():
        return find_channel(int(query[0]))
    found = []
    for number in CHANNELS:
        pool = words_of(CHANNELS[number])
        for tag in CHANNEL_TAGS.get(number, []):
            pool.append(tag)
        every_word = True
        for word in query:
            if word not in pool:
                every_word = False
        if every_word:
            found.append(number)
    if len(found) == 1:
        return "ok", found[0]
    if len(found) > 1:
        for number in found:
            if CHANNELS[number].lower() == " ".join(query):
                return "ok", number                    # an exact name wins over a partial match
        return "many", found
    return "none", None


def search_catalog(query, app=""):
    """Titles that fit a query (a title or a kind of show), optionally only on one app. Returns a list of dictionaries."""
    tokens = []
    for word in words_of(query):
        if word not in FILLER_WORDS:
            tokens.append(word)
    found = []
    phrase = " ".join(tokens)
    for item in CATALOG:
        match = False
        if phrase != "" and phrase in item["title"]:
            match = True
        elif len(tokens) > 0:
            all_genre = True
            for word in tokens:
                if GENRE_WORDS.get(word) != item["genre"]:
                    all_genre = False
            if all_genre:
                match = True
        elif len(tokens) == 0:
            match = True                               # "play something": everything fits
        if match and app != "" and app not in item["apps"]:
            match = False
        if match:
            found.append(item)
    return found


def check_value(name, value):
    """Is one slot value allowed? Returns (clean_value, problem). problem is '' when the value is fine, else the sentence to say."""
    if name == "level":
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None, "I need a number for the volume, from 0 to 100."
        if number < VOLUME_RANGE[0] or number > VOLUME_RANGE[1]:
            return None, "Sorry, that volume is not possible."
        return number, ""
    if name == "minutes":
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None, "I need a number of minutes."
        if number < SLEEP_RANGE[0] or number > SLEEP_RANGE[1]:
            return None, "The sleep timer goes from 1 to 180 minutes."
        return number, ""
    if name == "app":
        if value in APPS:
            return value, ""
        return None, "I cannot open " + str(value) + " yet."
    if name == "input":
        if value in INPUTS:
            return value, ""
        return None, "I do not have the input " + str(value) + "."
    if name == "amount":
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None, "I need a number for the step."
        if number < 1 or number > 100:
            return None, "A step goes from 1 to 100."
        return number, ""
    return value, ""


def required_slots(intent, slots):
    """The slots still missing for an intent. Returns a list of names (empty when the task is complete)."""
    if intent == "set_volume" and "level" not in slots:
        return ["level"]
    if intent == "set_channel" and "channel" not in slots:
        return ["channel"]
    if intent == "launch_app" and "app" not in slots:
        return ["app"]
    if intent == "set_input" and "input" not in slots:
        return ["input"]
    if intent == "search_content" and "query" not in slots:
        return ["query"]
    if intent == "rent_movie" and "title" not in slots:
        return ["title"]
    if intent == "sleep_timer" and "minutes" not in slots and slots.get("state") != "off":
        return ["minutes"]
    if intent == "playback" and "action" not in slots:
        return ["action"]
    if intent == "power" and "state" not in slots:
        return ["state"]
    return []


QUESTIONS = {
    "level": "What volume, from 0 to 100?",
    "channel": "Which channel?",
    "app": "Which app: Netflix, YouTube, Prime Video or Hotstar?",
    "input": "Which input: HDMI 1, HDMI 2 or AV?",
    "query": "What would you like to watch?",
    "title": "Which movie?",
    "minutes": "How many minutes?",
    "action": "Play, pause or next?",
    "state": "Do you want it on or off?",
}


def tier(intent, tv):
    """How much care this action needs: 'free', 'confirm' (ask yes or no) or 'pin' (ask for the PIN)."""
    if intent in TIER_CONFIRM:
        return "confirm"
    if intent in TIER_PIN and not tv["adult_unlocked"]:
        return "pin"
    return "free"


def confirm_text(intent, args, tv):
    """The question to ask before a risky action."""
    if intent == "factory_reset":
        return "This will erase all your settings and apps. Shall I go ahead? (yes/no)"
    if intent == "delete_recordings":
        return "This will delete all " + str(len(tv["recordings"])) + " recordings for good. Shall I go ahead? (yes/no)"
    if intent == "rent_movie":
        price = STORE[args["title"]][args["mode"]]
        verb = "Rent"
        if args["mode"] == "buy":
            verb = "Buy"
        return verb + " " + pretty_title(args["title"]) + " for " + str(price) + " rupees. Shall I go ahead? (yes/no)"
    return "Shall I go ahead? (yes/no)"


def check_task(intent, slots, tv):
    """Last check before an action. Returns (clean_args, error). error is '' when all is well, else the sentence to say."""
    args = {}
    for name in slots:
        args[name] = slots[name]
    if intent == "rent_movie":
        title = str(args.get("title", "")).lower()
        if title not in STORE:
            return None, "I cannot find " + str(args.get("title")) + " in the movie store."
        args["title"] = title
        if args.get("mode") != "buy":
            args["mode"] = "rent"
        if title in tv["rented"] and args["mode"] == "rent":
            return None, "You have already rented " + pretty_title(title) + "."
    if intent == "delete_recordings" and len(tv["recordings"]) == 0:
        return None, "There are no recordings to delete."
    return args, ""


# ------------------------------------------------------------------------------------------------ what each action does
def execute(tv, intent, args):
    """Do one action. Returns (new_tv, message). The old dictionary is not changed."""
    tv = copy.deepcopy(tv)
    if intent == "power":
        return do_power(tv, args)
    if intent in ["set_volume", "volume_up", "volume_down", "mute"]:
        return do_sound(tv, intent, args)
    if intent == "set_channel":
        tv["input"], tv["app"], tv["channel"], tv["playing"], tv["settings"] = "tv", "", args["channel"], True, ""
        tv["now_playing"] = CHANNELS.get(args["channel"], "Channel " + str(args["channel"]))
        return tv, channel_label(args["channel"]) + "."
    if intent == "launch_app":
        tv["input"], tv["app"], tv["playing"], tv["now_playing"], tv["settings"] = "smart", args["app"], False, "", ""
        return tv, "Opening " + APP_NAMES[args["app"]] + "."
    if intent == "set_input":
        tv["input"], tv["app"], tv["playing"], tv["settings"] = args["input"], "", True, ""
        tv["now_playing"] = INPUT_NAMES[args["input"]]
        return tv, "Switched to " + INPUT_NAMES[args["input"]] + "."
    if intent == "playback":
        return do_playback(tv, args)
    if intent == "play_title":
        tv["input"], tv["app"], tv["playing"], tv["now_playing"], tv["episode"], tv["settings"] = "smart", args["app"], True, pretty_title(args["title"]), 1, ""
        return tv, "Playing " + pretty_title(args["title"]) + " on " + APP_NAMES[args["app"]] + "."
    return do_other(tv, intent, args)


def do_power(tv, args):
    """Power on or off. Switching off also locks adult content again and clears the sleep timer."""
    if args["state"] == "on":
        if tv["power"] == "on":
            return tv, "The TV is already on."
        tv["power"] = "on"
        return tv, "The TV is on."
    if tv["power"] == "off":
        return tv, "The TV is already off."
    tv["power"], tv["playing"], tv["adult_unlocked"], tv["sleep_minutes"], tv["settings"] = "off", False, False, 0, ""
    return tv, "Goodbye. The TV is off."


def do_sound(tv, intent, args):
    """Volume and mute."""
    low, high = VOLUME_RANGE
    if intent == "set_volume":
        tv["volume"], tv["muted"] = args["level"], False
        return tv, "Volume set to " + str(args["level"]) + "."
    if intent == "volume_up" or intent == "volume_down":
        step = args.get("amount", VOLUME_STEP)
        if intent == "volume_up":
            new_level = tv["volume"] + step
        else:
            new_level = tv["volume"] - step
        if new_level > high:
            new_level = high
        if new_level < low:
            new_level = low
        if new_level == tv["volume"] and not tv["muted"]:
            if new_level == high:
                return tv, "The volume is already at the maximum."
            return tv, "The volume is already at " + str(low) + "."
        tv["volume"], tv["muted"] = new_level, False
        if intent == "volume_up":
            return tv, "Volume up to " + str(new_level) + "."
        return tv, "Volume down to " + str(new_level) + "."
    if args["state"] == "on":
        if tv["muted"]:
            return tv, "It is already muted."
        tv["muted"] = True
        return tv, "Muted."
    if not tv["muted"]:
        return tv, "It is not muted."
    tv["muted"] = False
    return tv, "Sound is back on."


def do_playback(tv, args):
    """Play, pause, next."""
    action = args["action"]
    if action == "pause":
        if not tv["playing"]:
            return tv, "It is not playing."
        tv["playing"] = False
        return tv, "Paused."
    if action == "play":
        if tv["playing"]:
            return tv, "It is already playing."
        if tv["now_playing"] == "":
            return tv, "There is nothing to play. Say what you would like to watch."
        tv["playing"] = True
        return tv, "Playing."
    if tv["input"] == "tv":
        tv["channel"] = tv["channel"] + 1
        if tv["channel"] > len(CHANNELS):
            tv["channel"] = 1
        tv["now_playing"] = CHANNELS[tv["channel"]]
        return tv, "Next channel. " + channel_label(tv["channel"]) + "."
    if tv["app"] != "" and tv["now_playing"] != "":
        tv["episode"] = tv["episode"] + 1
        return tv, "Next episode, episode " + str(tv["episode"]) + "."
    return tv, "There is nothing to skip."


def do_other(tv, intent, args):
    """Subtitles, sleep timer, settings, and the risky actions."""
    if intent == "subtitles":
        tv["subtitles"] = args["state"] == "on"
        return tv, "Subtitles are " + args["state"] + "."
    if intent == "sleep_timer":
        if args.get("state") == "off":
            tv["sleep_minutes"] = 0
            return tv, "Sleep timer cancelled."
        tv["sleep_minutes"] = args["minutes"]
        return tv, "Sleep timer set. The TV will switch off in " + str(args["minutes"]) + " minutes."
    if intent == "settings":
        section = args.get("section", "")
        tv["settings"] = section
        if section == "":
            tv["settings"] = "main"
            return tv, "Opening settings."
        return tv, "Opening " + section + " settings."
    if intent == "factory_reset":
        kept_recordings = tv["recordings"]
        kept_clock = tv["clock"]
        tv = new_tv({"recordings": kept_recordings, "clock": kept_clock})
        return tv, "Done. The TV is back to its factory settings."
    if intent == "delete_recordings":
        tv["recordings"] = []
        return tv, "Done. All recordings were deleted."
    if intent == "rent_movie":
        if args["mode"] == "rent":
            tv["rented"].append(args["title"])
        tv["input"], tv["app"], tv["playing"], tv["now_playing"], tv["episode"] = "smart", "rentals", True, pretty_title(args["title"]), 1
        price = STORE[args["title"]][args["mode"]]
        if args["mode"] == "buy":
            return tv, "Done. " + pretty_title(args["title"]) + " is yours for " + str(price) + " rupees. Enjoy."
        return tv, "Done. " + pretty_title(args["title"]) + " is rented for " + str(price) + " rupees. Enjoy."
    if intent == "adult_content":
        tv["input"], tv["app"], tv["playing"], tv["now_playing"] = "smart", "adult zone", True, "Adult zone"
        return tv, "Opening the adult zone."
    return tv, "Nothing happened."


def what_changed(before, after):
    """The keys whose value changed, as a dictionary: name -> (old, new). Used to print only what a turn changed."""
    changes = {}
    for name in after:
        if before.get(name) != after[name]:
            changes[name] = (before.get(name), after[name])
    return changes
