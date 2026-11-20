"""Helper for extra lab X4: an end-to-end voice assistant (speech in, dialogue manager, speech out).

The chain is   speech to text (Whisper)  ->  dialogue manager (the TV manager of X3, or the home manager of Day 2)  ->  text to speech.

Plain Python on purpose. Heavy imports (torch, transformers) are done inside the functions that need them, so the small helpers
(text cleaning, error rate, noise, the conversations) also work on a laptop without a GPU.

The dialogue manager sits behind ONE small adapter: new_assistant(), reset_assistant() and take_turn(). Nothing else in the lab
or in this file knows which manager is running. To switch manager, change the word you give to start_assistant().
Reuses edge_voice.py (the Whisper loader and the file reader of Lab 21) and the same text-to-speech model as its demo clips.
"""
import copy
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
VOICE_FOLDER = HERE / "data" / "voice"                 # recorded .wav files go here (same folder as Lab 21)
GENERATED_FOLDER = HERE / "outputs" / "x_voice"        # the clips we make with text to speech are saved here
WHISPER_RATE = 16000
TTS_MODEL = "facebook/mms-tts-eng"
KINDS = ["tv_model", "tv_rules", "home", "home_rules"]
PAD_SECONDS = 0.4                                      # silence before and after every clip (short clips confuse Whisper)
WAKE_PHRASE = "hey computer"

# ------------------------------------------------------------------------------------------------ the conversations
# Each conversation is a name and the sentences the user says, one per turn. They are spoken by the text-to-speech model.
HOME_CONVERSATIONS = [
    ("one sentence", ["turn on the kitchen light"]),
    ("slot filling", ["set the air conditioner in the bedroom", "make it twenty two degrees"]),
    ("blinds", ["please open the study blinds"]),
    ("interruption", ["set the air conditioner in the study", "what does error E twenty one mean on the washer",
                      "make it twenty one degrees"]),
    ("risky action", ["i am leaving the house now", "yes please do it"]),
    ("risky refused", ["run away mode for me", "no cancel that"]),
]
TV_CONVERSATIONS = [
    ("volume", ["set the volume to thirty please", "turn it up a little"]),
    ("channel and correction", ["please go to channel seven", "no i meant channel nine"]),
    ("app", ["open netflix on the tv", "turn the subtitles on please"]),
    ("two commands", ["mute it and switch to hdmi two"]),
    ("risky action", ["please factory reset the tv", "no cancel that"]),
]
# Every turn is a full natural sentence: one-word turns ("yes", "bedroom") are not recognised well by a speech model.


def conversations(kind):
    """The list of (name, sentences) for the manager that is running."""
    if kind.startswith("tv"):
        return TV_CONVERSATIONS
    return HOME_CONVERSATIONS


# ------------------------------------------------------------------------------------------------ text helpers
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SAY_AS = {"AC": "air conditioner", "TV": "T V", "HDMI": "H D M I", "AV": "A V", "PIN": "pin", "OK": "okay"}


def number_words(number):
    """23 -> 'twenty three'. Numbers of three digits or more are read digit by digit."""
    if number < 20:
        return ONES[number]
    if number < 100:
        text = TENS[number // 10]
        if number % 10 > 0:
            text = text + " " + ONES[number % 10]
        return text
    pieces = []
    for digit in str(number):
        pieces.append(ONES[int(digit)])
    return " ".join(pieces)


def _digits_to_words(match):
    return " " + number_words(int(match.group(0))) + " "


def speakable(text, max_words=40):
    """Make a reply easy for a text-to-speech model: numbers and short forms as words, no symbols, not too long."""
    text = re.sub(r"(\d+)\s*C\b", r"\1 degrees", text)
    text = text.replace("%", " percent")
    for short in SAY_AS:
        text = re.sub(r"\b" + short + r"\b", SAY_AS[short], text)
    text = re.sub(r"\d+", _digits_to_words, text)
    text = re.sub(r"[^A-Za-z' ]", " ", text).lower()
    words = text.split()
    if len(words) == 0:
        return "okay"
    return " ".join(words[:max_words])


def _number_value(word):
    """The value of a spoken number word below 100, or -1 if the word is not one."""
    if word in ONES:
        return ONES.index(word)
    if word in TENS and word != "":
        return 10 * TENS.index(word)
    return -1


def clean_words(text):
    """Lower case, no punctuation, spoken numbers as digits ('twenty three' -> '23'), 'e 21' -> 'e21'. A list of words."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    raw = text.split()
    words = []
    index = 0
    while index < len(raw):
        value = _number_value(raw[index])
        if value < 0:
            words.append(raw[index])
            index = index + 1
        elif value >= 20 and value % 10 == 0 and index + 1 < len(raw) and 0 < _number_value(raw[index + 1]) < 10:
            words.append(str(value + _number_value(raw[index + 1])))
            index = index + 2
        else:
            words.append(str(value))
            index = index + 1
    merged = []
    for word in words:
        if len(merged) > 0 and len(merged[-1]) == 1 and merged[-1].isalpha() and word.isdigit():
            merged[-1] = merged[-1] + word
        else:
            merged.append(word)
    return merged


def edit_distance(first, second):
    """The number of insertions, deletions and changes to turn one list (or text) into the other."""
    previous = list(range(len(second) + 1))
    for i in range(1, len(first) + 1):
        current = [i]
        for j in range(1, len(second) + 1):
            cost = 1
            if first[i - 1] == second[j - 1]:
                cost = 0
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[len(second)]


def wer(said, heard):
    """Word error rate in per cent (shown at most 100, so one hallucinated text cannot distort an average): the word edits needed to turn what was HEARD into what was SAID, over the words SAID."""
    said_words = clean_words(said)
    if len(said_words) == 0:
        return 0.0
    return min(100.0, round(100.0 * edit_distance(said_words, clean_words(heard)) / len(said_words), 1))


def diff_state(old, new):
    """The keys of the device state that changed, with their new values."""
    changed = {}
    for key in new:
        if key not in old or old[key] != new[key]:
            changed[key] = new[key]
    return changed


# ------------------------------------------------------------------------------------------------ audio helpers
def read_wav(path):
    """A .wav file as (mono audio as numbers between -1 and 1, sampling rate)."""
    rate, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(rate)


def save_wav(path, audio, rate):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), int(rate), np.asarray(audio, dtype=np.float32))


def to_16k(audio, rate):
    """Whisper listens at 16 000 samples per second. A simple straight-line resampling is enough for speech."""
    audio = np.asarray(audio, dtype=np.float32)
    if int(rate) == WHISPER_RATE:
        return audio
    new_length = int(round(len(audio) * WHISPER_RATE / float(rate)))
    old_positions = np.linspace(0.0, 1.0, num=len(audio))
    new_positions = np.linspace(0.0, 1.0, num=new_length)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def pad_silence(audio, rate, seconds=PAD_SECONDS):
    """Silence before and after the clip. Whisper is much more reliable on clips that do not start and end abruptly."""
    silence = np.zeros(int(seconds * rate), dtype=np.float32)
    return np.concatenate([silence, np.asarray(audio, dtype=np.float32), silence])


def add_noise(audio, snr_db, seed=11):
    """Add white noise so that the voice is snr_db decibels stronger than the noise. 20 = light, 10 = medium, 0 = same strength."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = 0.0
    if len(audio) > 0:
        peak = float(np.max(np.abs(audio)))
    voiced = audio[np.abs(audio) > 0.02 * peak]                 # measure the voice only, not the silence around it
    if len(voiced) == 0:
        voiced = audio
    voice_power = float(np.mean(voiced * voiced))
    noise_power = voice_power / (10.0 ** (snr_db / 10.0))
    random = np.random.RandomState(seed)
    noise = random.randn(len(audio)) * math.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


# ------------------------------------------------------------------------------------------------ the ears and the mouth
def load_listener():
    """Whisper small (reuses the loader of Lab 21)."""
    import edge_voice
    return edge_voice.load_listener()


def listen(listener, audio, rate):
    """Speech to text for an array of audio. Returns (text, seconds)."""
    audio16 = to_16k(audio, rate)
    started = time.time()
    options = {"language": "english", "task": "transcribe", "max_new_tokens": 40}
    result = listener({"array": audio16, "sampling_rate": WHISPER_RATE}, generate_kwargs=options)
    seconds = time.time() - started
    return result["text"].strip(), seconds


def listen_confidence(listener, audio, rate, text):
    """How sure was Whisper about this text? We feed the text back to it and read the probability of each token it would choose.

    Returns {"mean": average token probability, "lowest": the weakest token}, or {"mean": None, ...} with an "error" if it fails.
    """
    try:
        import torch
        model = listener.model
        tokenizer = listener.tokenizer
        audio16 = to_16k(audio, rate)
        features = listener.feature_extractor(audio16, sampling_rate=WHISPER_RATE, return_tensors="pt")["input_features"]
        features = features.to(model.device).to(model.dtype)
        prefix = []
        for name in ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]:
            prefix.append(tokenizer.convert_tokens_to_ids(name))
        body = []
        for token_id in tokenizer(" " + text.strip(), add_special_tokens=False)["input_ids"]:
            if token_id < tokenizer.eos_token_id:                       # keep ordinary tokens only
                body.append(token_id)
        ids = prefix + body + [tokenizer.eos_token_id]
        decoder_input = torch.tensor([ids[:-1]]).to(model.device)
        with torch.no_grad():
            logits = model(input_features=features, decoder_input_ids=decoder_input).logits[0].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = []
        for position in range(len(prefix) - 1, len(ids) - 1):
            probabilities.append(math.exp(float(log_probabilities[position, ids[position + 1]])))
        return {"mean": round(sum(probabilities) / len(probabilities), 3), "lowest": round(min(probabilities), 3)}
    except Exception as problem:
        return {"mean": None, "lowest": None, "error": str(problem)}


def load_speaker():
    """The text-to-speech model of edge_voice.make_demo_clips (facebook/mms-tts-eng). Uses the GPU if there is one."""
    import torch
    from transformers import AutoTokenizer, VitsModel
    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
    model = VitsModel.from_pretrained(TTS_MODEL)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device, "rate": int(model.config.sampling_rate)}


def speak(speaker, text, seed=5, pad=True):
    """Text to speech. Returns (audio as numbers, sampling rate), with a little silence added at both ends unless pad=False.
    The seed makes the same text sound the same each time."""
    import torch
    torch.manual_seed(seed)
    inputs = speaker["tokenizer"](text, return_tensors="pt").to(speaker["device"])
    with torch.no_grad():
        audio = speaker["model"](input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).waveform[0].cpu().numpy()
    if pad:
        return pad_silence(audio, speaker["rate"]), speaker["rate"]
    return audio.astype(np.float32), speaker["rate"]


def clip_name(conversation_number, turn_number):
    return "conv" + str(conversation_number) + "_turn" + str(turn_number) + ".wav"


def make_clips(speaker, conversation_list):
    """One clip per turn: [[{"said", "audio", "rate", "source", "path"}, ...], ...].

    If data/voice holds a recording named conv1_turn1.wav (conversation 1, turn 1) that file is used; otherwise the sentence is
    spoken by the text-to-speech model and saved in outputs/x_voice with the same name, so you can see what to record.
    """
    all_clips = []
    folder = "home"
    if conversation_list is TV_CONVERSATIONS:
        folder = "tv"                                  # recordings live in data/voice/tv or data/voice/home
    for c in range(len(conversation_list)):
        turns = conversation_list[c][1]
        clips = []
        for t in range(len(turns)):
            name = clip_name(c + 1, t + 1)
            recorded = VOICE_FOLDER / folder / name
            if recorded.exists():
                audio, rate = read_wav(recorded)
                audio = pad_silence(audio, rate)
                clips.append({"said": turns[t], "audio": audio, "rate": rate, "source": "recorded", "path": str(recorded)})
            else:
                audio, rate = speak(speaker, turns[t])
                path = GENERATED_FOLDER / name
                save_wav(path, audio, rate)
                clips.append({"said": turns[t], "audio": audio, "rate": rate, "source": "generated", "path": str(path)})
        all_clips.append(clips)
    return all_clips


# ------------------------------------------------------------------------------------------------ the dialogue manager adapter
# Everything that depends on WHICH manager runs is in the next four functions. The rest of the lab only calls these.
def new_assistant(kind):
    """Build the dialogue manager. kind is one of KINDS. Returns a handle (a dictionary)."""
    if kind.startswith("tv"):
        import tv_manager
        manager = tv_manager.make_manager(use_model=(kind == "tv_model"))
        tv_manager.reset(manager)
        return {"kind": kind, "tm": tv_manager, "manager": manager, "shadow": None, "state": copy.deepcopy(tv_manager.get_state(manager))}
    import home_manager
    import home_domain
    if kind == "home_rules":
        app = home_manager.build(understand=home_manager.rules_understand)
    else:
        app = home_manager.build()                       # the 3-billion-parameter model of Day 2 reads the messages
    return {"kind": kind, "app": app, "thread": "x4-0", "count": 0, "state": home_domain.new_home()}


def reset_assistant(assistant):
    """Start a new conversation on a new device."""
    if assistant["kind"].startswith("tv"):
        assistant["tm"].reset(assistant["manager"])
        if assistant["shadow"] is not None:
            assistant["tm"].reset(assistant["shadow"])
        assistant["state"] = copy.deepcopy(assistant["tm"].get_state(assistant["manager"]))
        return
    import home_domain
    assistant["count"] = assistant["count"] + 1
    assistant["thread"] = "x4-" + str(assistant["count"])
    assistant["state"] = home_domain.new_home()


def _home_turn(assistant, text):
    config = {"configurable": {"thread_id": assistant["thread"]}}
    started = time.time()
    understood_at = None
    for event in assistant["app"].stream({"utterance": text}, config, stream_mode="updates"):
        if understood_at is None:
            understood_at = time.time()                  # the first node to finish is "understand"
    finished = time.time()
    values = assistant["app"].get_state(config).values
    if understood_at is None:
        understood_at = finished
    nlu = values.get("nlu") or {}
    return {"reply": values.get("reply", ""), "state": values.get("home") or assistant["state"], "path": values.get("path", []),
            "intent": nlu.get("intent"), "confidence": None,
            "understand_ms": 1000 * (understood_at - started), "act_ms": 1000 * (finished - understood_at)}


def _tv_turn(assistant, text):
    tm = assistant["tm"]
    started = time.time()
    result = tm.turn(assistant["manager"], text)
    total_ms = 1000 * (time.time() - started)
    if "understand_ms" in result and result["understand_ms"] is not None:
        understand_ms = result["understand_ms"]
    elif assistant["kind"] == "tv_model":
        # The manager does not time its own steps, so we measure the CODE part with the rules version on the same sentence.
        if assistant["shadow"] is None:
            assistant["shadow"] = tm.make_manager(use_model=False)
            tm.reset(assistant["shadow"])
        shadow_started = time.time()
        tm.turn(assistant["shadow"], text)
        understand_ms = max(0.0, total_ms - 1000 * (time.time() - shadow_started))
    else:
        understand_ms = 0.0                              # the rules version has no model call: all of it is code
    return {"reply": result.get("reply", ""), "state": result.get("state") or assistant["state"], "path": result.get("path", []),
            "intent": result.get("intent"), "confidence": result.get("confidence"),
            "understand_ms": understand_ms, "act_ms": max(0.0, total_ms - understand_ms)}


def take_turn(assistant, text):
    """One text turn. Returns reply, state (after the turn), changed (what the turn changed), path, intent, confidence,
    understand_ms (the language reader) and act_ms (the code that decides and acts)."""
    if assistant["kind"].startswith("tv"):
        turn = _tv_turn(assistant, text)
    else:
        turn = _home_turn(assistant, text)
    turn["state"] = copy.deepcopy(turn["state"])
    turn["changed"] = diff_state(assistant["state"], turn["state"])
    assistant["state"] = turn["state"]
    turn["understand_ms"] = round(turn["understand_ms"])
    turn["act_ms"] = round(turn["act_ms"])
    return turn


def start_assistant(prefer="auto"):
    """Build the manager. prefer is 'auto' (the TV manager with the small model if it works, else the home manager),
    or one of KINDS. Returns (assistant, kind, a sentence that says why)."""
    if prefer != "auto":
        return new_assistant(prefer), prefer, "you asked for " + prefer
    try:
        return new_assistant("tv_model"), "tv_model", "the TV manager of X3 with the small model"
    except Exception as problem:
        reason = "the TV manager with the model is not available (" + str(problem)[:90] + "), so the home manager of Day 2 is used"
        return new_assistant("home"), "home", reason


# ------------------------------------------------------------------------------------------------ run conversations
def run_conversations(assistant, conversation_list, clips=None, listener=None, snr_db=None, score=True):
    """Run every conversation, turn by turn, on a fresh device. One row per turn.

    With listener=None the sentences go in as clean TEXT (the reference run). Otherwise each clip is heard by Whisper first, after adding
    noise at snr_db decibels (None = no noise added). The rows hold what was said and heard, the manager's answer and the state.
    """
    rows = []
    for c in range(len(conversation_list)):
        reset_assistant(assistant)
        turns = conversation_list[c][1]
        for t in range(len(turns)):
            row = {"conversation": c, "name": conversation_list[c][0], "turn": t + 1, "said": turns[t], "confidence_speech": None}
            heard = " ".join(clean_words(turns[t]))      # typed text: lower case, no punctuation, numbers as digits
            if listener is not None:
                clip = clips[c][t]
                audio = clip["audio"]
                if snr_db is not None:
                    audio = add_noise(audio, snr_db, seed=100 + 10 * c + t)
                heard, seconds = listen(listener, audio, clip["rate"])
                if score:
                    row["confidence_speech"] = listen_confidence(listener, audio, clip["rate"], heard)["mean"]
                row["speech_ms"] = round(1000 * seconds)
            turn = take_turn(assistant, heard)
            row["heard"] = heard
            row["wer"] = wer(turns[t], heard)
            row["reply"] = turn["reply"]
            row["intent"] = turn["intent"]
            row["intent_confidence"] = turn["confidence"]
            row["changed"] = turn["changed"]
            row["state_after"] = turn["state"]
            row["path"] = turn["path"]
            rows.append(row)
    return rows


def result_same(row, reference_row):
    """Did this turn end the same way as in the reference run: same reply and same device state?"""
    return row["reply"] == reference_row["reply"] and row["state_after"] == reference_row["state_after"]


def compare_runs(rows, reference):
    """Compare a heard run with the text run. Returns a dictionary of summary numbers (per cent, and counts of conversations)."""
    total = len(rows)
    wer_sum = 0.0
    heard_exact = 0
    same_intent = 0
    same_result = 0
    last_row = {}
    for i in range(total):
        wer_sum = wer_sum + rows[i]["wer"]
        if rows[i]["wer"] == 0:
            heard_exact = heard_exact + 1
        if rows[i]["intent"] == reference[i]["intent"]:
            same_intent = same_intent + 1
        if result_same(rows[i], reference[i]):
            same_result = same_result + 1
        last_row[rows[i]["conversation"]] = i
    same_final = 0
    for c in last_row:
        i = last_row[c]
        if rows[i]["state_after"] == reference[i]["state_after"]:
            same_final = same_final + 1
    return {"mean wer %": round(wer_sum / max(total, 1), 1), "heard exactly %": round(100.0 * heard_exact / max(total, 1)),
            "same intent %": round(100.0 * same_intent / max(total, 1)), "same result %": round(100.0 * same_result / max(total, 1)),
            "conversations ending right": str(same_final) + " of " + str(len(last_row))}


def classify_row(row, reference_row):
    """Where did this turn go wrong? One of four labels."""
    right_result = result_same(row, reference_row)
    if row["wer"] > 0 and right_result:
        return "speech wrong, result still right"
    if row["wer"] > 0:
        return "speech wrong, result wrong"
    if not right_result:
        return "speech right, result differs"
    return "all right"
