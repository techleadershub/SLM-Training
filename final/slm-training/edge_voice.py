"""Day 3 helper for Lab 21: the voice part. Speech recognition (Whisper) and a few demo clips.

If the folder data/voice holds recorded .wav files, those are used. If it does not, the helper makes a few spoken clips with a
text-to-speech model and adds a little noise, so the demo always works.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

VOICE_FOLDER = Path(__file__).resolve().parent / "data" / "voice"
GENERATED_FOLDER = Path(__file__).resolve().parent / "outputs" / "voice_generated"

DEMO_SENTENCES = [
    "turn on the kitchen light",
    "set the bedroom ac to twenty three",
    "close the living room blinds",
    "start movie night",
    "what does error E twenty one mean",
    "play some music",
]


def make_demo_clips(sentences=None, noise=0.01):
    """Speak each sentence with a text-to-speech model, add noise, and save .wav files. Returns the list of file paths."""
    from transformers import AutoTokenizer, VitsModel
    if sentences is None:
        sentences = DEMO_SENTENCES
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    speaker = VitsModel.from_pretrained("facebook/mms-tts-eng")
    paths = []
    random = np.random.RandomState(5)
    for number in range(len(sentences)):
        inputs = tokenizer(sentences[number], return_tensors="pt")
        with torch.no_grad():
            audio = speaker(**inputs).waveform[0].numpy()
        audio = audio + noise * random.randn(len(audio))                 # a little background noise
        path = GENERATED_FOLDER / ("clip" + str(number + 1) + ".wav")
        wavfile.write(str(path), speaker.config.sampling_rate, audio.astype(np.float32))
        paths.append(str(path))
    return paths


def find_clips():
    """The recorded clips if there are any, otherwise made-up clips. Returns (list of paths, where they came from)."""
    recorded = []
    if VOICE_FOLDER.exists():
        for name in sorted(os.listdir(VOICE_FOLDER)):
            if name.lower().endswith(".wav"):
                recorded.append(str(VOICE_FOLDER / name))
    if len(recorded) > 0:
        return recorded, "recorded"
    return make_demo_clips(), "generated"


def load_listener(model_id="openai/whisper-small"):
    """The speech recognition model (Whisper). It runs on your GPU."""
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=model_id, device=device)


def transcribe(listener, path):
    """Turn one .wav file into text. Returns (text, seconds)."""
    sampling_rate, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)                                         # stereo to mono
    if np.abs(audio).max() > 1.5:
        audio = audio / 32768.0                                            # 16-bit integers to numbers between -1 and 1
    started = time.time()
    result = listener({"array": audio, "sampling_rate": sampling_rate})
    seconds = time.time() - started
    return result["text"].strip(), seconds
