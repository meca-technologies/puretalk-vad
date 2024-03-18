import numpy as np
import torch
from faster_whisper import WhisperModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("tiny.en", device=device, compute_type="float16")


def transcribe_audio(audio_array: np.ndarray):
    audio_array = audio_array.squeeze() if audio_array.ndim > 1 else audio_array
    segments, info = model.transcribe(audio_array)
    content = "".join(segment.text for segment in segments)
    return content
