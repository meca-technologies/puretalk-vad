import threading

import numpy as np
import torch
from faster_whisper import WhisperModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("tiny.en", device="cpu", compute_type="float32")

# Lock for synchronizing access to the model
model_lock = threading.Lock()


def transcribe_audio(audio_array: np.ndarray):
    """
    Transcribes audio using the given audio array.

    Parameters:
    audio_array (np.ndarray): The input audio array.

    Returns:
    str: The transcribed content.
    """
    with model_lock:
        segments, info = model.transcribe(audio_array)
    content = "".join(segment.text for segment in segments)
    return content


def process_audio_chunks(audio_chunks):
    if audio_chunks is None:
        return
    chunk = audio_chunks.cpu().detach().numpy()
    transcription = transcribe_audio(chunk)
    print("Transcription:", transcription)
