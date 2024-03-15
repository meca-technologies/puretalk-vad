import queue
import threading

import numpy as np
import pyaudio
import torch

from audio_utils import setup_audio_stream, process_audio_chunk, int2float
from vad_utils import CustomVADIterator, model

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = setup_audio_stream(audio, FORMAT, CHANNELS, RATE, CHUNK)

# Initialize queues for inter-thread communication
vad_queue = queue.Queue()
audio_queue = queue.Queue()

# Initialize VADIterator
vad_iterator = CustomVADIterator(model=model, sampling_rate=RATE, min_silence_duration_ms=500, speech_pad_ms=30)


def vad_thread():
    try:
        while True:
            audio_data = vad_queue.get()
            speech_detection, chunk = vad_iterator(audio_data, return_seconds=True)
            if speech_detection is not None:
                if 'start' in speech_detection:
                    audio_queue.empty()
                    print("Speech started at:", speech_detection['start'])
                elif 'end' in speech_detection:
                    print("Speech ended at:", speech_detection['end'])
                    if chunk is not None:
                        audio_queue.put(chunk)
                        vad_iterator.reset_states()
    except KeyboardInterrupt:
        print("VAD thread interrupted")


def audio_thread():
    try:
        while True:
            chunk = audio_queue.get()
            print(process_audio_chunk(chunk))
    except KeyboardInterrupt:
        print("Audio processing thread interrupted")


# Start VAD thread
vad_thread = threading.Thread(target=vad_thread)
vad_thread.start()

# Start audio processing thread
audio_thread = threading.Thread(target=audio_thread)
audio_thread.start()

try:
    while True:
        data = stream.read(CHUNK)
        audio_np = np.frombuffer(data, dtype=np.int16)
        audio_float32 = int2float(audio_np)
        vad_queue.put(torch.from_numpy(audio_float32))

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
