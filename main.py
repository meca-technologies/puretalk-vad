import numpy as np
import pyaudio
import torch

from audio_utils import int2float, process_audio_chunk, setup_audio_stream
from vad_utils import CustomVADIterator, model

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = setup_audio_stream(audio, FORMAT, CHANNELS, RATE, CHUNK)

# Initialize VADIterator
vad_iterator = CustomVADIterator(
    model=model, threshold=0.3, sampling_rate=RATE, min_silence_duration_ms=500, speech_pad_ms=30)

try:
    while True:
        data = stream.read(CHUNK)
        audio_np = np.frombuffer(data, dtype=np.int16)
        audio_float32 = int2float(audio_np)
        speech_detection, chunk = vad_iterator(
            torch.from_numpy(audio_float32), return_seconds=True)
        if speech_detection is not None:
            if 'start' in speech_detection:
                print("Speech started at:", speech_detection['start'])
            elif 'end' in speech_detection:
                print("Speech ended at:", speech_detection['end'])
                if chunk is not None:
                    print(process_audio_chunk(chunk))
                    vad_iterator.reset_states()

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
