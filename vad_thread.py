import logging
import os
import queue
import threading

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv
from elevenlabs import generate, stream
from faster_whisper import WhisperModel
from groq import Groq

from audio_utils import int2float, setup_audio_stream
from vad_utils import CustomVADIterator, model as vad_model

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
audio_stream = setup_audio_stream(audio, FORMAT, CHANNELS, RATE, CHUNK)

# Initialize VADIterator
vad_iterator = CustomVADIterator(
    model=vad_model, threshold=0.7, sampling_rate=RATE, min_silence_duration_ms=500, speech_pad_ms=30)

device = "cuda" if torch.cuda.is_available() else "cpu"
transcription_model = WhisperModel("tiny.en", device="cpu", compute_type="float32")

interrupting_event = threading.Event()

audio_queue = queue.Queue()
llm_queue = queue.Queue()
eleven_queue = queue.Queue()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_with_llm():
    while True:
        if interrupting_event.is_set():
            logger.info("Interrupting...")
            # break
        else:
            try:
                query = llm_queue.get()
                if query != "":
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": query},
                        ],
                        model="mixtral-8x7b-32768",
                        stream=False
                    )
                    logger.info("LLM Response: " + response.choices[0].message.content)
                    llm_queue.task_done()
                    eleven_queue.put(response.choices[0].message.content)
            except queue.Empty:
                continue


def eleven_thread():
    while not interrupting_event.is_set():
        # if interrupting_event.is_set():
        #     logger.info("Interrupting...")
        #     break
        try:
            response = eleven_queue.get()
            if response != "":
                output = generate(
                    api_key=os.environ.get("ELEVENLABS_API_KEY"),
                    text=response,
                    model="eleven_monolingual_v1",
                    stream=True
                )
                stream(output)
                eleven_queue.task_done()
        except queue.Empty:
            continue


def vad_thread():
    logger.info("Listening...")
    while True:
        data = audio_stream.read(CHUNK)
        audio_np = np.frombuffer(data, dtype=np.int16)
        audio_float32 = int2float(audio_np)
        speech_detection, chunk = vad_iterator(
            torch.from_numpy(audio_float32), return_seconds=True)
        if speech_detection is not None:
            if 'start' in speech_detection:
                interrupting_event.set()
                audio_queue.queue.clear()
                llm_queue.queue.clear()
                eleven_queue.queue.clear()
                logger.info("Speech started at: %s", speech_detection['start'])
            elif 'end' in speech_detection:
                interrupting_event.clear()
                logger.info("Speech ended at: %s", speech_detection['end'])
                if chunk is not None:
                    audio_queue.put(chunk)
                    vad_iterator.reset_states()


def transcribe_audio_chunks():
    while True:
        if interrupting_event.is_set():
            logger.info("Interrupting...")
        else:
            try:
                audio_chunks = audio_queue.get()
                chunk = audio_chunks.cpu().detach().numpy()
                segments, info = transcription_model.transcribe(chunk)
                content = "".join(segment.text for segment in segments)
                logger.info("Transcription: %s", content)
                llm_queue.put(content)
            except queue.Empty:
                continue


# Thread for LLM processing
llm_thread = threading.Thread(target=process_with_llm)

# Thread for VAD
vad_thread = threading.Thread(target=vad_thread)

# Thread for audio transcription
whisper_thread = threading.Thread(target=transcribe_audio_chunks)

# Thread for ElevenLabs
eleven_thread = threading.Thread(target=eleven_thread)

# Start LLM, VAD, and audio transcription threads
llm_thread.start()
vad_thread.start()
whisper_thread.start()
eleven_thread.start()
