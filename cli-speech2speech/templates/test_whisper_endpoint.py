import asyncio
import base64
import json
import pyaudio
import numpy as np
import websockets

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024

async def transcribe_audio():
    async with websockets.connect('ws://localhost:8000/v1/text-to-speech/transcribe') as websocket:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Listening...")

        while True:
            try:
                data = stream.read(CHUNK)
                audio_bytes = base64.b64encode(np.frombuffer(data, dtype=np.float16)).decode('utf-8')
                request_data = {
                    "audio_bytes": audio_bytes,
                    "dtype": "float16"
                }
                await websocket.send(json.dumps(request_data))

                response = await websocket.recv()
                response_dict = json.loads(response)
                transcription = response_dict.get("transcript")
                if transcription:
                    print("Transcription:", transcription)
            except KeyboardInterrupt:
                break

        print("Stopping...")
        stream.stop_stream()
        stream.close()
        p.terminate()

asyncio.run(transcribe_audio())
