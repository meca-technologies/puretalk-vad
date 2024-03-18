import base64

import numpy as np
from fastapi import APIRouter, WebSocket

from server.core.libs.faster_whisper import transcribe_audio
from server.core.schemas import TranscribeRequest

router = APIRouter(prefix="/text-to-speech")


class TranscribeWebSocketConnection:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_error(self, message: str):
        await self.websocket.send_json({"error": message})

    async def process_transcription(self, request: TranscribeRequest):
        if not request.audio_bytes or not request.dtype:
            await self.send_error("Audio data and dtype are required for transcription.")
            return

        try:
            audio_bytes = base64.b64decode(request.audio_bytes)
        except Exception as e:
            await self.send_error("Invalid base64 audio data")
            return

        audio_array = np.frombuffer(audio_bytes, dtype=request.dtype)
        transcription = transcribe_audio(audio_array)
        await self.websocket.send_json({"transcript": transcription})


@router.websocket("/transcribe")
async def transcribe(websocket: WebSocket):
    await websocket.accept()
    connection = TranscribeWebSocketConnection(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            request = TranscribeRequest.parse_raw(data)
            await connection.process_transcription(request)

    except Exception as e:
        await connection.send_error(str(e))
