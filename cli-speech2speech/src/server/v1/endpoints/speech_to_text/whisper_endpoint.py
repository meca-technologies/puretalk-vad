import base64
from typing import Dict, Any

import numpy as np
from fastapi import APIRouter, HTTPException

from server.core.libs.faster_whisper import transcribe_audio
from server.core.schemas import TranscribeRequest, TranscribeResponse

router = APIRouter(prefix="/text-to-speech")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest) -> Dict[str, Any]:
    if not request.audio_bytes or not request.dtype:
        raise HTTPException(status_code=400, detail="Audio data and dtype are required for transcription.")


    # Decode base64 audio data
    try:
        audio_bytes = base64.b64decode(request.audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    # Transcribe audio
    audio_array = np.frombuffer(audio_bytes, dtype=request.dtype)
    transcription = transcribe_audio(audio_array)

    return {"transcript": transcription}
