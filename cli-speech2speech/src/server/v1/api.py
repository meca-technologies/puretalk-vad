from fastapi import APIRouter

from server.v1.endpoints.speech_to_text import whisper_socket

router = APIRouter()
router.include_router(whisper_socket.router)
api_router = APIRouter()
api_router.include_router(router, prefix="/v1")
