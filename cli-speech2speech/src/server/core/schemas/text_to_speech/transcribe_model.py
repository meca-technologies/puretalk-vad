from typing import Union, Optional, List, Tuple

from pydantic import BaseModel


#option to add model_id
class TranscribeRequest(BaseModel):
    audio_bytes: str
    dtype: str
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    temperature: Union[float, List[float], Tuple[float, ...]] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    vad_filter: bool = False
    vad_parameters: Optional[dict] = None


class TranscribeResponse(BaseModel):
    transcript: str
