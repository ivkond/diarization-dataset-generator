"""Audio processing module."""

from .encoder import encode_to_wav_bytes
from .noise import add_background_noise, generate_noise
from .processor import extract_audio_array, normalize_audio

__all__ = [
    "extract_audio_array",
    "normalize_audio",
    "generate_noise",
    "add_background_noise",
    "encode_to_wav_bytes",
]

