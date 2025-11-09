"""Audio encoding utilities."""

import io
from typing import Optional

import numpy as np
import soundfile as sf

from ..constants import SAMPLING_RATE
from ..exceptions import AudioProcessingError


def encode_to_wav_bytes(
    audio: np.ndarray, sampling_rate: int = SAMPLING_RATE, format: str = "WAV"
) -> bytes:
    """
    Encode audio array to WAV bytes.

    Args:
        audio: Audio array to encode.
        sampling_rate: Sampling rate of the audio.
        format: Audio format (default: "WAV").

    Returns:
        WAV file as bytes.

    Raises:
        AudioProcessingError: If encoding fails.
    """
    if len(audio) == 0:
        raise AudioProcessingError("Cannot encode empty audio array")

    # Ensure data is in correct format
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    try:
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sampling_rate, format=format)
        wav_bytes = wav_buffer.getvalue()
        wav_buffer.close()
        return wav_bytes
    except Exception as e:
        raise AudioProcessingError(
            f"Failed to encode audio to WAV bytes: {e}. "
            f"Audio shape: {audio.shape}, dtype: {audio.dtype}, "
            f"range: [{audio.min():.3f}, {audio.max():.3f}]"
        ) from e

