"""Audio processing utilities."""

from typing import Dict, Union

import numpy as np

from ..exceptions import AudioProcessingError


def extract_audio_array(audio_data: Union[Dict, np.ndarray]) -> np.ndarray:
    """
    Extract numpy array from audio data.

    Args:
        audio_data: Audio data, either a dict with 'array' key or numpy array.

    Returns:
        Numpy array with audio data.

    Raises:
        AudioProcessingError: If audio data cannot be extracted.
    """
    try:
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = audio_data["array"]
        else:
            audio_array = audio_data

        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)

        return audio_array
    except Exception as e:
        raise AudioProcessingError(f"Failed to extract audio array: {e}") from e


def normalize_audio(audio: np.ndarray, max_value: float = 1.0) -> np.ndarray:
    """
    Normalize audio to prevent clipping.

    Args:
        audio: Audio array to normalize.
        max_value: Maximum allowed absolute value (default: 1.0).

    Returns:
        Normalized audio array.
    """
    max_val = np.abs(audio).max()
    if max_val > max_value:
        audio = audio / max_val
    return audio

