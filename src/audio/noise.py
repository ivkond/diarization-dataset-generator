"""Noise generation and mixing utilities."""

import random
from typing import Optional, Tuple

import numpy as np

from ..constants import (
    AUDIO_NORMAL_RANGE,
    HUM_FREQUENCY_BASE,
    HUM_FREQUENCY_VARIATION,
    HUM_HARMONIC_2_AMPLITUDE,
    HUM_HARMONIC_3_AMPLITUDE,
    PINK_NOISE_FEEDBACK_COEFFS,
    PINK_NOISE_FILTER_COEFFS,
    SAMPLING_RATE,
    STATIC_NOISE_SPIKE_MAX,
    STATIC_NOISE_SPIKE_MIN,
    STATIC_NOISE_SPIKE_RATIO,
)
from ..exceptions import AudioProcessingError


def generate_noise(
    noise_type: str, length_samples: int, sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Generate noise of the specified type.

    Args:
        noise_type: Noise type ('white', 'pink', 'brown', 'crowd', 'cafe', 'static', 'hum').
        length_samples: Length in samples.
        sampling_rate: Sampling rate.

    Returns:
        Array with noise.

    Raises:
        AudioProcessingError: If noise generation fails.
    """
    try:
        if noise_type == "white":
            # White noise - uniform distribution
            noise = np.random.normal(0, 1, length_samples).astype(np.float32)
        elif noise_type == "pink":
            # Pink noise (1/f noise) - approximation via filter
            white = np.random.normal(0, 1, length_samples).astype(np.float32)
            # Simple pink noise approximation via accumulation
            pink = np.zeros_like(white)
            b0, b1, b2, b3, b4, b5, b6 = PINK_NOISE_FILTER_COEFFS
            a1, a2, a3, a4, a5 = PINK_NOISE_FEEDBACK_COEFFS
            for i in range(6, len(white)):
                pink[i] = (
                    b0 * white[i]
                    + b1 * white[i - 1]
                    + b2 * white[i - 2]
                    + b3 * white[i - 3]
                    + b4 * white[i - 4]
                    + b5 * white[i - 5]
                    + b6 * white[i - 6]
                    - a1 * pink[i - 1]
                    - a2 * pink[i - 2]
                    - a3 * pink[i - 3]
                    - a4 * pink[i - 4]
                    - a5 * pink[i - 5]
                )
            noise = pink
        elif noise_type == "brown":
            # Brown noise (1/f^2) - integration of white noise
            white = np.random.normal(0, 1, length_samples).astype(np.float32)
            brown = np.cumsum(white)
            noise = brown
        elif noise_type in ("crowd", "cafe"):
            # Crowd/cafe noise - combination of white and pink noise
            white = np.random.normal(0, 0.7, length_samples).astype(np.float32)
            pink = generate_noise("pink", length_samples, sampling_rate) * 0.3
            noise = white + pink
        elif noise_type == "static":
            # Static noise - white noise with sharp spikes
            white = np.random.normal(0, 1, length_samples).astype(np.float32)
            # Add random spikes
            spikes = np.random.choice(
                length_samples, size=length_samples // STATIC_NOISE_SPIKE_RATIO, replace=False
            )
            for spike in spikes:
                white[spike] = np.random.choice([-1, 1]) * np.random.uniform(
                    STATIC_NOISE_SPIKE_MIN, STATIC_NOISE_SPIKE_MAX
                )
            noise = white
        elif noise_type == "hum":
            # Hum - low-frequency sinusoidal signal
            t = np.arange(length_samples) / sampling_rate
            freq = HUM_FREQUENCY_BASE + np.random.uniform(
                -HUM_FREQUENCY_VARIATION, HUM_FREQUENCY_VARIATION
            )
            noise = np.sin(2 * np.pi * freq * t).astype(np.float32)
            # Add harmonics
            noise += HUM_HARMONIC_2_AMPLITUDE * np.sin(2 * np.pi * freq * 2 * t)
            noise += HUM_HARMONIC_3_AMPLITUDE * np.sin(2 * np.pi * freq * 3 * t)
        else:
            # Default to white noise
            noise = np.random.normal(0, 1, length_samples).astype(np.float32)

        # Normalize
        max_val = np.abs(noise).max()
        if max_val > 0:
            noise = noise / max_val

        return noise
    except Exception as e:
        raise AudioProcessingError(f"Failed to generate {noise_type} noise: {e}") from e


def add_background_noise(
    audio: np.ndarray,
    snr_db: float,
    noise_type: Optional[str] = None,
    noise_types: Optional[list] = None,
) -> Tuple[np.ndarray, str, float]:
    """
    Add background noise to audio with specified SNR.

    Args:
        audio: Audio array.
        snr_db: Signal-to-noise ratio in dB.
        noise_type: Noise type (if None, selected randomly from noise_types).
        noise_types: List of available noise types.

    Returns:
        Tuple of (audio with noise, used noise type, actual SNR).

    Raises:
        AudioProcessingError: If noise addition fails.
    """
    try:
        if noise_type is None:
            if noise_types is None:
                raise AudioProcessingError("Either noise_type or noise_types must be provided")
            noise_type = random.choice(noise_types)

        # Generate noise of the same length
        noise = generate_noise(noise_type, len(audio))

        # Calculate signal and noise power
        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return audio, noise_type, float("inf")

        # Calculate required power ratio
        snr_linear = 10 ** (snr_db / 10.0)

        # Scale noise to achieve desired SNR
        target_noise_power = signal_power / snr_linear
        noise_scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * noise_scale

        # Mix signal and noise
        noisy_audio = audio + scaled_noise

        # Check for clipping and normalize if needed
        max_val = np.abs(noisy_audio).max()
        if max_val > AUDIO_NORMAL_RANGE:
            noisy_audio = noisy_audio / max_val

        # Calculate actual SNR
        actual_signal_power = np.mean(audio**2)
        actual_noise_power = np.mean(scaled_noise**2)
        if actual_noise_power > 0:
            actual_snr = 10 * np.log10(actual_signal_power / actual_noise_power)
        else:
            actual_snr = float("inf")

        return noisy_audio, noise_type, actual_snr
    except Exception as e:
        raise AudioProcessingError(f"Failed to add background noise: {e}") from e

