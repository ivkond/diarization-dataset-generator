"""Audio builder module - assembles final audio from plan."""

import logging
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset

from ..audio.processor import extract_audio_array
from ..constants import AUDIO_TOLERANCE, SAMPLING_RATE
from .types import TrackEvent

logger = logging.getLogger(__name__)


class AudioBuilder:
    """Builder for assembling audio from track plan."""

    def __init__(
        self, dataset: Dataset, speaker_volumes: Dict[int, float], config: Any
    ) -> None:
        """
        Initialize audio builder.

        Args:
            dataset: Dataset to use for audio segments.
            speaker_volumes: Dictionary mapping speaker index to volume.
            config: Configuration object.
        """
        self.dataset = dataset
        self.speaker_volumes = speaker_volumes
        self.config = config

    def build_audio(self, plan: List[TrackEvent]) -> np.ndarray:
        """
        Build final audio from plan.

        Args:
            plan: List of track events.

        Returns:
            Final audio array.
        """
        final_audio_parts: List[np.ndarray] = []

        for i, event in enumerate(plan):
            if event["type"] == "segment":
                audio_array = self._process_segment(event)
                if len(audio_array) > 0:
                    final_audio_parts.append(audio_array)

            elif event["type"] == "pause":
                pause = np.zeros(event["duration_samples"], dtype=np.float32)
                final_audio_parts.append(pause)

            elif event["type"] == "overlap":
                self._apply_overlap(final_audio_parts, plan, i, event)

            elif event["type"] == "simultaneous":
                mixed = self._process_simultaneous(event)
                final_audio_parts.append(mixed)

        # Assemble final audio
        if not final_audio_parts:
            raise ValueError("No audio parts to assemble")

        final_audio = np.concatenate(final_audio_parts)
        return final_audio

    def _process_segment(self, event: TrackEvent) -> np.ndarray:
        """Process a segment event."""
        audio_array = extract_audio_array(
            self.dataset[event["dataset_idx"]][self.config.dataset.feature_audio]
        )
        audio_array = audio_array * event["volume"]

        # Apply trim if specified
        if "trim_start" in event:
            audio_array = audio_array[event["trim_start"] :]

        return audio_array

    def _apply_overlap(
        self, final_audio_parts: List[np.ndarray], plan: List[TrackEvent], i: int, event: TrackEvent
    ) -> None:
        """Apply overlap between segments."""
        if (
            len(final_audio_parts) > 0
            and i + 1 < len(plan)
            and plan[i + 1]["type"] == "segment"
        ):
            # Find last audio segment (skip pauses)
            last_audio_idx = -1
            for j in range(len(final_audio_parts) - 1, -1, -1):
                if isinstance(final_audio_parts[j], np.ndarray):
                    if not np.allclose(final_audio_parts[j], 0, atol=AUDIO_TOLERANCE):
                        last_audio_idx = j
                        break

            if last_audio_idx >= 0:
                # Get next segment (before trimming)
                next_event = plan[i + 1]
                next_audio = extract_audio_array(
                    self.dataset[next_event["dataset_idx"]][self.config.dataset.feature_audio]
                )
                next_audio = next_audio * next_event["volume"]

                # Calculate overlap
                last_segment = final_audio_parts[last_audio_idx]
                overlap_samples = int(len(last_segment) * event["overlap_percent"])
                overlap_samples = min(overlap_samples, len(last_segment), len(next_audio))

                if overlap_samples > 0:
                    # Mix overlap
                    overlap_previous = last_segment[-overlap_samples:].copy()
                    overlap_new = next_audio[:overlap_samples].copy()
                    mixed_overlap = (overlap_previous + overlap_new) / 2.0

                    # Replace end of last segment
                    final_audio_parts[last_audio_idx] = np.concatenate(
                        [last_segment[:-overlap_samples], mixed_overlap]
                    )

                    # Mark next segment to be trimmed
                    if "trim_start" not in next_event:
                        next_event["trim_start"] = 0
                    next_event["trim_start"] += overlap_samples

    def _process_simultaneous(self, event: TrackEvent) -> np.ndarray:
        """Process simultaneous speech event."""
        audio1 = extract_audio_array(
            self.dataset[event["dataset_idx1"]][self.config.dataset.feature_audio]
        )
        audio2 = extract_audio_array(
            self.dataset[event["dataset_idx2"]][self.config.dataset.feature_audio]
        )

        volume1 = self.speaker_volumes.get(event["speaker1_idx"], 1.0)
        volume2 = self.speaker_volumes.get(event["speaker2_idx"], 1.0)

        audio1 = audio1[: event["duration_samples"]] * volume1
        audio2 = audio2[: event["duration_samples"]] * volume2

        mixed = (audio1 + audio2) / 2.0
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

