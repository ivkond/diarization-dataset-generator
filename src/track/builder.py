"""Audio builder module - assembles final audio from plan."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from ..audio.processor import extract_audio_array
from ..constants import AUDIO_TOLERANCE, SAMPLING_RATE
from .types import TrackEvent

logger = logging.getLogger(__name__)


class AudioBuilder:
    """Builder for assembling audio from track plan."""

    def __init__(
        self, 
        dataset: Dataset, 
        speaker_volumes: Dict[int, float], 
        config: Any,
        audio_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Initialize audio builder.

        Args:
            dataset: Dataset to use for audio segments.
            speaker_volumes: Dictionary mapping speaker index to volume.
            config: Configuration object.
            audio_cache: Optional audio cache dictionary (dataset_idx -> audio_array).
        """
        self.dataset = dataset
        self.speaker_volumes = speaker_volumes
        self.config = config
        self.audio_cache = audio_cache if audio_cache is not None else {}

    def _get_audio_from_cache_or_dataset(self, dataset_idx: int) -> Optional[np.ndarray]:
        """
        Get audio array from cache or dataset.
        
        Args:
            dataset_idx: Dataset index.
            
        Returns:
            Audio array or None if extraction fails.
        """
        # Check cache first
        if dataset_idx in self.audio_cache:
            return self.audio_cache[dataset_idx]
        
        # Extract from dataset
        try:
            sample = self.dataset[dataset_idx]
            audio_array = extract_audio_array(sample[self.config.dataset.feature_audio])
            # Store in cache for future use
            self.audio_cache[dataset_idx] = audio_array
            return audio_array
        except Exception as e:
            logger.debug(f"Failed to extract audio from dataset index {dataset_idx}: {e}")
            return None

    def build_audio(self, plan: List[TrackEvent]) -> np.ndarray:
        """
        Build final audio from plan using pre-allocation for memory efficiency.

        Args:
            plan: List of track events.

        Returns:
            Final audio array.
        """
        # Pre-calculate total size for memory efficiency
        total_samples = 0
        for event in plan:
            if event["type"] == "segment":
                # Estimate segment size (will be adjusted during processing)
                audio_array = self._get_audio_from_cache_or_dataset(event["dataset_idx"])
                if audio_array is not None:
                    trim_start = event.get("trim_start", 0)
                    total_samples += len(audio_array) - trim_start
            elif event["type"] == "pause":
                total_samples += event["duration_samples"]
            elif event["type"] == "simultaneous":
                total_samples += event["duration_samples"]
            # Overlap doesn't add samples, it modifies existing ones

        if total_samples == 0:
            raise ValueError("No audio parts to assemble")

        # Pre-allocate final audio array
        final_audio = np.zeros(total_samples, dtype=np.float32)
        offset = 0
        segment_positions = []  # Track (start, end) positions for each segment event

        for i, event in enumerate(plan):
            if event["type"] == "segment":
                segment_start = offset
                audio_array = self._process_segment(event)
                if len(audio_array) > 0:
                    end = offset + len(audio_array)
                    if end > total_samples:
                        # Resize if needed (shouldn't happen often)
                        new_size = max(end, int(total_samples * 1.1))
                        final_audio = np.resize(final_audio, new_size)
                        total_samples = new_size
                    final_audio[offset:end] = audio_array
                    offset = end
                    segment_positions.append((i, segment_start, offset))

            elif event["type"] == "pause":
                pause_samples = event["duration_samples"]
                end = offset + pause_samples
                if end > total_samples:
                    new_size = max(end, int(total_samples * 1.1))
                    final_audio = np.resize(final_audio, new_size)
                    total_samples = new_size
                # Pause is already zeros, no need to set
                offset = end

            elif event["type"] == "overlap":
                # Overlap modifies previous segments, will be handled after all segments are written
                pass

            elif event["type"] == "simultaneous":
                mixed = self._process_simultaneous(event)
                end = offset + len(mixed)
                if end > total_samples:
                    new_size = max(end, int(total_samples * 1.1))
                    final_audio = np.resize(final_audio, new_size)
                    total_samples = new_size
                final_audio[offset:end] = mixed
                offset = end

        # Apply overlaps (modify already written segments)
        for i, event in enumerate(plan):
            if event["type"] == "overlap":
                self._apply_overlap_inplace(final_audio, plan, i, event, segment_positions, offset)

        # Trim to actual size
        if offset < total_samples:
            final_audio = final_audio[:offset]

        return final_audio

    def _process_segment(self, event: TrackEvent) -> np.ndarray:
        """Process a segment event."""
        # Use cache for audio extraction
        audio_array = self._get_audio_from_cache_or_dataset(event["dataset_idx"])
        if audio_array is None:
            raise ValueError(f"Failed to get audio for dataset_idx {event['dataset_idx']}")
        
        audio_array = audio_array * event["volume"]

        # Apply trim if specified
        if "trim_start" in event:
            audio_array = audio_array[event["trim_start"] :]

        return audio_array

    def _apply_overlap_inplace(
        self, final_audio: np.ndarray, plan: List[TrackEvent], i: int, event: TrackEvent, 
        segment_positions: List[Tuple[int, int, int]], current_offset: int
    ) -> None:
        """Apply overlap between segments in pre-allocated array."""
        if i + 1 >= len(plan) or plan[i + 1]["type"] != "segment":
            return

        # Find the last segment before this overlap event
        last_segment_idx = -1
        last_segment_start = 0
        last_segment_end = 0
        
        for seg_idx, seg_start, seg_end in reversed(segment_positions):
            if seg_idx < i:  # Segment is before overlap event
                last_segment_idx = seg_idx
                last_segment_start = seg_start
                last_segment_end = seg_end
                break
        
        if last_segment_idx < 0:
            return  # No previous segment found
        
        # Get next segment (before trimming)
        next_event = plan[i + 1]
        next_audio = self._get_audio_from_cache_or_dataset(next_event["dataset_idx"])
        if next_audio is None:
            return
        next_audio = next_audio * next_event["volume"]
        
        # Calculate overlap
        last_segment_length = last_segment_end - last_segment_start
        overlap_samples = int(last_segment_length * event["overlap_percent"])
        overlap_samples = min(overlap_samples, last_segment_length, len(next_audio))
        
        if overlap_samples > 0:
            overlap_start = last_segment_end - overlap_samples
            overlap_end = last_segment_end
            
            # Mix overlap
            overlap_previous = final_audio[overlap_start:overlap_end].copy()
            overlap_new = next_audio[:overlap_samples].copy()
            mixed_overlap = (overlap_previous + overlap_new) / 2.0
            
            # Replace overlap region
            final_audio[overlap_start:overlap_end] = mixed_overlap
            
            # Mark next segment to be trimmed (this will be handled in _process_segment)
            if "trim_start" not in next_event:
                next_event["trim_start"] = 0
            next_event["trim_start"] += overlap_samples

    def _process_simultaneous(self, event: TrackEvent) -> np.ndarray:
        """Process simultaneous speech event."""
        # Use cache for audio extraction
        audio1 = self._get_audio_from_cache_or_dataset(event["dataset_idx1"])
        audio2 = self._get_audio_from_cache_or_dataset(event["dataset_idx2"])
        
        if audio1 is None or audio2 is None:
            raise ValueError(f"Failed to get audio for simultaneous event")

        volume1 = self.speaker_volumes.get(event["speaker1_idx"], 1.0)
        volume2 = self.speaker_volumes.get(event["speaker2_idx"], 1.0)

        audio1 = audio1[: event["duration_samples"]] * volume1
        audio2 = audio2[: event["duration_samples"]] * volume2

        mixed = (audio1 + audio2) / 2.0
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

