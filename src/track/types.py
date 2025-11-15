"""Type definitions for track generation."""

from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np


class SegmentEvent(TypedDict, total=False):
    """Event representing an audio segment."""

    type: str  # "segment"
    speaker_idx: int
    dataset_idx: int
    volume: float
    start_time: float
    original_duration: float
    trim_start: int  # Optional: samples to trim from start


class PauseEvent(TypedDict):
    """Event representing a pause."""

    type: str  # "pause"
    duration_samples: int


class OverlapEvent(TypedDict):
    """Event representing an overlap between segments."""

    type: str  # "overlap"
    overlap_percent: float


class SimultaneousEvent(TypedDict):
    """Event representing simultaneous speech."""

    type: str  # "simultaneous"
    speaker1_idx: int
    speaker2_idx: int
    dataset_idx1: int
    dataset_idx2: int
    duration_samples: int
    start_time: float


TrackEvent = Union[SegmentEvent, PauseEvent, OverlapEvent, SimultaneousEvent]


class SegmentMetadata(TypedDict, total=False):
    """Metadata for a single segment."""

    speaker_id: int
    start: float
    end: float
    duration: float
    text: str  # Optional


class SimultaneousSegmentMetadata(TypedDict):
    """Metadata for simultaneous speech segment."""

    start: float
    end: float
    speaker1_id: int
    speaker2_id: int
    duration: float


class SpeakerAudioPool:
    """Container for speaker audio segments."""

    def __init__(self, speaker_idx: int, dataset_speaker_id: str) -> None:
        """
        Initialize speaker audio pool.

        Args:
            speaker_idx: Index of speaker in track.
            dataset_speaker_id: Speaker ID from dataset.
        """
        self.speaker_idx = speaker_idx
        self.dataset_speaker_id = dataset_speaker_id
        self.segment_indices: List[int] = []  # List of dataset indices only (no audio arrays)

    def add_segment(self, dataset_index: int) -> None:
        """Add segment index to pool."""
        self.segment_indices.append(dataset_index)

    def get_available_segments(self, used_indices: set) -> List[int]:
        """Get segment indices not yet used."""
        return [idx for idx in self.segment_indices if idx not in used_indices]
    
    def get_audio(
        self, 
        dataset, 
        cache: Dict[int, np.ndarray], 
        dataset_idx: int,
        config: Any,
    ) -> Optional[np.ndarray]:
        """
        Get audio array for a dataset index using cache or dataset.
        
        Args:
            dataset: Dataset to use for audio extraction.
            cache: Audio cache dictionary.
            dataset_idx: Dataset index.
            config: Configuration object.
            
        Returns:
            Audio array or None if extraction fails.
        """
        # Check cache first
        if dataset_idx in cache:
            return cache[dataset_idx]
        
        # Extract from dataset
        try:
            from ..audio.processor import extract_audio_array
            sample = dataset[dataset_idx]
            audio_array = extract_audio_array(sample[config.dataset.feature_audio])
            # Store in cache for future use
            cache[dataset_idx] = audio_array
            return audio_array
        except Exception:
            return None

