"""Type definitions for track generation."""

from typing import List, Tuple, TypedDict, Union

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
        self.segments: List[Tuple[np.ndarray, int]] = []  # List of (audio_array, dataset_index) tuples

    def add_segment(self, audio_array: np.ndarray, dataset_index: int) -> None:
        """Add audio segment to pool."""
        self.segments.append((audio_array, dataset_index))

    def get_available_segments(self, used_indices: set) -> List[Tuple[np.ndarray, int]]:
        """Get segments not yet used."""
        return [(audio, idx) for audio, idx in self.segments if idx not in used_indices]

