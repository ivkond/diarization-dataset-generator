"""Track planning module - creates sequence of events for track generation."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from ..audio.processor import extract_audio_array
from ..constants import SAMPLING_RATE
from ..patterns.base import ConversationPattern
from .types import (
    OverlapEvent,
    PauseEvent,
    SegmentEvent,
    SegmentMetadata,
    SimultaneousEvent,
    SimultaneousSegmentMetadata,
    SpeakerAudioPool,
    TrackEvent,
)

logger = logging.getLogger(__name__)


class TrackPlanner:
    """Planner for creating track event sequences."""

    def __init__(
        self,
        dataset: Dataset,
        config: Any,  # Config from config.models
        pattern: ConversationPattern,
        speaker_pools: List[SpeakerAudioPool],
        speaker_volumes: Dict[int, float],
        metadata_cache: Dict[int, Tuple[str, str]],
        speaker_index: Dict[str, List[int]],
    ) -> None:
        """
        Initialize track planner.

        Args:
            dataset: Dataset to use for audio segments.
            config: Configuration object.
            pattern: Conversation pattern instance.
            speaker_pools: List of speaker audio pools.
            speaker_volumes: Dictionary mapping speaker index to volume.
            metadata_cache: Metadata cache (index -> (speaker_id, text)).
            speaker_index: Speaker index (speaker_id -> list of indices).
        """
        self.dataset = dataset
        self.config = config
        self.pattern = pattern
        self.speaker_pools = speaker_pools
        self.speaker_volumes = speaker_volumes
        self.metadata_cache = metadata_cache
        self.speaker_index = speaker_index

        # Planning state
        self.plan: List[TrackEvent] = []
        self.metadata_segments: List[SegmentMetadata] = []
        self.simultaneous_segments: List[SimultaneousSegmentMetadata] = []
        self.current_time = 0.0
        self.current_total_samples = 0
        self.used_segments_by_speaker: Dict[int, set] = {
            i: set() for i in range(len(speaker_pools))
        }
        self.has_overlaps = False
        self.has_simultaneous = False

    def create_plan(
        self,
        target_duration: float,
        use_short_segments: bool,
        will_have_simultaneous: bool,
    ) -> Tuple[List[TrackEvent], List[SegmentMetadata], List[SimultaneousSegmentMetadata], bool, bool]:
        """
        Create plan for track generation.

        This method creates a sequence of events (segments, pauses, overlaps, simultaneous speech)
        that will be used to build the final audio track.

        Args:
            target_duration: Target duration in seconds.
            use_short_segments: Whether to use short segments (currently not used in planning).
            will_have_simultaneous: Whether to include simultaneous speech events.

        Returns:
            Tuple containing:
                - plan: List of track events
                - metadata_segments: List of segment metadata
                - simultaneous_segments: List of simultaneous speech metadata
                - has_overlaps: Whether overlaps were added
                - has_simultaneous: Whether simultaneous speech was added
        """
        target_length_samples = int(target_duration * SAMPLING_RATE)
        available_speakers = list(range(len(self.speaker_pools)))

        while self.current_total_samples < target_length_samples:
            # Select next speaker using pattern
            speaker_pool_idx = self.pattern.select_next_speaker(
                available_speakers, self.config.speakers.max_consecutive
            )

            # Get speaker pool and select segment
            speaker_pool = self.speaker_pools[speaker_pool_idx]
            used_indices = self.used_segments_by_speaker[speaker_pool_idx]
            available_segments = speaker_pool.get_available_segments(used_indices)

            if not available_segments:
                # All segments used, reset and reuse
                if len(used_indices) > 0:
                    logger.info(
                        f"Speaker {speaker_pool_idx + 1}: all {len(used_indices)} unique segments used, resetting"
                    )
                    self.used_segments_by_speaker[speaker_pool_idx].clear()
                available_segments = speaker_pool.segments

            audio_array, segment_dataset_index = random.choice(available_segments)
            self.used_segments_by_speaker[speaker_pool_idx].add(segment_dataset_index)

            # Get metadata
            if segment_dataset_index in self.metadata_cache:
                dataset_speaker_id, segment_text = self.metadata_cache[segment_dataset_index]
            else:
                sample = self.dataset[segment_dataset_index]
                dataset_speaker_id = sample[self.config.dataset.feature_speaker_id]
                segment_text = sample[self.config.dataset.feature_text]

            original_segment_duration = len(audio_array) / SAMPLING_RATE
            segment_start_time = self.current_time

            # Check for simultaneous speech
            is_simultaneous = False
            simultaneous_event: Optional[SimultaneousEvent] = None
            if (
                will_have_simultaneous
                and len(self.plan) > 0
                and len(self.speaker_pools) > 1
                and any(e.get("type") == "segment" for e in self.plan[-5:])
            ):
                simultaneous_event = self._create_simultaneous_event(
                    speaker_pool_idx, segment_dataset_index, target_length_samples
                )
                if simultaneous_event:
                    is_simultaneous = True
                    self.has_simultaneous = True
                    self.plan.append(simultaneous_event)
                    self.current_total_samples += simultaneous_event["duration_samples"]
                    simultaneous_duration_sec = (
                        simultaneous_event["duration_samples"] / SAMPLING_RATE
                    )
                    self.current_time += simultaneous_duration_sec

                    self.simultaneous_segments.append({
                        "start": round(simultaneous_event["start_time"], 2),
                        "end": round(self.current_time, 2),
                        "speaker1_id": speaker_pool_idx + 1,
                        "speaker2_id": simultaneous_event["speaker2_idx"] + 1,
                        "duration": round(simultaneous_duration_sec, 2),
                    })

            # Determine pause or overlap (if not simultaneous)
            trim_start = simultaneous_event["duration_samples"] if is_simultaneous else 0
            if not is_simultaneous and len(self.plan) > 0:
                pause_or_overlap = self._determine_pause_or_overlap(target_length_samples)
                if pause_or_overlap:
                    self.plan.append(pause_or_overlap)
                    if pause_or_overlap["type"] == "pause":
                        self.current_total_samples += pause_or_overlap["duration_samples"]
                        self.current_time += pause_or_overlap["duration_samples"] / SAMPLING_RATE

            # Add segment event
            if len(audio_array) > trim_start:
                segment_event = self._create_segment_event(
                    speaker_pool_idx,
                    segment_dataset_index,
                    segment_start_time,
                    original_segment_duration,
                    trim_start,
                )
                self.plan.append(segment_event)

                remaining_length = len(audio_array) - trim_start
                segment_duration = remaining_length / SAMPLING_RATE
                segment_end_time = self.current_time + segment_duration
                self.current_total_samples += remaining_length
                self.current_time = segment_end_time

                # Save metadata
                segment_metadata: SegmentMetadata = {
                    "speaker_id": speaker_pool_idx + 1,
                    "start": round(segment_start_time, 2),
                    "end": round(segment_end_time, 2),
                    "duration": round(original_segment_duration, 2),
                }
                if segment_text is not None:
                    segment_metadata["text"] = segment_text
                self.metadata_segments.append(segment_metadata)

            # Check if we exceeded target length
            if self.current_total_samples >= target_length_samples:
                break

        return (
            self.plan,
            self.metadata_segments,
            self.simultaneous_segments,
            self.has_overlaps,
            self.has_simultaneous,
        )

    def _create_simultaneous_event(
        self, speaker1_idx: int, dataset_idx1: int, target_length_samples: int
    ) -> Optional[SimultaneousEvent]:
        """Create simultaneous speech event."""
        other_speaker_idx = random.choice(
            [i for i in range(len(self.speaker_pools)) if i != speaker1_idx]
        )
        other_pool = self.speaker_pools[other_speaker_idx]

        if len(other_pool.segments) == 0:
            return None

        other_audio, other_dataset_idx = random.choice(other_pool.segments)
        simultaneous_duration_sec = random.uniform(
            self.config.simultaneous_speech.min_duration,
            self.config.simultaneous_speech.max_duration,
        )
        simultaneous_samples = int(simultaneous_duration_sec * SAMPLING_RATE)

        # Get first speaker's audio length
        first_audio = extract_audio_array(
            self.dataset[dataset_idx1][self.config.dataset.feature_audio]
        )
        min_length = min(len(first_audio), len(other_audio))
        simultaneous_samples = min(simultaneous_samples, min_length)

        if simultaneous_samples > 0:
            return {
                "type": "simultaneous",
                "speaker1_idx": speaker1_idx,
                "speaker2_idx": other_speaker_idx,
                "dataset_idx1": dataset_idx1,
                "dataset_idx2": other_dataset_idx,
                "duration_samples": simultaneous_samples,
                "start_time": self.current_time,
            }
        return None

    def _determine_pause_or_overlap(self, target_length_samples: int) -> Optional[TrackEvent]:
        """Determine whether to add pause or overlap."""
        # Check if last event was a segment
        last_segment_event = None
        for i in range(len(self.plan) - 1, -1, -1):
            if self.plan[i].get("type") == "segment":
                last_segment_event = self.plan[i]
                break

        if last_segment_event is not None and random.random() < self.config.overlaps.probability:
            # Add overlap event
            self.has_overlaps = True
            overlap_percent = random.uniform(
                self.config.overlaps.min_percent, self.config.overlaps.max_percent
            )
            return OverlapEvent(type="overlap", overlap_percent=overlap_percent)
        else:
            # Add pause
            if random.random() < self.config.track.long_pause.probability:
                pause_duration_sec = random.uniform(
                    self.config.track.long_pause.min, self.config.track.long_pause.max
                )
                pause_duration_ms = pause_duration_sec * 1000.0
            else:
                pause_duration_ms = random.uniform(
                    self.config.track.pause.min_ms, self.config.track.pause.max_ms
                )

            pause_samples = int(pause_duration_ms * SAMPLING_RATE / 1000)

            # Check if pause will exceed target length
            if (
                self.current_total_samples + pause_samples
                > target_length_samples - 1000  # Leave some room
            ):
                remaining_samples = target_length_samples - self.current_total_samples
                if remaining_samples > 1000:
                    pause_samples = min(pause_samples, remaining_samples - 1000)
                else:
                    pause_samples = 0

            if pause_samples > 0:
                return PauseEvent(type="pause", duration_samples=pause_samples)

        return None

    def _create_segment_event(
        self,
        speaker_idx: int,
        dataset_idx: int,
        start_time: float,
        original_duration: float,
        trim_start: int,
    ) -> SegmentEvent:
        """Create segment event."""
        event: SegmentEvent = {
            "type": "segment",
            "speaker_idx": speaker_idx,
            "dataset_idx": dataset_idx,
            "volume": self.speaker_volumes.get(speaker_idx, 1.0),
            "start_time": start_time,
            "original_duration": original_duration,
        }
        if trim_start > 0:
            event["trim_start"] = trim_start
        return event

