"""Track quality validation utilities."""

from typing import Dict, List, Tuple

from ..constants import DEFAULT_MAX_IMBALANCE_RATIO, DEFAULT_MIN_SPEAKER_DURATION


class TrackValidator:
    """Validator for track quality."""

    def __init__(
        self,
        min_speaker_duration: float = DEFAULT_MIN_SPEAKER_DURATION,
        max_imbalance_ratio: float = DEFAULT_MAX_IMBALANCE_RATIO,
    ):
        """
        Initialize validator.

        Args:
            min_speaker_duration: Minimum duration for each speaker (seconds).
            max_imbalance_ratio: Maximum ratio of dominant speaker time to total.
        """
        self.min_speaker_duration = min_speaker_duration
        self.max_imbalance_ratio = max_imbalance_ratio

    def validate(
        self, metadata_segments: List[Dict], num_speakers: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate track quality.

        Args:
            metadata_segments: List of segments with metadata.
            num_speakers: Expected number of speakers.

        Returns:
            Tuple of (is_valid, warnings).
        """
        warnings: List[str] = []

        # Calculate duration of each speaker
        speaker_durations: Dict[int, float] = {}
        total_duration = 0.0

        for segment in metadata_segments:
            speaker_id = segment["speaker_id"]
            duration = segment.get("duration", segment["end"] - segment["start"])

            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0.0
            speaker_durations[speaker_id] += duration
            total_duration += duration

        # Check minimum duration of each speaker
        for speaker_id, duration in speaker_durations.items():
            if duration < self.min_speaker_duration:
                warnings.append(
                    f"Speaker {speaker_id} has duration {duration:.2f}s < {self.min_speaker_duration}s"
                )

        # Check time balance between speakers
        if len(speaker_durations) > 0 and total_duration > 0:
            max_duration = max(speaker_durations.values())
            max_ratio = max_duration / total_duration

            if max_ratio > self.max_imbalance_ratio:
                warnings.append(
                    f"Speaker imbalance: max ratio {max_ratio:.2f} > {self.max_imbalance_ratio}"
                )

        # Check that all speakers are represented
        if len(speaker_durations) < num_speakers:
            warnings.append(
                f"Only {len(speaker_durations)} speakers found, expected {num_speakers}"
            )

        is_valid = len(warnings) == 0
        return is_valid, warnings

