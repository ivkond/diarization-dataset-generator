"""Base class for conversation patterns."""

from abc import ABC, abstractmethod
from typing import List, Optional


class ConversationPattern(ABC):
    """Base class for conversation patterns."""

    def __init__(self, num_speakers: int):
        """
        Initialize conversation pattern.

        Args:
            num_speakers: Number of speakers in the conversation.
        """
        self.num_speakers = num_speakers
        self.previous_speaker_idx: Optional[int] = None
        self.consecutive_count = 0
        self.used_speakers: set = set()

    @abstractmethod
    def select_next_speaker(
        self, available_speakers: List[int], max_consecutive: int
    ) -> int:
        """
        Select next speaker based on pattern.

        Args:
            available_speakers: List of available speaker indices.
            max_consecutive: Maximum consecutive segments for same speaker.

        Returns:
            Selected speaker index.
        """
        pass

    def reset(self) -> None:
        """Reset pattern state."""
        self.previous_speaker_idx = None
        self.consecutive_count = 0
        self.used_speakers = set()

