"""Conversation pattern selector."""

import random
from typing import Dict

from .base import ConversationPattern
from .dialogue import DialoguePattern
from .group_discussion import GroupDiscussionPattern
from .interview import InterviewPattern
from .monologue import MonologuePattern


class PatternSelector:
    """Selector for conversation patterns based on probability distribution."""

    def __init__(self, pattern_probabilities: Dict[str, float]):
        """
        Initialize pattern selector.

        Args:
            pattern_probabilities: Dictionary mapping pattern names to probabilities.
        """
        self.pattern_probabilities = pattern_probabilities
        self._validate_probabilities()

    def _validate_probabilities(self) -> None:
        """Validate that probabilities sum to approximately 1.0."""
        total = sum(self.pattern_probabilities.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Pattern probabilities must sum to 1.0, got {total}"
            )

    def select_pattern(self) -> str:
        """
        Select conversation pattern based on probability distribution.

        Returns:
            Pattern type name ('monologue', 'dialogue', 'group_discussion', 'interview').
        """
        rand = random.random()
        cumulative = 0.0
        for pattern, prob in self.pattern_probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return pattern
        # Fallback to dialogue if something goes wrong
        return "dialogue"

    def create_pattern(self, pattern_name: str, num_speakers: int) -> ConversationPattern:
        """
        Create pattern instance for given pattern name.

        Args:
            pattern_name: Name of the pattern.
            num_speakers: Number of speakers.

        Returns:
            Pattern instance.

        Raises:
            ValueError: If pattern name is unknown.
        """
        if pattern_name == "monologue":
            return MonologuePattern(num_speakers)
        elif pattern_name == "dialogue":
            return DialoguePattern(num_speakers)
        elif pattern_name == "interview":
            return InterviewPattern(num_speakers)
        elif pattern_name == "group_discussion":
            return GroupDiscussionPattern(num_speakers)
        else:
            raise ValueError(f"Unknown pattern: {pattern_name}")

    def get_speaker_count_for_pattern(self, pattern_name: str) -> int:
        """
        Get number of speakers for a given pattern.

        Args:
            pattern_name: Name of the pattern.

        Returns:
            Number of speakers.
        """
        if pattern_name == "monologue":
            return 1
        elif pattern_name in ("dialogue", "interview"):
            return 2
        elif pattern_name == "group_discussion":
            return random.randint(3, 4)
        else:
            return random.randint(2, 4)

