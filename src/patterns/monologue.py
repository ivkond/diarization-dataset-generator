"""Monologue conversation pattern."""

from typing import List

from .base import ConversationPattern


class MonologuePattern(ConversationPattern):
    """Monologue pattern: always the same speaker."""

    def select_next_speaker(
        self, available_speakers: List[int], max_consecutive: int
    ) -> int:
        """
        Select next speaker (always speaker 0 for monologue).

        Args:
            available_speakers: List of available speaker indices (ignored).
            max_consecutive: Maximum consecutive segments (ignored).

        Returns:
            Always returns 0 (first speaker).
        """
        speaker_idx = 0
        if self.previous_speaker_idx != speaker_idx:
            self.consecutive_count = 1
            self.previous_speaker_idx = speaker_idx
        else:
            self.consecutive_count += 1
        return speaker_idx

