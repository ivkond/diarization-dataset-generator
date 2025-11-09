"""Dialogue conversation pattern."""

from typing import List

from .base import ConversationPattern


class DialoguePattern(ConversationPattern):
    """Dialogue pattern: uniform alternation between two speakers."""

    def select_next_speaker(
        self, available_speakers: List[int], max_consecutive: int
    ) -> int:
        """
        Select next speaker (alternate between two speakers).

        Args:
            available_speakers: List of available speaker indices.
            max_consecutive: Maximum consecutive segments (ignored).

        Returns:
            Selected speaker index (alternating between 0 and 1).
        """
        if len(self.used_speakers) < len(available_speakers):
            # First use all speakers
            unused_speakers = [i for i in available_speakers if i not in self.used_speakers]
            speaker_idx = (
                unused_speakers[0]
                if unused_speakers
                else (
                    1 - self.previous_speaker_idx
                    if self.previous_speaker_idx is not None
                    else 0
                )
            )
        else:
            # Alternate between two speakers
            if self.previous_speaker_idx is None:
                speaker_idx = 0
            else:
                speaker_idx = 1 - self.previous_speaker_idx

        self.previous_speaker_idx = speaker_idx
        self.consecutive_count = 1
        self.used_speakers.add(speaker_idx)
        return speaker_idx

