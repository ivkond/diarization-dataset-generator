"""Group discussion conversation pattern."""

import random
from typing import List

from ..constants import GROUP_DISCUSSION_CONTINUE_PROBABILITY
from .base import ConversationPattern


class GroupDiscussionPattern(ConversationPattern):
    """Group discussion pattern: frequent switching between speakers."""

    def select_next_speaker(
        self, available_speakers: List[int], max_consecutive: int
    ) -> int:
        """
        Select next speaker (frequent switching, rarely same speaker consecutively).

        Args:
            available_speakers: List of available speaker indices.
            max_consecutive: Maximum consecutive segments.

        Returns:
            Selected speaker index.
        """
        if len(self.used_speakers) < len(available_speakers):
            unused_speakers = [i for i in available_speakers if i not in self.used_speakers]
            speaker_idx = (
                unused_speakers[0]
                if unused_speakers
                else random.choice(available_speakers)
            )
            self.previous_speaker_idx = speaker_idx
            self.consecutive_count = 1
            self.used_speakers.add(speaker_idx)
        else:
            # Frequent switching, rarely one speaker speaks consecutively
            if (
                self.previous_speaker_idx is not None
                and self.consecutive_count < 2
                and random.random() < GROUP_DISCUSSION_CONTINUE_PROBABILITY
            ):
                speaker_idx = self.previous_speaker_idx
                self.consecutive_count += 1
            else:
                available = [
                    i
                    for i in available_speakers
                    if i != self.previous_speaker_idx or self.previous_speaker_idx is None
                ]
                speaker_idx = random.choice(available)
                if speaker_idx == self.previous_speaker_idx:
                    self.consecutive_count += 1
                else:
                    self.consecutive_count = 1
                    self.previous_speaker_idx = speaker_idx

        return speaker_idx

