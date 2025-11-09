"""Interview conversation pattern."""

import random
from typing import List

from ..constants import INTERVIEW_QUESTION_PROBABILITY
from .base import ConversationPattern


class InterviewPattern(ConversationPattern):
    """Interview pattern: interviewer asks short questions, interviewee gives long answers."""

    def __init__(self, num_speakers: int):
        """
        Initialize interview pattern.

        Args:
            num_speakers: Number of speakers (should be 2 for interview).
        """
        super().__init__(num_speakers)
        self.interviewer_idx = 0 if num_speakers >= 2 else None

    def select_next_speaker(
        self, available_speakers: List[int], max_consecutive: int
    ) -> int:
        """
        Select next speaker (interviewer asks, interviewee answers).

        Args:
            available_speakers: List of available speaker indices.
            max_consecutive: Maximum consecutive segments.

        Returns:
            Selected speaker index.
        """
        if len(self.used_speakers) < len(available_speakers):
            unused_speakers = [i for i in available_speakers if i not in self.used_speakers]
            speaker_idx = (
                unused_speakers[0] if unused_speakers else self.interviewer_idx
            )
        else:
            # Alternate, but interviewer speaks more often and shorter
            if self.previous_speaker_idx == self.interviewer_idx:
                # After interviewer's question - interviewee's answer
                speaker_idx = 1 - self.interviewer_idx
                self.consecutive_count = 1
            else:
                # After answer - question again (with probability)
                if random.random() < INTERVIEW_QUESTION_PROBABILITY:
                    speaker_idx = self.interviewer_idx
                    self.consecutive_count = 1
                else:
                    # Or continuation of answer
                    speaker_idx = self.previous_speaker_idx
                    self.consecutive_count += 1

        self.previous_speaker_idx = speaker_idx
        self.used_speakers.add(speaker_idx)
        return speaker_idx

