"""Conversation pattern implementations."""

from .base import ConversationPattern
from .dialogue import DialoguePattern
from .group_discussion import GroupDiscussionPattern
from .interview import InterviewPattern
from .monologue import MonologuePattern
from .selector import PatternSelector

__all__ = [
    "ConversationPattern",
    "MonologuePattern",
    "DialoguePattern",
    "InterviewPattern",
    "GroupDiscussionPattern",
    "PatternSelector",
]

