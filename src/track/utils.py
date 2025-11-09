"""Utility functions for track generation."""

from typing import List


def calculate_difficulty(
    num_speakers: int,
    has_overlaps: bool,
    has_simultaneous: bool,
    has_noise: bool,
    avg_segment_duration: float,
) -> str:
    """
    Calculate track difficulty based on a simple formula.

    Args:
        num_speakers: Number of speakers.
        has_overlaps: Whether there are overlaps.
        has_simultaneous: Whether there is simultaneous speech.
        has_noise: Whether there is noise.
        avg_segment_duration: Average segment duration.

    Returns:
        'easy', 'medium' or 'hard'.
    """
    difficulty_score = num_speakers
    if has_overlaps:
        difficulty_score += 1
    if has_simultaneous:
        difficulty_score += 1
    if has_noise:
        difficulty_score += 1
    if avg_segment_duration < 2.0:
        difficulty_score += 1

    # Determine difficulty level
    if difficulty_score <= 2:
        return "easy"
    elif difficulty_score <= 4:
        return "medium"
    else:
        return "hard"

