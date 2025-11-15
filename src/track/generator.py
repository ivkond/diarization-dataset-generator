"""Main track generator module."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from ..audio.encoder import encode_to_wav_bytes
from ..audio.noise import add_background_noise
from ..audio.processor import extract_audio_array, normalize_audio
from ..constants import SAMPLING_RATE
from ..exceptions import TrackGenerationError
from ..patterns.base import ConversationPattern
from .builder import AudioBuilder
from .planner import TrackPlanner
from .types import SegmentMetadata, SimultaneousSegmentMetadata, SpeakerAudioPool
from .utils import calculate_difficulty
from .validator import TrackValidator

logger = logging.getLogger(__name__)


class TrackGenerator:
    """Main generator for creating audio tracks."""

    def __init__(
        self,
        dataset: Dataset,
        config: Any,  # Config from config.models
        speaker_index: Dict[str, List[int]],
        metadata_cache: Dict[int, Tuple[str, str]],
        pattern_selector: Any,  # PatternSelector
        audio_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Initialize track generator.

        Args:
            dataset: Dataset to use for audio segments.
            config: Configuration object.
            speaker_index: Speaker index (speaker_id -> list of indices).
            metadata_cache: Metadata cache (index -> (speaker_id, text)).
            pattern_selector: Pattern selector instance.
            audio_cache: Optional audio cache dictionary (dataset_idx -> audio_array).
        """
        self.dataset = dataset
        self.config = config
        self.speaker_index = speaker_index
        self.metadata_cache = metadata_cache
        self.pattern_selector = pattern_selector
        self.audio_cache = audio_cache if audio_cache is not None else {}

    def _get_audio_from_cache_or_dataset(self, dataset_idx: int) -> Optional[np.ndarray]:
        """
        Get audio array from cache or dataset.
        
        Args:
            dataset_idx: Dataset index.
            
        Returns:
            Audio array or None if extraction fails.
        """
        # Check cache first
        if dataset_idx in self.audio_cache:
            return self.audio_cache[dataset_idx]
        
        # Extract from dataset
        try:
            sample = self.dataset[dataset_idx]
            audio_array = extract_audio_array(sample[self.config.dataset.feature_audio])
            # Store in cache for future use
            self.audio_cache[dataset_idx] = audio_array
            return audio_array
        except Exception as e:
            logger.debug(f"Failed to extract audio from dataset index {dataset_idx}: {e}")
            return None

    def generate(
        self, track_id: int
    ) -> Dict[str, Any]:
        """
        Generate a single audio track.

        Args:
            track_id: Track identifier.

        Returns:
            Dictionary with track metadata and audio bytes.

        Raises:
            TrackGenerationError: If track generation fails.
        """
        try:
            # Select conversation pattern
            pattern_name = self.pattern_selector.select_pattern()
            num_speakers = self.pattern_selector.get_speaker_count_for_pattern(pattern_name)
            pattern = self.pattern_selector.create_pattern(pattern_name, num_speakers)

            # Random duration
            target_duration = random.uniform(
                self.config.track.min_duration, self.config.track.max_duration
            )

            # Determine features
            use_short_segments = random.random() < self.config.track.short_segment.probability
            will_have_simultaneous = (
                random.random() < self.config.simultaneous_speech.probability
            )
            will_have_noise = random.random() < self.config.noise.probability
            snr_db = (
                random.uniform(self.config.noise.snr.min, self.config.noise.snr.max)
                if will_have_noise
                else None
            )

            logger.info(f"Generating track {track_id}:")
            logger.info(f"  Pattern: {pattern_name}")
            logger.info(f"  Target duration: {target_duration:.2f} seconds")
            logger.info(f"  Number of speakers: {num_speakers}")
            logger.info(f"  Short segments: {use_short_segments}")
            logger.info(f"  Simultaneous speech: {will_have_simultaneous}")
            if will_have_noise:
                logger.info(f"  Background noise: True (SNR: {snr_db:.1f} dB)")
            else:
                logger.info("  Background noise: False")

            # Create speaker audio pools
            speaker_pools, speaker_volumes = self._create_speaker_pools(
                num_speakers, use_short_segments
            )

            if not speaker_pools:
                raise TrackGenerationError("Failed to create speaker pools")

            # Create planner
            planner = TrackPlanner(
                self.dataset,
                self.config,
                pattern,
                speaker_pools,
                speaker_volumes,
                self.metadata_cache,
                self.speaker_index,
                self.audio_cache,
            )

            # Create plan
            plan, metadata_segments, simultaneous_segments, has_overlaps, has_simultaneous = (
                planner.create_plan(target_duration, use_short_segments, will_have_simultaneous)
            )

            logger.info(f"  Building audio from plan ({len(plan)} events)...")

            # Build audio
            builder = AudioBuilder(self.dataset, speaker_volumes, self.config, self.audio_cache)
            final_audio = builder.build_audio(plan)

            # Count actual speakers
            actual_speaker_ids = set(seg["speaker_id"] for seg in metadata_segments)
            actual_num_speakers = len(actual_speaker_ids)

            if actual_num_speakers != num_speakers:
                logger.warning(
                    f"  Expected {num_speakers} speakers, but only {actual_num_speakers} appeared"
                )
                logger.warning(f"  Actual speaker IDs: {sorted(actual_speaker_ids)}")
                num_speakers = actual_num_speakers

            # Add background noise if needed
            noise_type_used = None
            actual_snr = None
            if will_have_noise and snr_db is not None:
                final_audio, noise_type_used, actual_snr = add_background_noise(
                    final_audio, snr_db, noise_types=self.config.noise.types
                )
                logger.info(f"  Added {noise_type_used} noise with SNR: {actual_snr:.1f} dB")

            # Normalize audio
            final_audio = normalize_audio(final_audio)

            # Encode to WAV
            actual_duration = len(final_audio) / SAMPLING_RATE
            logger.info(f"  Actual duration: {actual_duration:.2f} seconds")

            wav_bytes = encode_to_wav_bytes(final_audio, SAMPLING_RATE)
            logger.info(f"  Encoded to WAV: {len(wav_bytes) / 1024:.2f} KB")

            # Calculate difficulty
            avg_segment_duration = (
                sum(seg.get("duration", seg["end"] - seg["start"]) for seg in metadata_segments)
                / len(metadata_segments)
                if metadata_segments
                else 0.0
            )
            difficulty = calculate_difficulty(
                num_speakers,
                has_overlaps,
                has_simultaneous,
                will_have_noise,
                avg_segment_duration,
            )

            logger.info(f"  Actual number of speakers: {num_speakers}")
            logger.info(f"  Difficulty: {difficulty}")

            # Validate quality
            validator = TrackValidator()
            is_valid, warnings = validator.validate(metadata_segments, num_speakers)
            if warnings:
                logger.warning("  Quality warnings:")
                for warning in warnings:
                    logger.warning(f"    - {warning}")

            # Build metadata
            track_metadata = self._build_metadata(
                wav_bytes,
                final_audio,
                pattern_name,
                num_speakers,
                difficulty,
                has_overlaps,
                has_simultaneous,
                will_have_noise,
                metadata_segments,
                speaker_volumes,
                simultaneous_segments,
                noise_type_used,
                actual_snr,
            )

            return track_metadata

        except Exception as e:
            raise TrackGenerationError(f"Failed to generate track {track_id}: {e}") from e

    def _create_speaker_pools(
        self, num_speakers: int, use_short_segments: bool
    ) -> Tuple[List[SpeakerAudioPool], Dict[int, float]]:
        """Create audio pools for each speaker."""
        available_indices = list(range(len(self.dataset)))
        selected_indices = random.sample(available_indices, num_speakers)

        speaker_pools: List[SpeakerAudioPool] = []
        speaker_volumes: Dict[int, float] = {}

        for speaker_idx, dataset_idx in enumerate(selected_indices):
            try:
                sample = self.dataset[dataset_idx]
                target_dataset_speaker_id = sample[self.config.dataset.feature_speaker_id]
                # Use cache for audio extraction
                audio_array = self._get_audio_from_cache_or_dataset(dataset_idx)

                if audio_array is None or len(audio_array) == 0:
                    logger.warning(
                        f"  Audio at index {dataset_idx} is empty, skipping speaker..."
                    )
                    continue

                # Generate random volume
                speaker_volume = random.uniform(
                    self.config.speakers.volume.min, self.config.speakers.volume.max
                )
                speaker_volumes[speaker_idx] = speaker_volume

                # Create pool (store only index, not audio array)
                pool = SpeakerAudioPool(speaker_idx, target_dataset_speaker_id)
                pool.add_segment(dataset_idx)

                # Find matching segments
                matching_indices = []
                if (
                    target_dataset_speaker_id is not None
                    and target_dataset_speaker_id in self.speaker_index
                ):
                    matching_indices = [
                        idx
                        for idx in self.speaker_index[target_dataset_speaker_id]
                        if idx != dataset_idx
                    ]
                else:
                    matching_indices = [idx for idx in available_indices if idx != dataset_idx]

                random.shuffle(matching_indices)

                # Filter by duration if needed
                if use_short_segments:
                    short_segment_indices = []
                    for idx in matching_indices:
                        # Use cache for audio extraction
                        audio_check = self._get_audio_from_cache_or_dataset(idx)
                        if audio_check is not None:
                            duration = len(audio_check) / SAMPLING_RATE
                            if (
                                duration <= self.config.track.short_segment.max_duration
                                and len(audio_check) > 0
                            ):
                                short_segment_indices.append(idx)

                    if short_segment_indices:
                        matching_indices = short_segment_indices
                        logger.info(
                            f"    Speaker {speaker_idx + 1}: using {len(short_segment_indices)} short segments"
                        )

                # Add segments
                target_count = min(self.config.speakers.target_count, len(matching_indices))
                added_count = 0

                for other_idx in matching_indices[:target_count]:
                    # Use cache to check if audio is valid
                    other_array = self._get_audio_from_cache_or_dataset(other_idx)
                    if other_array is not None and len(other_array) > 0:
                        # Store only index, not audio array
                        pool.add_segment(other_idx)
                        added_count += 1

                if added_count == 0:
                    logger.warning(
                        f"    Speaker {speaker_idx + 1}: found no additional segments"
                    )
                elif added_count < 10:
                    logger.info(
                        f"    Speaker {speaker_idx + 1}: found {added_count + 1} segments"
                    )

                speaker_pools.append(pool)

            except Exception as e:
                logger.error(f"  Error creating pool for speaker {speaker_idx}: {e}")
                continue

        return speaker_pools, speaker_volumes

    def _build_metadata(
        self,
        wav_bytes: bytes,
        final_audio: np.ndarray,
        pattern_name: str,
        num_speakers: int,
        difficulty: str,
        has_overlaps: bool,
        has_simultaneous: bool,
        has_noise: bool,
        metadata_segments: List[SegmentMetadata],
        speaker_volumes: Dict[int, float],
        simultaneous_segments: List[SimultaneousSegmentMetadata],
        noise_type_used: Optional[str],
        actual_snr: Optional[float],
    ) -> Dict[str, Any]:
        """Build track metadata dictionary."""
        track_metadata = {
            "audio": wav_bytes,
            "duration": round(len(final_audio) / SAMPLING_RATE, 2),
            "num_speakers": num_speakers,
            "sampling_rate": SAMPLING_RATE,
            "conversation_type": pattern_name,
            "difficulty": difficulty,
            "has_overlaps": has_overlaps,
            "has_simultaneous": has_simultaneous,
            "has_noise": has_noise,
            "speakers": metadata_segments,
        }

        if has_noise and noise_type_used is not None:
            track_metadata["noise_type"] = noise_type_used
            if actual_snr is not None:
                track_metadata["snr"] = round(actual_snr, 2)

        speaker_volumes_dict = {
            speaker_idx + 1: round(volume, 2)
            for speaker_idx, volume in speaker_volumes.items()
        }
        track_metadata["speaker_volumes"] = speaker_volumes_dict

        if simultaneous_segments:
            track_metadata["simultaneous_segments"] = simultaneous_segments

        return track_metadata

