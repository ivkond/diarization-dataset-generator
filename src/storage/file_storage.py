"""File storage utilities for WAV files and JSONL metadata."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Any

from ..constants import SAMPLING_RATE

logger = logging.getLogger(__name__)


class FileStorageWriter:
    """Writer for saving tracks to WAV files and JSONL metadata."""

    def __init__(self, output_dir: Path):
        """
        Initialize file storage writer.

        Args:
            output_dir: Output directory for audio files and metadata.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create audio subdirectory
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Metadata file path
        self.metadata_file = self.output_dir / "metadata.jsonl"
        
        # Track counter for sequential naming
        self.track_counter = 0

    def write_tracks_incremental(self, tracks_iterator: Iterator[Dict]) -> int:
        """
        Write tracks to files incrementally as they are generated.

        Args:
            tracks_iterator: Iterator of track metadata dictionaries.

        Returns:
            Number of tracks written.
        """
        logger.info("Writing tracks incrementally to files...")
        
        total_tracks = 0
        
        # Open metadata file for appending
        with open(self.metadata_file, "w", encoding="utf-8") as metadata_f:
            for track in tracks_iterator:
                try:
                    # Generate sequential filename
                    track_filename = f"track-{self.track_counter:05d}.wav"
                    audio_path = self.audio_dir / track_filename
                    relative_audio_path = f"audio/{track_filename}"
                    
                    # Write audio file
                    audio_bytes = track.get("audio")
                    if not isinstance(audio_bytes, bytes):
                        logger.error(f"Track {self.track_counter}: audio is not bytes, skipping")
                        continue
                    
                    with open(audio_path, "wb") as audio_f:
                        audio_f.write(audio_bytes)
                    
                    # Prepare metadata (remove audio bytes, add path)
                    metadata = self._prepare_metadata(track, relative_audio_path)
                    
                    # Write metadata line to JSONL
                    json_line = json.dumps(metadata, ensure_ascii=False)
                    metadata_f.write(json_line + "\n")
                    metadata_f.flush()  # Ensure data is written immediately
                    
                    total_tracks += 1
                    self.track_counter += 1
                    
                    if total_tracks % 100 == 0:
                        logger.info(f"  Written {total_tracks} tracks...")
                        
                except Exception as e:
                    logger.error(f"Failed to write track {self.track_counter}: {e}")
                    continue
        
        logger.info(f"Incremental write complete: {total_tracks} tracks written")
        logger.info(f"Audio files: {self.audio_dir.absolute()}")
        logger.info(f"Metadata file: {self.metadata_file.absolute()}")
        
        return total_tracks

    def _prepare_metadata(self, track: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
        """
        Prepare metadata dictionary for JSONL output.
        
        Args:
            track: Original track metadata dictionary.
            audio_path: Relative path to audio file.
            
        Returns:
            Metadata dictionary with audio_path instead of audio bytes.
        """
        metadata = {
            "audio_path": audio_path,
            "duration": track.get("duration"),
            "num_speakers": track.get("num_speakers"),
            "sampling_rate": track.get("sampling_rate", SAMPLING_RATE),
            "conversation_type": track.get("conversation_type"),
            "difficulty": track.get("difficulty"),
            "has_overlaps": track.get("has_overlaps", False),
            "has_simultaneous": track.get("has_simultaneous", False),
            "has_noise": track.get("has_noise", False),
            "speakers": track.get("speakers", []),
        }
        
        # Add optional fields
        if "noise_type" in track:
            metadata["noise_type"] = track["noise_type"]
        if "snr" in track:
            metadata["snr"] = track["snr"]
        if "speaker_volumes" in track:
            metadata["speaker_volumes"] = track["speaker_volumes"]
        if "simultaneous_segments" in track:
            metadata["simultaneous_segments"] = track.get("simultaneous_segments", [])
        
        return metadata

