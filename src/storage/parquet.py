"""Parquet storage utilities."""

import io
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from ..constants import METADATA_OVERHEAD_BYTES

logger = logging.getLogger(__name__)


class ParquetWriter:
    """Writer for saving tracks to Parquet files."""

    def __init__(self, output_dir: Path, max_file_size_mb: float):
        """
        Initialize Parquet writer.

        Args:
            output_dir: Output directory for Parquet files.
            max_file_size_mb: Maximum file size in MB.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def write_tracks(self, tracks: List[Dict]) -> int:
        """
        Write tracks to Parquet files, splitting by size.

        Args:
            tracks: List of track metadata dictionaries.

        Returns:
            Number of files written.
        """
        logger.info(f"Preparing {len(tracks)} tracks for Parquet storage...")

        # Convert to Parquet format (keep as structured data, not JSON)
        parquet_data = []
        for track in tracks:
            # Keep audio as raw bytes for Hugging Face Audio feature
            record = {
                "audio": track["audio"],  # Keep as raw WAV bytes
                "duration": track["duration"],
                "num_speakers": track["num_speakers"],
                "sampling_rate": track["sampling_rate"],
                "conversation_type": track["conversation_type"],
                "difficulty": track["difficulty"],
                "has_overlaps": track["has_overlaps"],
                "has_simultaneous": track["has_simultaneous"],
                "has_noise": track["has_noise"],
                "speakers": track["speakers"],  # Keep as structured data
            }

            # Add optional fields
            if "noise_type" in track:
                record["noise_type"] = track["noise_type"]
            if "snr" in track:
                record["snr"] = track["snr"]
            if "speaker_volumes" in track:
                record["speaker_volumes"] = track["speaker_volumes"]  # Keep as dict
            if "simultaneous_segments" in track:
                record["simultaneous_segments"] = track["simultaneous_segments"]  # Keep as list

            parquet_data.append(record)

        # Split into batches by size
        batches = self._split_into_batches(parquet_data)

        # Write batches
        total_files = len(batches)
        logger.info(f"Writing {total_files} Parquet file(s) (max {self.max_file_size_bytes / (1024*1024):.0f} MB per file)...")

        for file_index, batch in enumerate(batches):
            self._write_batch(batch, file_index, total_files)

        return total_files

    def _convert_audio_bytes_to_dict(self, audio_bytes: bytes, sampling_rate: int) -> Dict[str, Any]:
        """
        Convert WAV bytes to audio dict format expected by HuggingFace.

        Args:
            audio_bytes: Bytes of WAV file.
            sampling_rate: Sampling rate of the audio.

        Returns:
            dict with 'array' and 'sampling_rate' keys.
        """
        wav_buffer = io.BytesIO(audio_bytes)
        audio_array, sr = sf.read(wav_buffer)
        wav_buffer.close()

        # Ensure float32 format
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        return {
            "array": audio_array,  # Keep as numpy array for internal processing
            "sampling_rate": sr if sr else sampling_rate,
        }

    def _split_into_batches(self, records: List[Dict]) -> List[List[Dict]]:
        """Split records into batches by size."""
        batches = []
        current_batch = []
        current_size = 0

        for record in records:
            # Estimate size - audio dict is larger than bytes
            if isinstance(record["audio"], dict):
                # Python lists have significant overhead per element (pointers + object overhead)
                # Use conservative multiplier: ~3x the raw float32 size to account for:
                # - List pointer overhead (8 bytes per pointer on 64-bit systems)
                # - Python float object overhead (~24 bytes per float object)
                # - Parquet serialization overhead
                array_len = len(record["audio"].get("array", []))
                array_size = array_len * 12  # Conservative: ~12 bytes per element (3x float32)
                record_size = array_size + METADATA_OVERHEAD_BYTES
            else:
                record_size = len(record["audio"]) + METADATA_OVERHEAD_BYTES

            if current_batch and (current_size + record_size) > self.max_file_size_bytes:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

            current_batch.append(record)
            current_size += record_size

        if current_batch:
            batches.append(current_batch)

        return batches

    def _write_batch(self, batch: List[Dict], file_index: int, total_files: int | None = None) -> Path:
        """
        Write a batch to a Parquet file.
        
        Args:
            batch: List of records to write.
            file_index: Index of the file.
            total_files: Total number of files (None for incremental writes).
            
        Returns:
            Path to the written file.
        """
        if total_files is not None:
            filename = f"train-{file_index:05d}-of-{total_files:05d}.parquet"
        else:
            # For incremental writes, use temporary name
            filename = f"train-{file_index:05d}.tmp.parquet"
        filepath = self.output_dir / filename

        # Define structured schema for better compression
        speaker_struct = pa.struct([
            ("speaker_id", pa.int64()),
            ("start", pa.float64()),
            ("end", pa.float64()),
            ("duration", pa.float64()),
            ("text", pa.string()),
        ])
        
        simultaneous_struct = pa.struct([
            ("start", pa.float64()),
            ("end", pa.float64()),
            ("speaker1_id", pa.int64()),
            ("speaker2_id", pa.int64()),
            ("duration", pa.float64()),
        ])
        
        # Use struct for speaker_volumes instead of map (map is not supported by datasets)
        speaker_volume_struct = pa.struct([
            ("speaker_id", pa.int64()),
            ("volume", pa.float64()),
        ])
        
        # Audio struct for dict format
        # audio_struct = pa.struct([
        #     ("array", pa.list_(pa.float32())),  # Audio array as list of floats
        #     ("sampling_rate", pa.int64()),
        # ])
        
        # Audio struct for dict format expected by HuggingFace Audio feature
        audio_struct = pa.struct([
            ("array", pa.list_(pa.float32())),  # Audio array as list of floats
            ("sampling_rate", pa.int64()),
        ])

        schema = pa.schema([
            ("audio", audio_struct),  # Store as structured data for Hugging Face Audio feature
            ("duration", pa.float64()),
            ("num_speakers", pa.int64()),
            ("sampling_rate", pa.int64()),
            ("conversation_type", pa.string()),
            ("difficulty", pa.string()),
            ("has_overlaps", pa.bool_()),
            ("has_simultaneous", pa.bool_()),
            ("has_noise", pa.bool_()),
            ("speakers", pa.list_(speaker_struct)),
            ("noise_type", pa.string()),
            ("snr", pa.float64()),
            ("speaker_volumes", pa.list_(speaker_volume_struct)),
            ("simultaneous_segments", pa.list_(simultaneous_struct)),
        ])

        # Data already in correct format from generator - use directly
        # Only handle backward compatibility for old JSON format
        speakers_list = []
        speaker_volumes_list = []
        simultaneous_segments_list = []
        audio_list = []

        for r in batch:
            # Audio - convert to dict format expected by HuggingFace Audio feature
            audio_data = r.get("audio")
            if isinstance(audio_data, bytes):
                # Convert raw WAV bytes to array format for Hugging Face
                audio_dict = self._convert_audio_bytes_to_dict(
                    audio_data, r.get("sampling_rate", 16000)
                )
                audio_list.append({
                    "array": audio_dict["array"].tolist(),  # Convert to list for Parquet
                    "sampling_rate": audio_dict["sampling_rate"],
                })
            else:
                # Fallback
                audio_list.append({
                    "array": [],
                    "sampling_rate": r.get("sampling_rate", 16000),
                })

            # Speakers - already in correct format (list of dicts)
            speakers_data = r.get("speakers", [])
            if isinstance(speakers_data, str):
                # Backward compatibility: parse JSON if string
                speakers_data = json.loads(speakers_data)
            # Ensure text field exists (it's optional)
            speakers_structs = [
                {
                    "speaker_id": s["speaker_id"],
                    "start": s["start"],
                    "end": s["end"],
                    "duration": s["duration"],
                    "text": s.get("text", ""),
                }
                for s in speakers_data
            ]
            speakers_list.append(speakers_structs)

            # Speaker volumes - already in correct format (list of structs) from generator
            volumes_data = r.get("speaker_volumes", [])
            if isinstance(volumes_data, str):
                # Backward compatibility: parse JSON if string
                volumes_data = json.loads(volumes_data)
            # If it's still a dict (old format), convert it
            if isinstance(volumes_data, dict):
                volumes_data = [
                    {"speaker_id": int(sid), "volume": float(vol)}
                    for sid, vol in volumes_data.items()
                ]
            speaker_volumes_list.append(volumes_data)

            # Simultaneous segments - already in correct format (list of dicts)
            sim_data = r.get("simultaneous_segments", [])
            if isinstance(sim_data, str):
                # Backward compatibility: parse JSON if string
                sim_data = json.loads(sim_data)
            simultaneous_segments_list.append(sim_data)

        arrays = {
            "audio": audio_list,  # Now using raw bytes as expected by Hugging Face
            "duration": [r["duration"] for r in batch],
            "num_speakers": [r["num_speakers"] for r in batch],
            "sampling_rate": [r["sampling_rate"] for r in batch],
            "conversation_type": [r["conversation_type"] for r in batch],
            "difficulty": [r["difficulty"] for r in batch],
            "has_overlaps": [r["has_overlaps"] for r in batch],
            "has_simultaneous": [r["has_simultaneous"] for r in batch],
            "has_noise": [r["has_noise"] for r in batch],
            "speakers": speakers_list,
            "noise_type": [r.get("noise_type") for r in batch],
            "snr": [r.get("snr") for r in batch],
            "speaker_volumes": speaker_volumes_list,
            "simultaneous_segments": simultaneous_segments_list,
        }

        table = pa.table(arrays, schema=schema)
        # Use zstd compression with high compression level for better size reduction
        pq.write_table(
            table, 
            filepath, 
            compression="zstd",
            compression_level=9,
            row_group_size=1000,
            use_dictionary=True,
        )

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"  Written {filename}: {len(batch)} records, {file_size_mb:.2f} MB")
        return filepath

    def write_tracks_incremental(self, tracks_iterator: Iterator[Dict]) -> int:
        """
        Write tracks to Parquet files incrementally as they are generated.

        Args:
            tracks_iterator: Iterator of track metadata dictionaries.

        Returns:
            Number of files written.
        """
        current_batch = []
        current_size = 0
        file_index = 0
        total_tracks = 0
        written_files = []  # Store paths to written files for renaming

        logger.info("Writing tracks incrementally to Parquet files...")

        for track in tracks_iterator:
            # Keep audio as raw bytes for Hugging Face Audio feature
            # No conversion needed - keep as wav_bytes from generator
            record = {
                "audio": track["audio"],  # Keep as raw WAV bytes
                "duration": track["duration"],
                "num_speakers": track["num_speakers"],
                "sampling_rate": track["sampling_rate"],
                "conversation_type": track["conversation_type"],
                "difficulty": track["difficulty"],
                "has_overlaps": track["has_overlaps"],
                "has_simultaneous": track["has_simultaneous"],
                "has_noise": track["has_noise"],
                "speakers": track["speakers"],  # Keep as structured data
            }

            # Add optional fields
            if "noise_type" in track:
                record["noise_type"] = track["noise_type"]
            if "snr" in track:
                record["snr"] = track["snr"]
            if "speaker_volumes" in track:
                record["speaker_volumes"] = track["speaker_volumes"]  # Keep as dict
            if "simultaneous_segments" in track:
                record["simultaneous_segments"] = track["simultaneous_segments"]  # Keep as list

            # Estimate size - audio as raw bytes
            audio_data = record["audio"]
            if isinstance(audio_data, bytes):
                record_size = len(audio_data) + METADATA_OVERHEAD_BYTES
            else:
                record_size = 50000  # Fallback estimate

            # Check if adding this record would exceed the size limit
            if current_batch and (current_size + record_size) > self.max_file_size_bytes:
                # Write current batch
                filepath = self._write_batch(current_batch, file_index, None)
                written_files.append(filepath)
                file_index += 1
                current_batch = []
                current_size = 0

            current_batch.append(record)
            current_size += record_size
            total_tracks += 1

        # Write remaining batch
        if current_batch:
            filepath = self._write_batch(current_batch, file_index, None)
            written_files.append(filepath)
            file_index += 1

        total_files = file_index

        # Rename temporary files to final names
        if written_files:
            logger.info(f"Renaming {len(written_files)} temporary file(s)...")
            for idx, temp_path in enumerate(written_files):
                final_filename = f"train-{idx:05d}-of-{total_files:05d}.parquet"
                final_path = self.output_dir / final_filename
                temp_path.rename(final_path)
                logger.debug(f"  Renamed {temp_path.name} -> {final_filename}")

        logger.info(f"Incremental write complete: {total_tracks} tracks in {total_files} file(s)")
        return total_files

