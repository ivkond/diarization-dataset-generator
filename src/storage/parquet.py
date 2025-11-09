"""Parquet storage utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List

import pyarrow as pa
import pyarrow.parquet as pq

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

        # Convert to Parquet format
        parquet_data = []
        for track in tracks:
            record = {
                "audio": track["audio"],
                "duration": track["duration"],
                "num_speakers": track["num_speakers"],
                "sampling_rate": track["sampling_rate"],
                "conversation_type": track["conversation_type"],
                "difficulty": track["difficulty"],
                "has_overlaps": track["has_overlaps"],
                "has_simultaneous": track["has_simultaneous"],
                "has_noise": track["has_noise"],
                "speakers": json.dumps(track["speakers"], ensure_ascii=False),
            }

            # Add optional fields
            if "noise_type" in track:
                record["noise_type"] = track["noise_type"]
            if "snr" in track:
                record["snr"] = track["snr"]
            if "speaker_volumes" in track:
                record["speaker_volumes"] = json.dumps(
                    track["speaker_volumes"], ensure_ascii=False
                )
            if "simultaneous_segments" in track:
                record["simultaneous_segments"] = json.dumps(
                    track["simultaneous_segments"], ensure_ascii=False
                )

            parquet_data.append(record)

        # Split into batches by size
        batches = self._split_into_batches(parquet_data)

        # Write batches
        total_files = len(batches)
        logger.info(f"Writing {total_files} Parquet file(s) (max {self.max_file_size_bytes / (1024*1024):.0f} MB per file)...")

        for file_index, batch in enumerate(batches):
            self._write_batch(batch, file_index, total_files)

        return total_files

    def _split_into_batches(self, records: List[Dict]) -> List[List[Dict]]:
        """Split records into batches by size."""
        batches = []
        current_batch = []
        current_size = 0

        for record in records:
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

        schema = pa.schema([
            ("audio", pa.binary()),
            ("duration", pa.float64()),
            ("num_speakers", pa.int64()),
            ("sampling_rate", pa.int64()),
            ("conversation_type", pa.string()),
            ("difficulty", pa.string()),
            ("has_overlaps", pa.bool_()),
            ("has_simultaneous", pa.bool_()),
            ("has_noise", pa.bool_()),
            ("speakers", pa.string()),
            ("noise_type", pa.string()),
            ("snr", pa.float64()),
            ("speaker_volumes", pa.string()),
            ("simultaneous_segments", pa.string()),
        ])

        arrays = {
            "audio": [r["audio"] for r in batch],
            "duration": [r["duration"] for r in batch],
            "num_speakers": [r["num_speakers"] for r in batch],
            "sampling_rate": [r["sampling_rate"] for r in batch],
            "conversation_type": [r["conversation_type"] for r in batch],
            "difficulty": [r["difficulty"] for r in batch],
            "has_overlaps": [r["has_overlaps"] for r in batch],
            "has_simultaneous": [r["has_simultaneous"] for r in batch],
            "has_noise": [r["has_noise"] for r in batch],
            "speakers": [r["speakers"] for r in batch],
            "noise_type": [r.get("noise_type") for r in batch],
            "snr": [r.get("snr") for r in batch],
            "speaker_volumes": [r.get("speaker_volumes") for r in batch],
            "simultaneous_segments": [r.get("simultaneous_segments") for r in batch],
        }

        table = pa.table(arrays, schema=schema)
        pq.write_table(table, filepath, compression="snappy")

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
            # Convert track to Parquet format
            record = {
                "audio": track["audio"],
                "duration": track["duration"],
                "num_speakers": track["num_speakers"],
                "sampling_rate": track["sampling_rate"],
                "conversation_type": track["conversation_type"],
                "difficulty": track["difficulty"],
                "has_overlaps": track["has_overlaps"],
                "has_simultaneous": track["has_simultaneous"],
                "has_noise": track["has_noise"],
                "speakers": json.dumps(track["speakers"], ensure_ascii=False),
            }

            # Add optional fields
            if "noise_type" in track:
                record["noise_type"] = track["noise_type"]
            if "snr" in track:
                record["snr"] = track["snr"]
            if "speaker_volumes" in track:
                record["speaker_volumes"] = json.dumps(
                    track["speaker_volumes"], ensure_ascii=False
                )
            if "simultaneous_segments" in track:
                record["simultaneous_segments"] = json.dumps(
                    track["simultaneous_segments"], ensure_ascii=False
                )

            record_size = len(record["audio"]) + METADATA_OVERHEAD_BYTES

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

