"""HuggingFace Hub streaming uploader."""

import io
import logging
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Features, Value
from huggingface_hub import HfApi, create_repo

from ..constants import SAMPLING_RATE

logger = logging.getLogger(__name__)


class HubUploader:
    """Streaming uploader for HuggingFace Hub."""

    def __init__(
        self,
        repo_id: str,
        hf_token: str,
        private: bool = False,
        batch_size: int = 100,
        max_batch_size_mb: float = 100.0,
    ):
        """
        Initialize Hub uploader.

        Args:
            repo_id: HuggingFace repository ID (format: "username/dataset-name").
            hf_token: HuggingFace authentication token.
            private: Whether the dataset should be private.
            batch_size: Target number of examples per batch.
            max_batch_size_mb: Maximum batch size in MB before uploading.
        """
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.private = private
        self.batch_size = batch_size
        self.max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
        self.hf_api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            logger.info(f"Repository {repo_id} is ready")
        except Exception as e:
            logger.warning(f"Could not create/verify repository: {e}")

    def _convert_audio_bytes_to_audio(self, audio_bytes: bytes, sampling_rate: int) -> Dict[str, Any]:
        """
        Convert WAV bytes to audio array format expected by HuggingFace.

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
            "array": audio_array,
            "sampling_rate": sr if sr else sampling_rate,
        }

    def _convert_track_to_example(self, track: Dict[str, Any], convert_audio: bool = False) -> Dict[str, Any]:
        """
        Convert track metadata to HuggingFace dataset example format.

        Args:
            track: Track metadata dictionary.
            convert_audio: Whether to convert audio bytes to dict format (for final dataset).
                          If False, keeps audio as bytes (for Parquet).

        Returns:
            Example dictionary in HuggingFace format.
        """
        # For Parquet upload, keep audio as bytes
        # For final dataset conversion, convert to Audio dict
        if convert_audio and isinstance(track["audio"], bytes):
            audio_data = self._convert_audio_bytes_to_audio(
                track["audio"], track.get("sampling_rate", SAMPLING_RATE)
            )
        else:
            audio_data = track["audio"]  # Keep as bytes for Parquet
        
        example = {
            "audio": audio_data,
            "duration": track["duration"],
            "num_speakers": track["num_speakers"],
            "sampling_rate": track["sampling_rate"],
            "conversation_type": track["conversation_type"],
            "difficulty": track["difficulty"],
            "has_overlaps": track["has_overlaps"],
            "has_simultaneous": track["has_simultaneous"],
            "has_noise": track["has_noise"],
            "speakers": track["speakers"],
            "speaker_volumes": track.get("speaker_volumes", []),
            "simultaneous_segments": track.get("simultaneous_segments", []),
        }

        # Add optional fields
        if "noise_type" in track:
            example["noise_type"] = track["noise_type"]
        if "snr" in track:
            example["snr"] = track["snr"]

        return example

    def _get_features(self) -> Features:
        """Get dataset features schema."""
        from datasets import Sequence

        speaker_struct = {
            "speaker_id": Value("int64"),
            "start": Value("float64"),
            "end": Value("float64"),
            "duration": Value("float64"),
            "text": Value("string"),
        }

        simultaneous_struct = {
            "start": Value("float64"),
            "end": Value("float64"),
            "speaker1_id": Value("int64"),
            "speaker2_id": Value("int64"),
            "duration": Value("float64"),
        }

        speaker_volume_struct = {
            "speaker_id": Value("int64"),
            "volume": Value("float64"),
        }

        features = Features({
            "audio": Audio(sampling_rate=SAMPLING_RATE),
            "duration": Value("float64"),
            "num_speakers": Value("int64"),
            "sampling_rate": Value("int64"),
            "conversation_type": Value("string"),
            "difficulty": Value("string"),
            "has_overlaps": Value("bool"),
            "has_simultaneous": Value("bool"),
            "has_noise": Value("bool"),
            "speakers": Sequence(speaker_struct),
            "speaker_volumes": Sequence(speaker_volume_struct),
            "simultaneous_segments": Sequence(simultaneous_struct),
            "noise_type": Value("string"),
            "snr": Value("float64"),
        })

        return features

    def _create_parquet_schema(self) -> pa.Schema:
        """Create Parquet schema for batch upload."""
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
        
        speaker_volume_struct = pa.struct([
            ("speaker_id", pa.int64()),
            ("volume", pa.float64()),
        ])
        
        # Audio struct for dict format
        audio_struct = pa.struct([
            ("array", pa.list_(pa.float32())),  # Audio array as list of floats
            ("sampling_rate", pa.int64()),
        ])
        
        return pa.schema([
            ("audio", audio_struct),  # Changed from pa.binary() to audio_struct
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

    def _create_parquet_batch(self, batch: list[Dict[str, Any]], schema: pa.Schema) -> bytes:
        """
        Create Parquet file in memory from batch of examples.
        
        Args:
            batch: List of example dictionaries.
            schema: Parquet schema.
            
        Returns:
            Parquet file as bytes.
        """
        # Convert examples to Parquet format
        speakers_list = []
        speaker_volumes_list = []
        simultaneous_segments_list = []
        
        for example in batch:
            # Speakers
            speakers_data = example.get("speakers", [])
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
            
            # Speaker volumes
            volumes_data = example.get("speaker_volumes", [])
            if isinstance(volumes_data, dict):
                volumes_data = [
                    {"speaker_id": int(sid), "volume": float(vol)}
                    for sid, vol in volumes_data.items()
                ]
            speaker_volumes_list.append(volumes_data)
            
            # Simultaneous segments
            sim_data = example.get("simultaneous_segments", [])
            simultaneous_segments_list.append(sim_data)
        
        # Audio - convert to dict format if needed
        audio_list = []
        for example in batch:
            audio_data = example.get("audio")
            if isinstance(audio_data, dict):
                # Already in dict format - ensure array is a list, not numpy array
                # Use .get() to safely access keys with fallback values
                array = audio_data.get("array", [])
                if isinstance(array, np.ndarray):
                    array = array.tolist()
                elif not isinstance(array, list):
                    # Fallback: try to convert to list
                    array = list(array) if hasattr(array, '__iter__') else []
                audio_list.append({
                    "array": array,
                    "sampling_rate": audio_data.get("sampling_rate", example.get("sampling_rate", SAMPLING_RATE)),
                })
            elif isinstance(audio_data, bytes):
                # Convert bytes to dict format
                audio_dict = self._convert_audio_bytes_to_audio(
                    audio_data, example.get("sampling_rate", SAMPLING_RATE)
                )
                audio_list.append({
                    "array": audio_dict["array"].tolist(),  # Convert numpy array to list
                    "sampling_rate": audio_dict["sampling_rate"],
                })
            else:
                # Fallback
                audio_list.append({
                    "array": [],
                    "sampling_rate": example.get("sampling_rate", SAMPLING_RATE),
                })
        
        arrays = {
            "audio": audio_list,
            "duration": [e["duration"] for e in batch],
            "num_speakers": [e["num_speakers"] for e in batch],
            "sampling_rate": [e["sampling_rate"] for e in batch],
            "conversation_type": [e["conversation_type"] for e in batch],
            "difficulty": [e["difficulty"] for e in batch],
            "has_overlaps": [e["has_overlaps"] for e in batch],
            "has_simultaneous": [e["has_simultaneous"] for e in batch],
            "has_noise": [e["has_noise"] for e in batch],
            "speakers": speakers_list,
            "noise_type": [e.get("noise_type") for e in batch],
            "snr": [e.get("snr") for e in batch],
            "speaker_volumes": speaker_volumes_list,
            "simultaneous_segments": simultaneous_segments_list,
        }
        
        table = pa.table(arrays, schema=schema)
        
        # Write to bytes buffer
        buffer = io.BytesIO()
        pq.write_table(
            table,
            buffer,
            compression="zstd",
            compression_level=9,
            row_group_size=1000,
            use_dictionary=True,
        )
        return buffer.getvalue()

    def upload_streaming(self, tracks_iterator: Iterator[Dict[str, Any]]) -> None:
        """
        Upload tracks to HuggingFace Hub in streaming mode using batched Parquet uploads.

        Args:
            tracks_iterator: Iterator of track metadata dictionaries.
        """
        logger.info(f"Starting streaming upload to {self.repo_id}...")
        logger.info("Using batched Parquet upload for memory efficiency")

        schema = self._create_parquet_schema()
        current_batch = []
        current_size = 0
        file_index = 0
        total_tracks = 0
        
        for track in tracks_iterator:
            try:
                # Audio will be converted to dict format in _create_parquet_batch
                # Just prepare metadata structure
                example = self._convert_track_to_example(track, convert_audio=False)
                
                # Estimate size (audio dict is larger than bytes)
                audio_data = track["audio"]
                if isinstance(audio_data, dict):
                    # Python lists have significant overhead per element (pointers + object overhead)
                    # Use conservative multiplier: ~3x the raw float32 size to account for:
                    # - List pointer overhead (8 bytes per pointer on 64-bit systems)
                    # - Python float object overhead (~24 bytes per float object)
                    # - Parquet serialization overhead
                    array_len = len(audio_data.get("array", []))
                    array_size = array_len * 12  # Conservative: ~12 bytes per element (3x float32)
                    estimated_size = array_size + 2000  # Metadata overhead
                elif isinstance(audio_data, bytes):
                    # Estimate based on expected number of float32 samples after conversion
                    # WAV files: approximate samples = (bytes - header) / bytes_per_sample
                    # For 16-bit mono WAV: bytes_per_sample = 2, header ~44 bytes
                    # After conversion to float32 list: ~12 bytes per sample
                    wav_header_size = 44  # Standard WAV header size
                    bytes_per_sample = 2  # 16-bit = 2 bytes per sample
                    estimated_samples = max(0, (len(audio_data) - wav_header_size) // bytes_per_sample)
                    # Use same multiplier as dict format: 12 bytes per sample
                    array_size = estimated_samples * 12
                    estimated_size = array_size + 2000  # Metadata overhead
                else:
                    estimated_size = 50000  # Fallback estimate
                
                # Check if we need to upload current batch
                if current_batch and (current_size + estimated_size) > self.max_batch_size_bytes:
                    # Create Parquet batch and upload
                    parquet_bytes = self._create_parquet_batch(current_batch, schema)
                    filename = f"train-{file_index:05d}.parquet"
                    
                    # Upload to Hub in ./data directory
                    self.hf_api.upload_file(
                        path_or_fileobj=io.BytesIO(parquet_bytes),
                        path_in_repo=f"data/{filename}",
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.hf_token,
                    )
                    
                    logger.info(f"  Uploaded batch {file_index + 1}: {len(current_batch)} tracks, {len(parquet_bytes) / (1024*1024):.2f} MB")
                    
                    file_index += 1
                    current_batch = []
                    current_size = 0
                
                # Add to current batch
                current_batch.append(example)
                current_size += estimated_size
                total_tracks += 1
                
                if total_tracks % 100 == 0:
                    logger.info(f"  Processed {total_tracks} tracks...")
                    
            except Exception as e:
                logger.error(f"Failed to process track: {e}")
                continue
        
        # Upload remaining batch
        if current_batch:
            parquet_bytes = self._create_parquet_batch(current_batch, schema)
            filename = f"train-{file_index:05d}.parquet"
            
            # Upload to Hub in ./data directory
            self.hf_api.upload_file(
                path_or_fileobj=io.BytesIO(parquet_bytes),
                path_in_repo=f"data/{filename}",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token,
            )
            
            logger.info(f"  Uploaded final batch {file_index + 1}: {len(current_batch)} tracks, {len(parquet_bytes) / (1024*1024):.2f} MB")
            file_index += 1
        
        total_files = file_index
        
        # Create dataset card and README
        logger.info("Creating dataset card...")
        self._create_dataset_card(total_files, total_tracks)
        
        logger.info(f"\n✓ Successfully uploaded {total_tracks} tracks in {total_files} file(s) to {self.repo_id}")
        logger.info(f"  View at: https://huggingface.co/datasets/{self.repo_id}")

    def _create_dataset_card(self, num_files: int, num_tracks: int) -> None:
        """Create a basic dataset card README."""
        readme_content = f"""---
license: mit
task_categories:
- automatic-speech-recognition
- audio-classification
tags:
- audio
- speech
- diarization
- synthetic
---

# {self.repo_id.split('/')[-1]}

Synthetic speech diarization dataset.

## Dataset Details

- **Number of tracks**: {num_tracks}
- **Number of Parquet files**: {num_files}
- **Sampling rate**: {SAMPLING_RATE} Hz

## Dataset Structure

The dataset contains audio tracks with speaker diarization annotations.

### Features

- `audio`: Audio waveform (Audio feature)
- `duration`: Track duration in seconds
- `num_speakers`: Number of speakers in the track
- `speakers`: List of speaker segments with timestamps and text
- `speaker_volumes`: Speaker volume levels
- `conversation_type`: Type of conversation (dialogue, monologue, etc.)
- `difficulty`: Difficulty level (easy, medium, hard)
- `has_overlaps`: Whether track contains overlapping speech
- `has_simultaneous`: Whether track contains simultaneous speech
- `has_noise`: Whether track contains background noise

## Usage

```python
from datasets import load_dataset

# Load dataset from HuggingFace Hub
# Files are stored in the 'data' directory
dataset = load_dataset("{self.repo_id}", data_dir="data")
```
"""
        
        self.hf_api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.hf_token,
        )

