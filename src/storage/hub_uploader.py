"""HuggingFace Hub streaming uploader."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator

from huggingface_hub import HfApi, CommitOperationAdd, create_commit, create_repo, upload_file, upload_folder

from ..constants import SAMPLING_RATE
from .file_storage import FileStorageWriter

logger = logging.getLogger(__name__)

# Batch size for uploading files (upload in batches to reduce API calls)
UPLOAD_BATCH_SIZE = 50


class HubUploader:
    """Streaming uploader for HuggingFace Hub."""

    def __init__(
        self,
        repo_id: str,
        hf_token: str,
        private: bool = False,
    ):
        """
        Initialize Hub uploader.

        Args:
            repo_id: HuggingFace repository ID (format: "username/dataset-name").
            hf_token: HuggingFace authentication token.
            private: Whether the dataset should be private.
        """
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.private = private
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

    def _delete_existing_files(self) -> None:
        """Delete existing files from repository."""
        try:
            files = self.hf_api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            
            # Delete audio files and metadata
            files_to_delete = [
                f for f in files 
                if f.endswith('.wav') or f == 'metadata.jsonl' or f.startswith('audio/')
            ]
            
            if files_to_delete:
                logger.info(f"Deleting {len(files_to_delete)} existing file(s) from repository...")
                for file_path in files_to_delete:
                    self.hf_api.delete_file(
                        path_in_repo=file_path,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.hf_token,
                    )
                logger.info(f"✓ Deleted {len(files_to_delete)} file(s)")
            else:
                logger.info("No existing files found in repository")
        except Exception as e:
            logger.warning(f"Could not delete existing files: {e}")

    def _create_gitattributes(self, output_dir: Path) -> None:
        """
        Create .gitattributes file for Git LFS support.
        
        Args:
            output_dir: Directory where to create .gitattributes.
        """
        gitattributes_path = output_dir / ".gitattributes"
        with open(gitattributes_path, "w", encoding="utf-8") as f:
            f.write("*.wav filter=lfs diff=lfs merge=lfs -text\n")
        logger.debug("Created .gitattributes for Git LFS")

    def upload_streaming(self, tracks_iterator: Iterator[Dict[str, Any]]) -> None:
        """
        Upload tracks to HuggingFace Hub in streaming mode with batch processing.
        Audio files are deleted in batches immediately after successful upload to save disk space.

        Args:
            tracks_iterator: Iterator of track metadata dictionaries.
        """
        # Delete existing files before uploading new ones
        self._delete_existing_files()
        
        logger.info(f"Starting streaming upload to {self.repo_id}...")
        logger.info("Using file-based storage with Git LFS for audio files")
        logger.info(f"Uploading in batches of {UPLOAD_BATCH_SIZE} files for efficiency")
        logger.info("Audio files will be deleted in batches after upload to save disk space")

        # Create temporary directory for file storage
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create audio subdirectory
            audio_dir = temp_path / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            # Metadata file path
            metadata_file = temp_path / "metadata.jsonl"
            
            # Create .gitattributes for Git LFS
            self._create_gitattributes(temp_path)
            
            # Upload .gitattributes first
            self.hf_api.upload_file(
                path_or_fileobj=str(temp_path / ".gitattributes"),
                path_in_repo=".gitattributes",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token,
                commit_message=None,  # Will commit at the end
            )
            
            # Track counter and batch tracking
            track_counter = 0
            total_tracks = 0
            batch_num = 0  # Current batch number
            batch_files = []  # List of (audio_path, relative_path) tuples for current batch
            batch_metadata = []  # List of metadata dicts for current batch
            
            # Create a writer instance for metadata preparation (reused for all tracks)
            temp_writer = FileStorageWriter(output_dir=temp_path)
            
            def upload_batch(batch_num: int, batch_files_list: list, batch_metadata_list: list):
                """
                Upload a batch of files in a single commit and delete them.
                
                Returns:
                    List of successfully uploaded metadata entries (to write to JSONL).
                """
                if not batch_files_list:
                    return []
                
                successfully_uploaded_metadata = []
                successfully_uploaded_files = []  # Track which files were successfully uploaded
                commit_operations = []  # Accumulate operations for batch commit
                
                try:
                    # Prepare commit operations for all files in the batch
                    for idx, (audio_path, relative_path) in enumerate(batch_files_list):
                        try:
                            # Create commit operation (doesn't upload yet)
                            operation = CommitOperationAdd(
                                path_in_repo=relative_path,
                                path_or_fileobj=str(audio_path),
                            )
                            commit_operations.append(operation)
                            # Track files and metadata for successful operations
                            successfully_uploaded_files.append(audio_path)
                            successfully_uploaded_metadata.append(batch_metadata_list[idx])
                        except Exception as e:
                            logger.error(f"Failed to prepare {relative_path} for batch {batch_num}: {e}")
                            # Continue with other files in batch
                    
                    # Create a single commit for all files in the batch
                    if commit_operations:
                        try:
                            self.hf_api.create_commit(
                                repo_id=self.repo_id,
                                repo_type="dataset",
                                operations=commit_operations,
                                commit_message=f"Upload batch {batch_num}: {len(commit_operations)} files",
                                token=self.hf_token,
                            )
                            uploaded_count = len(commit_operations)
                            
                            # Delete only successfully uploaded files to prevent data loss
                            deleted_count = 0
                            for audio_path in successfully_uploaded_files:
                                if audio_path.exists():
                                    try:
                                        audio_path.unlink()
                                        deleted_count += 1
                                    except Exception as e:
                                        logger.warning(f"Failed to delete {audio_path}: {e}")
                            
                            logger.info(f"  Batch {batch_num}: Uploaded {uploaded_count}/{len(batch_files_list)} files in single commit, deleted {deleted_count} files")
                            
                        except Exception as e:
                            logger.error(f"Failed to commit batch {batch_num}: {e}")
                            # Don't delete files if commit failed - they may be retried
                            successfully_uploaded_metadata = []  # No files were actually uploaded
                            successfully_uploaded_files = []
                    else:
                        logger.warning(f"  Batch {batch_num}: No files prepared for upload")
                    
                    # Log warning if some files failed to prepare (they remain on disk)
                    failed_count = len(batch_files_list) - len(successfully_uploaded_files)
                    if failed_count > 0:
                        logger.warning(f"  Batch {batch_num}: {failed_count} file(s) failed to prepare and were NOT deleted to prevent data loss")
                    
                except Exception as e:
                    logger.error(f"Failed to upload batch {batch_num}: {e}")
                    # Don't delete any files on batch-level error - they may be retried
                    successfully_uploaded_metadata = []
                    successfully_uploaded_files = []
                
                return successfully_uploaded_metadata
            
            # Process tracks: accumulate in batches, upload batch, delete batch
            with open(metadata_file, "w", encoding="utf-8") as metadata_f:
                for track in tracks_iterator:
                    # Increment track_counter at the start to ensure unique filenames
                    # even if this track is skipped due to errors
                    current_track_id = track_counter
                    track_counter += 1
                    
                    # Initialize audio_path before try block to ensure it's always defined
                    # in the except handler, even if an exception occurs before assignment
                    track_filename = f"track-{current_track_id:05d}.wav"
                    audio_path = audio_dir / track_filename
                    relative_audio_path = f"audio/{track_filename}"
                    
                    try:
                        # Write audio file
                        audio_bytes = track.get("audio")
                        if not isinstance(audio_bytes, bytes):
                            logger.error(f"Track {current_track_id}: audio is not bytes, skipping")
                            continue
                        
                        with open(audio_path, "wb") as audio_f:
                            audio_f.write(audio_bytes)
                        
                        # Add to current batch
                        batch_files.append((audio_path, relative_audio_path))
                        
                        # Prepare metadata (remove audio bytes, add path)
                        # Store metadata but don't write to JSONL yet - only after successful upload
                        metadata = temp_writer._prepare_metadata(track, relative_audio_path)
                        batch_metadata.append(metadata)
                        
                        # Upload batch when it reaches the batch size
                        if len(batch_files) >= UPLOAD_BATCH_SIZE:
                            batch_num += 1
                            # Upload batch and get successfully uploaded metadata
                            successfully_uploaded = upload_batch(batch_num, batch_files, batch_metadata)
                            
                            # Write metadata only for successfully uploaded files
                            # Use try-finally to ensure batch is cleared even if metadata writing fails
                            try:
                                for uploaded_metadata in successfully_uploaded:
                                    json_line = json.dumps(uploaded_metadata, ensure_ascii=False)
                                    metadata_f.write(json_line + "\n")
                                    total_tracks += 1
                                
                                metadata_f.flush()  # Ensure data is written immediately
                            except Exception as e:
                                logger.error(f"Failed to write metadata for batch {batch_num}: {e}")
                                # Continue - batch will be cleared in finally block
                            finally:
                                # Always clear batch after upload (files are already deleted)
                                # This prevents attempts to upload non-existent files
                                batch_files = []
                                batch_metadata = []
                        
                    except Exception as e:
                        logger.error(f"Failed to process track {current_track_id}: {e}")
                        # Clean up audio file if it exists
                        if audio_path.exists():
                            audio_path.unlink()
                        continue
                
                # Upload remaining files in the last incomplete batch
                if batch_files:
                    batch_num += 1
                    # Upload batch and get successfully uploaded metadata
                    successfully_uploaded = upload_batch(batch_num, batch_files, batch_metadata)
                    
                    # Write metadata only for successfully uploaded files
                    # Use try-finally to ensure batch is cleared even if metadata writing fails
                    try:
                        for uploaded_metadata in successfully_uploaded:
                            json_line = json.dumps(uploaded_metadata, ensure_ascii=False)
                            metadata_f.write(json_line + "\n")
                            total_tracks += 1
                        
                        metadata_f.flush()  # Ensure data is written immediately
                    except Exception as e:
                        logger.error(f"Failed to write metadata for final batch {batch_num}: {e}")
                        # Continue - batch will be cleared in finally block
                    finally:
                        # Always clear batch after upload (files are already deleted)
                        batch_files = []
                        batch_metadata = []
            
            if total_tracks == 0:
                logger.warning("No tracks to upload")
                return
            
            # Upload metadata.jsonl and dataset_script.py in a single commit
            logger.info(f"Uploading metadata.jsonl and dataset_script.py...")
            final_operations = []
            
            # Add metadata.jsonl
            final_operations.append(
                CommitOperationAdd(
                    path_in_repo="metadata.jsonl",
                    path_or_fileobj=str(metadata_file),
                )
            )
            
            # Add dataset_script.py if it exists
            dataset_script_path = Path(__file__).parent.parent.parent / "dataset_script.py"
            if dataset_script_path.exists():
                dest_script_path = temp_path / "dataset_script.py"
                shutil.copy2(dataset_script_path, dest_script_path)
                final_operations.append(
                    CommitOperationAdd(
                        path_in_repo="dataset_script.py",
                        path_or_fileobj=str(dest_script_path),
                    )
                )
                logger.info("Prepared dataset_script.py for upload")
            else:
                logger.warning(f"dataset_script.py not found at {dataset_script_path}")
            
            # Create commit for metadata and script
            if final_operations:
                self.hf_api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=final_operations,
                    commit_message=f"Add metadata.jsonl and dataset_script.py for {total_tracks} tracks",
                    token=self.hf_token,
                )
                logger.info("Uploaded metadata.jsonl and dataset_script.py in single commit")
        
        # Upload README in separate commit (this will also trigger commit for previous files)
        logger.info("Creating dataset card...")
        readme_content = self._create_dataset_card_content(total_tracks)
        self.hf_api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.hf_token,
            commit_message=f"Upload dataset: {total_tracks} tracks",
        )
        
        logger.info(f"\n✓ Successfully uploaded {total_tracks} tracks to {self.repo_id}")
        logger.info(f"  View at: https://huggingface.co/datasets/{self.repo_id}")

    def _create_dataset_card_content(self, num_tracks: int) -> str:
        """Create dataset card README content."""
        return f"""---
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
- **Sampling rate**: {SAMPLING_RATE} Hz
- **Audio format**: WAV (16-bit PCM)
- **Storage**: Audio files with JSONL metadata

## Dataset Structure

The dataset contains audio tracks with speaker diarization annotations.

### Directory Structure

```
audio/
  track-00001.wav
  track-00002.wav
  ...
metadata.jsonl
```

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
from datasets import load_dataset, Audio

# Load dataset from HuggingFace Hub
dataset = load_dataset("{self.repo_id}")
dataset = dataset.cast_column("audio", Audio(sampling_rate={SAMPLING_RATE}))

# Access a sample
sample = dataset[0]
print(f"Duration: {{sample['duration']}}s")
print(f"Speakers: {{sample['num_speakers']}}")
```

## Notes

- Audio files are stored using Git LFS for efficient version control
- Metadata is stored in JSONL format (one JSON object per line)
- Each track has a unique sequential identifier (track-00001, track-00002, etc.)
"""
