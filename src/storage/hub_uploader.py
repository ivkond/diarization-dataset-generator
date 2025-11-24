"""HuggingFace Hub streaming uploader."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator

from huggingface_hub import HfApi, create_repo, upload_folder

from ..constants import SAMPLING_RATE
from .file_storage import FileStorageWriter

logger = logging.getLogger(__name__)


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
        Upload tracks to HuggingFace Hub in streaming mode.

        Args:
            tracks_iterator: Iterator of track metadata dictionaries.
        """
        # Delete existing files before uploading new ones
        self._delete_existing_files()
        
        logger.info(f"Starting streaming upload to {self.repo_id}...")
        logger.info("Using file-based storage with Git LFS for audio files")

        # Create temporary directory for file storage
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Use FileStorageWriter to write files
            writer = FileStorageWriter(output_dir=temp_path)
            total_tracks = writer.write_tracks_incremental(tracks_iterator)
            
            if total_tracks == 0:
                logger.warning("No tracks to upload")
                return
            
            # Create .gitattributes for Git LFS
            self._create_gitattributes(temp_path)
            
            # Copy dataset_script.py to temp directory for upload
            # This allows Hugging Face to use the custom script instead of AudioFolder builder
            dataset_script_path = Path(__file__).parent.parent.parent / "dataset_script.py"
            if dataset_script_path.exists():
                dest_script_path = temp_path / "dataset_script.py"
                shutil.copy2(dataset_script_path, dest_script_path)
                logger.info("Copied dataset_script.py to upload directory")
            else:
                logger.warning(f"dataset_script.py not found at {dataset_script_path}")
            
            # Upload entire directory structure
            logger.info(f"\nUploading {total_tracks} tracks to HuggingFace Hub...")
            logger.info("This may take a while for large datasets...")
            
            upload_folder(
                folder_path=str(temp_path),
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token,
                commit_message=f"Upload dataset: {total_tracks} tracks",
            )
            
            logger.info(f"✓ Uploaded {total_tracks} tracks")
        
        # Upload README in separate commit
        logger.info("Creating dataset card...")
        readme_content = self._create_dataset_card_content(total_tracks)
        self.hf_api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.hf_token,
            commit_message="Update dataset card",
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
