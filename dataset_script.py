"""Dataset script for Hugging Face dataset with Audio feature support."""

import json
from pathlib import Path

from datasets import Audio, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value, Sequence


class DiarizationDataset(GeneratorBasedBuilder):
    """Dataset builder for synthetic speech diarization dataset."""
    
    def _info(self) -> DatasetInfo:
        """Return dataset info with Audio feature."""
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
            "audio": Audio(sampling_rate=16000),
            "audio_path": Value("string"),
            "duration": Value("float64"),
            "num_speakers": Value("int64"),
            "sampling_rate": Value("int64"),
            "conversation_type": Value("string"),
            "difficulty": Value("string"),
            "has_overlaps": Value("bool"),
            "has_simultaneous": Value("bool"),
            "has_noise": Value("bool"),
            "speakers": Sequence(speaker_struct),
            "noise_type": Value("string"),
            "snr": Value("float64"),
            "speaker_volumes": Sequence(speaker_volume_struct),
            "simultaneous_segments": Sequence(simultaneous_struct),
        })
        
        return DatasetInfo(features=features)
    
    def _split_generators(self, dl_manager):
        """
        Generate splits for the dataset.
        
        Args:
            dl_manager: DownloadManager instance.
        
        Returns:
            List of SplitGenerator objects.
        """
        # When loading from Hub, files are already in the repository
        # Use manual_dir to get the base path, or download to get the file path
        # For files already in repo, download returns the local path
        try:
            # Try to get the metadata file path
            metadata_path = dl_manager.download("metadata.jsonl")
        except Exception:
            # If download doesn't work, use manual_dir approach
            # Get base directory - files are in repo root
            base_path = dl_manager.manual_dir or Path(".")
            metadata_path = str(base_path / "metadata.jsonl")
        
        # Provide the metadata file - _generate_examples will find audio files relative to it
        files = [metadata_path]
        
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"files": files},
            )
        ]
    
    def _generate_examples(self, files):
        """
        Generate examples from JSONL metadata file and audio files.
        
        Args:
            files: List of file paths. Should contain metadata.jsonl file.
        
        Yields:
            Tuple of (example_id, example_dict)
        """
        # Find metadata.jsonl file
        metadata_file = None
        base_path = None
        
        for filepath in files:
            path = Path(filepath)
            if path.name == "metadata.jsonl":
                metadata_file = path
                base_path = path.parent
                break
        
        if metadata_file is None:
            raise ValueError("metadata.jsonl file not found in provided files")
        
        # Read JSONL file line by line
        with open(metadata_file, "r", encoding="utf-8") as f:
            for example_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    metadata = json.loads(line)
                    
                    # Get audio path from metadata
                    audio_path = metadata.get("audio_path")
                    if not audio_path:
                        continue
                    
                    # Resolve audio file path relative to metadata file location
                    if base_path:
                        audio_file = base_path / audio_path
                    else:
                        audio_file = Path(audio_path)
                    
                    # Check if audio file exists
                    if not audio_file.exists():
                        continue
                    
                    # Create example dictionary
                    # Always include all fields to match the schema, using None or defaults for optional fields
                    example = {
                        "audio": str(audio_file),  # Audio feature will load from path
                        "audio_path": audio_path,  # Relative path from metadata
                        "duration": metadata.get("duration"),
                        "num_speakers": metadata.get("num_speakers"),
                        "sampling_rate": metadata.get("sampling_rate", 16000),
                        "conversation_type": metadata.get("conversation_type"),
                        "difficulty": metadata.get("difficulty"),
                        "has_overlaps": metadata.get("has_overlaps", False),
                        "has_simultaneous": metadata.get("has_simultaneous", False),
                        "has_noise": metadata.get("has_noise", False),
                        "speakers": metadata.get("speakers", []),
                        # Optional fields - always include with None or defaults
                        "noise_type": metadata.get("noise_type"),
                        "snr": metadata.get("snr"),
                        "speaker_volumes": metadata.get("speaker_volumes", []),
                        "simultaneous_segments": metadata.get("simultaneous_segments", []),
                    }
                    
                    yield example_id, example
                    
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
                except Exception:
                    # Skip examples with errors
                    continue
