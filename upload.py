"""
Script to upload the generated dataset to HuggingFace Hub.

This script loads Parquet files from the dataset directory, converts audio bytes
to Audio features, and publishes the dataset to HuggingFace Hub.
"""

import os
from pathlib import Path

import yaml
from datasets import Audio, load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration from YAML
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Configuration
DATASET_DIR = Path(config["output"]["path"])
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = config["output"]["repo_id"]

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Please set it in your .env file or environment."
    )

if not HF_REPO_ID:
    raise ValueError(
        "HF_REPO_ID environment variable is not set. "
        "Please set it in your .env file or environment. "
        "Format: 'username/dataset-name'"
    )


def convert_audio_bytes_to_audio(audio_bytes, sampling_rate=16000):
    """
    Convert WAV bytes to audio array format expected by HuggingFace.
    
    Args:
        audio_bytes: Bytes of WAV file
        sampling_rate: Sampling rate of the audio
        
    Returns:
        dict with 'array' and 'sampling_rate' keys
    """
    import io
    import soundfile as sf
    import numpy as np
    
    # Read WAV bytes into numpy array
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


def load_parquet_dataset(dataset_dir):
    """
    Load dataset from Parquet files in the specified directory.
    
    Args:
        dataset_dir: Path to directory containing Parquet files
        
    Returns:
        Dataset object
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all Parquet files
    parquet_files = sorted(dataset_dir.glob("train-*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {dataset_dir}. "
            "Please run main.py first to generate the dataset."
        )
    
    print(f"Found {len(parquet_files)} Parquet file(s)")
    
    # Load dataset from Parquet files
    # HuggingFace datasets can load Parquet files directly
    dataset = load_dataset(
        "parquet",
        data_files=[str(f) for f in parquet_files],
        split="train",
    )
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Convert audio bytes to Audio feature
    print("Converting audio bytes to Audio format...")
    
    def process_audio(example):
        """Process a single example to convert audio bytes to Audio format."""
        audio_bytes = example["audio"]
        audio_dict = convert_audio_bytes_to_audio(
            audio_bytes, 
            sampling_rate=example.get("sampling_rate", 16000)
        )
        example["audio"] = audio_dict
        return example
    
    # Apply conversion
    dataset = dataset.map(
        process_audio,
        desc="Converting audio bytes",
        #num_proc=1,  # Process sequentially to avoid memory issues
    )
    
    # Cast audio column to Audio feature
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Parse JSON strings back to objects
    import json
    
    def parse_json_fields(example):
        """Parse JSON string fields back to objects."""
        if "speakers" in example and isinstance(example["speakers"], str):
            example["speakers"] = json.loads(example["speakers"])
        if "speaker_volumes" in example and isinstance(example["speaker_volumes"], str):
            example["speaker_volumes"] = json.loads(example["speaker_volumes"])
        if "simultaneous_segments" in example and isinstance(example["simultaneous_segments"], str):
            example["simultaneous_segments"] = json.loads(example["simultaneous_segments"])
        return example
    
    dataset = dataset.map(
        parse_json_fields,
        desc="Parsing JSON fields",
    )
    
    return dataset


def main():
    """Main function to upload dataset to HuggingFace Hub."""
    print("=" * 60)
    print("HuggingFace Dataset Upload Script")
    print("=" * 60)
    print(f"Dataset directory: {DATASET_DIR.absolute()}")
    print(f"HuggingFace repository: {HF_REPO_ID}")
    print()
    
    # Load dataset
    dataset = load_parquet_dataset(DATASET_DIR)
    
    print("\nDataset info:")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")
    print()
    
    # Upload to HuggingFace Hub
    print("Uploading to HuggingFace Hub...")
    print("This may take a while depending on dataset size...")
    
    try:
        dataset.push_to_hub(
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            private=False,  # Set to True if you want a private dataset
        )
        print(f"\n✓ Successfully uploaded dataset to: {HF_REPO_ID}")
        print(f"  View at: https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

