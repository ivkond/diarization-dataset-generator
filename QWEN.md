# Diarization Dataset Generator - QWEN Context

## Project Overview

This is a **Speaker Diarization Dataset Generator** project that creates synthetic multi-speaker audio datasets for training and evaluating speaker diarization systems. The project generates realistic audio tracks with multiple speakers, incorporating features like overlaps, simultaneous speech, and background noise.

**Main Purpose**: Generate synthetic audio datasets for speaker diarization tasks, designed specifically for Russian language (ru-RU) from the FBK-MT/Speech-MASSIVE-test dataset.

**Architecture**: Modular Python application with clear separation of concerns using Pydantic for configuration validation, and parallel processing for efficient dataset generation.

## Key Technologies & Dependencies

- Python 3.10+
- Hugging Face Datasets library (with audio support)
- Pydantic (configuration validation)
- NumPy (audio processing)
- PyArrow (Parquet file support)
- PyYAML (configuration loading)
- Python-dotenv (environment variable management)
- SoundFile (audio I/O)

## Project Structure

```
diarization-dataset-generator/
├── src/
│   ├── audio/              # Audio processing utilities (noise, encoding, normalization)
│   ├── config/             # Configuration management with Pydantic validation
│   ├── dataset/            # Source dataset loading and indexing
│   ├── patterns/           # Conversation patterns (Strategy pattern implementation)
│   ├── storage/            # Storage utilities (Parquet writer, Hub uploader)
│   ├── track/              # Track generation (planner, builder, generator)
│   ├── constants.py        # Application-wide constants
│   ├── exceptions.py       # Custom exception definitions
│   └── main.py             # Main entry point
├── config.yaml             # Configuration file for dataset generation parameters
├── dataset_script.py       # Hugging Face dataset script with Audio feature support
├── main.py                 # Entry point wrapper
├── pyproject.toml          # Project dependencies and metadata
├── README.md               # Comprehensive project documentation
└── uv.lock                 # Dependency lock file
```

## Configuration

The project uses `config.yaml` with Pydantic validation for all parameters. Key configuration areas:

- **Dataset**: Source dataset (FBK-MT/Speech-MASSIVE-test), language, features
- **Output**: Local path or HuggingFace Hub repository ID for direct upload
- **Track**: Duration, pause, and short segment parameters
- **Speakers**: Speaker count range, volume settings, consecutive limits
- **Overlaps**: Probability and percentage ranges for overlapping speech
- **Simultaneous speech**: Probability and duration settings
- **Noise**: Background noise probability, types, and SNR settings
- **Conversation patterns**: Probability distribution for dialogue types
- **Difficulty distribution**: Distribution of easy, medium, hard tracks

## Building and Running

### Setup
```bash
# Install dependencies using uv
uv sync

# Set up environment variables in .env file:
# HF_TOKEN=your_huggingface_token_here
```

### Generation
```bash
# Generate dataset using configuration
python -m src.main
```

**Output modes:**
1. **Direct upload to HuggingFace Hub**:
   - Set `output.repo_id` in config.yaml
   - Dataset uploads streamingly during generation (saves disk space)

2. **Local Parquet files**:
   - Leave `output.repo_id` empty
   - Dataset saved as Parquet files to specified path

## Key Components

### Audio Processing (`src/audio/`)
- `processor.py`: Audio extraction, normalization utilities
- `noise.py`: Various noise generation (white, pink, brown, crowd, cafe, static, hum)
- `encoder.py`: Audio encoding functions

### Configuration (`src/config/`)
- `models.py`: Pydantic models for configuration validation
- `loader.py`: Configuration loading from YAML and environment variables

### Dataset Handling (`src/dataset/`)
- `loader.py`: Dataset loading from HuggingFace Hub
- `indexer.py`: Speaker indexing and metadata caching

### Track Generation (`src/track/`)
- `planner.py`: Planning conversation patterns and speaker assignments
- `builder.py`: Building audio tracks with segments, overlaps, noise
- `generator.py`: Main track generation orchestrator
- `types.py`: Type definitions for track events and metadata

### Storage (`src/storage/`)
- `parquet.py`: Parquet file writer with size-based splitting
- `hub_uploader.py`: Direct streaming upload to HuggingFace Hub

### Patterns (`src/patterns/`)
- Strategy pattern implementation for different conversation types:
  - Dialogue, Monologue, Group Discussion, Interview

## Data Format

Generated dataset contains tracks with:
- Multi-speaker conversations (2-4 speakers)
- Precise timestamps and transcriptions
- Various conversation patterns
- Realistic features (overlaps, simultaneous speech, background noise)
- Difficulty levels (easy, medium, hard)

Stored in Parquet format, compatible with Hugging Face Datasets, with structured metadata including audio waveforms, speaker segments, and noise information.

## Development Conventions

- Type hints used throughout the codebase
- Pydantic models for configuration validation
- Parallel processing with ProcessPoolExecutor
- Audio caching in worker processes for performance
- Comprehensive logging
- Modular, well-documented code with clear separation of concerns
- Error handling with custom exceptions