# Speaker Diarization Dataset

A synthetic multi-speaker audio dataset for speaker diarization tasks, generated from the FBK-MT/Speech-MASSIVE-test dataset.

## Dataset Description

This dataset contains synthetic audio tracks with multiple speakers, designed for training and evaluating speaker diarization systems. Each track includes:

- **Multi-speaker conversations** with 2-4 speakers
- **Speaker segments** with precise timestamps and transcriptions
- **Various conversation patterns**: dialogues, monologues, group discussions, interviews
- **Realistic features**: overlaps, simultaneous speech, background noise
- **Difficulty levels**: easy, medium, hard

### Dataset Structure

The dataset is stored in Parquet format, compatible with Hugging Face Datasets:

```
.
├── dataset/
│   ├── train-00000-of-00001.parquet
│   ├── train-00001-of-00001.parquet
│   └── ...
└── README.md
```

Each Parquet file contains multiple tracks with embedded audio (WAV bytes) and metadata.

### Data Fields

Each record in the Parquet dataset contains:

- `audio` (bytes): WAV audio file as bytes
- `duration` (float): Track duration in seconds
- `num_speakers` (int): Number of speakers in the track
- `sampling_rate` (int): Audio sampling rate (16000 Hz)
- `conversation_type` (string): Type of conversation pattern (dialogue, monologue, group_discussion, interview)
- `difficulty` (string): Difficulty level (easy, medium, hard)
- `has_overlaps` (bool): Whether the track contains overlapping speech
- `has_simultaneous` (bool): Whether the track contains simultaneous speech
- `has_noise` (bool): Whether background noise was added
- `speakers` (string): JSON string containing array of speaker segments, each containing:
  - `speaker_id` (int): Speaker identifier (1-indexed)
  - `start` (float): Start time in seconds
  - `end` (float): End time in seconds
  - `duration` (float): Segment duration in seconds
  - `text` (string, optional): Transcription of the segment
- `noise_type` (string, optional): Type of background noise if present
- `snr` (float, optional): Signal-to-noise ratio in dB if noise is present
- `speaker_volumes` (string, optional): JSON string with speaker volume levels
- `simultaneous_segments` (string, optional): JSON string with simultaneous speech segments

### Example Record

```python
{
  "audio": b"RIFF...",  # WAV bytes
  "duration": 55.44,
  "num_speakers": 3,
  "sampling_rate": 16000,
  "conversation_type": "dialogue",
  "difficulty": "medium",
  "has_overlaps": True,
  "has_simultaneous": False,
  "has_noise": True,
  "speakers": '[{"speaker_id": 1, "start": 0.0, "end": 6.0, "duration": 6.0, "text": "добавь встречу в офисе с василием на три часа дня во вторник"}, {"speaker_id": 2, "start": 8.15, "end": 10.45, "duration": 4.45, "text": "ответь на электронное письмо"}]',
  "noise_type": "white",
  "snr": 20.5
}
```

## Dataset Statistics

- **Total tracks**: Variable (configurable)
- **Duration range**: 30-60 seconds per track
- **Sampling rate**: 16 kHz
- **Audio format**: WAV (PCM)
- **Language**: Russian (ru-RU)

## Usage

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset
import json

# Load from local Parquet files
dataset = load_dataset("parquet", data_files="dataset/train-*.parquet", split="train")

# Or load from Hugging Face Hub (if uploaded)
# dataset = load_dataset("your-username/diarization-dataset")
```

### Accessing Audio and Metadata

```python
# Get a sample
sample = dataset[0]

# Access audio (needs conversion from bytes)
import io
import soundfile as sf
import numpy as np

audio_bytes = sample["audio"]
wav_buffer = io.BytesIO(audio_bytes)
audio_array, sampling_rate = sf.read(wav_buffer)
# audio_array is now a numpy array

# Access metadata
duration = sample["duration"]
num_speakers = sample["num_speakers"]
conversation_type = sample["conversation_type"]
difficulty = sample["difficulty"]

# Parse speakers JSON string
speakers = json.loads(sample["speakers"])  # List of speaker segments
```

### Working with Speaker Segments

```python
import json

# Parse speakers JSON string
speakers = json.loads(sample["speakers"])

for segment in speakers:
    speaker_id = segment["speaker_id"]
    start_time = segment["start"]
    end_time = segment["end"]
    text = segment.get("text", "")  # Text is optional
    print(f"Speaker {speaker_id}: {start_time:.2f}s - {end_time:.2f}s: {text}")
```

## Dataset Generation

This dataset is generated using a modular Python script.

### Project Structure

```
.
├── src/
│   ├── main.py              # Main entry point
│   ├── config/              # Configuration management with Pydantic validation
│   ├── audio/               # Audio processing (noise, encoding, etc.)
│   ├── dataset/             # Source dataset loading and indexing
│   ├── track/               # Track generation (planner, builder, generator)
│   ├── patterns/            # Conversation patterns (Strategy pattern)
│   └── storage/             # Storage utilities (Parquet writer, Hub uploader)
├── config.yaml              # Configuration file
└── main.py                  # Entry point wrapper
```

### Installation

1. Install dependencies:
```bash
uv sync
```

This will install all required packages including:
- `datasets[audio]` - Hugging Face datasets library
- `soundfile` - Audio file I/O
- `pyarrow` - Parquet file support
- `pydantic` - Configuration validation
- And more...

### Configuration

1. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Hugging Face token:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```
   Get your token from: https://huggingface.co/settings/tokens

2. Configure parameters in `config.yaml`:
   - `random_seed`: Random seed for reproducibility
   - `track_count`: Number of tracks to generate
   - `dataset`: Dataset source configuration (path, language, split)
   - `output`: Output configuration:
     - `path`: Output directory for Parquet files (used if `repo_id` is empty)
     - `repo_id`: HuggingFace Hub repository ID (e.g., "username/dataset-name"). If set, dataset will be uploaded directly to Hub during generation. If empty, dataset will be saved to local Parquet files.
     - `parquet_file_size_mb`: Maximum Parquet file size in MB
   - `track`: Track generation parameters (duration range, pauses, short segments)
   - `speakers`: Speaker configuration (count, volumes, consecutive limits)
   - `overlaps`: Overlap probability and percentage ranges
   - `simultaneous_speech`: Simultaneous speech probability and duration
   - `noise`: Background noise settings (probability, types, SNR range)
   - `conversation_patterns`: Distribution of conversation types (probabilities must sum to 1.0)
   - `difficulty_distribution`: Distribution of difficulty levels

### Generation

Run the generation script:
```bash
python -m src.main
```

The script will:
- Load and validate configuration from `config.yaml` (using Pydantic)
- Load Hugging Face token from `.env`
- Load the source dataset from Hugging Face Hub
- Build speaker and metadata indices for efficient access
- Generate audio tracks with various conversation patterns
- Apply realistic features (overlaps, simultaneous speech, noise)

**Output modes:**

1. **Direct upload to HuggingFace Hub** (recommended):
   - Set `output.repo_id` in `config.yaml` to your repository ID (e.g., `"username/dataset-name"`)
   - Dataset will be uploaded **streamingly during generation** using batched Parquet uploads
   - No local files are created - saves disk space and time
   - Example:
     ```yaml
     output:
       path: "./dataset"  # Not used when repo_id is set
       repo_id: "username/dataset-name"
     ```

2. **Save to local Parquet files**:
   - Leave `output.repo_id` empty in `config.yaml`
   - Dataset will be saved as Parquet files in the directory specified by `output.path`
   - Example:
     ```yaml
     output:
       path: "./dataset"
       repo_id: ""  # Empty = save locally
     ```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{diarization_dataset,
  title={Synthetic speech diarization dataset},
  author={ivkond},
  year={2025},
  url={https://huggingface.co/datasets/ivkond/synthetic-speech-diarization-ru}
}
```

## License

MIT License

## Acknowledgments

- Generated from [FBK-MT/Speech-MASSIVE-test](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test)
- Language: Russian (ru-RU)
