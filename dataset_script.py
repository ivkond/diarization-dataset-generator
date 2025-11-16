"""Dataset script for Hugging Face dataset with Audio feature support."""

import numpy as np
from datasets import Audio, DatasetInfo, Features, Value, Sequence
from datasets.data_files import DataFilesDict
import pyarrow.parquet as pq


def _generate_examples(files):
    """Generate examples from Parquet files."""
    for filepath in files:
        table = pq.read_table(filepath)
        for i in range(len(table)):
            example = {}
            for column_name in table.column_names:
                value = table[column_name][i].as_py()
                
                # Convert audio struct to dict format with numpy array
                if column_name == "audio" and isinstance(value, dict):
                    # Value is in dict format with "array" (list) and "sampling_rate"
                    # Convert list back to numpy array for Audio feature
                    example[column_name] = {
                        "array": np.array(value["array"], dtype=np.float32),
                        "sampling_rate": value["sampling_rate"],
                    }
                else:
                    example[column_name] = value
            
            yield i, example


def _info() -> DatasetInfo:
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

