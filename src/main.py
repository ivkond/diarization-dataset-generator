"""Main entry point for dataset generation."""

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config.loader import get_hf_token, load_config
from .constants import SAMPLING_RATE
from .dataset.indexer import DatasetIndexer
from .dataset.loader import DatasetLoader
from .patterns.selector import PatternSelector
from .storage.parquet import ParquetWriter
from .track.generator import TrackGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global variable to store dataset in each process
_worker_dataset = None
_worker_dataset_loader = None
_worker_audio_cache: Dict[int, np.ndarray] = {}

def _init_worker(
    dataset_path: str,
    language: str,
    split_name: str,
    hf_token: str,
    sampling_rate: int,
):
    """
    Initialize worker process - load dataset once per process.
    This is called once when each worker process starts.
    """
    global _worker_dataset, _worker_dataset_loader, _worker_audio_cache
    
    # Load dataset once per worker process
    _worker_dataset_loader = DatasetLoader(
        dataset_path=dataset_path,
        language=language,
        split_name=split_name,
        hf_token=hf_token,
        sampling_rate=sampling_rate,
    )
    _worker_dataset = _worker_dataset_loader.load()
    # Initialize audio cache for this worker process
    _worker_audio_cache.clear()


def _generate_track_worker(
    args: Tuple[int, Any, Dict, Dict, Dict, int]
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """
    Worker function for generating a single track in a separate process.
    
    Args:
        args: Tuple containing:
            - track_id: Track identifier
            - config: Configuration object
            - speaker_index: Speaker index dictionary
            - metadata_cache: Metadata cache dictionary
            - conversation_patterns: Conversation pattern probabilities
            - base_seed: Base random seed
            
    Returns:
        Tuple of (track_id, track_metadata or None, error_message or None)
    """
    global _worker_dataset, _worker_audio_cache
    
    (
        track_id,
        config,
        speaker_index,
        metadata_cache,
        conversation_patterns,
        base_seed,
    ) = args
    
    try:
        # Initialize random generators with unique seed for this process and track
        process_id = os.getpid()
        unique_seed = base_seed + process_id + track_id
        random.seed(unique_seed)
        np.random.seed(unique_seed)
        
        # Use pre-loaded dataset from worker initialization
        if _worker_dataset is None:
            return (track_id, None, "Dataset not initialized in worker process")
        dataset = _worker_dataset
        
        # Recreate pattern selector (it uses random internally)
        pattern_selector = PatternSelector(conversation_patterns)
        
        # Create track generator with audio cache
        track_generator = TrackGenerator(
            dataset=dataset,
            config=config,
            speaker_index=speaker_index,
            metadata_cache=metadata_cache,
            pattern_selector=pattern_selector,
            audio_cache=_worker_audio_cache,
        )
        
        # Generate track
        track_metadata = track_generator.generate(track_id)
        return (track_id, track_metadata, None)
    except Exception as e:
        return (track_id, None, str(e))
    finally:
        # Note: We keep the audio cache for reuse across tracks in the same worker
        # as it significantly improves performance. The cache will be cleared
        # when the worker process terminates.
        # Removed gc.collect() as it adds overhead without significant benefit
        pass


def main() -> None:
    """Main function to generate dataset."""
    logger.info("=" * 60)
    logger.info("Diarization Dataset Generator")
    logger.info("=" * 60)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    hf_token = get_hf_token()

    # Setup random seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DatasetLoader(
        dataset_path=config.dataset.path,
        language=config.dataset.language,
        split_name=config.dataset.split_name,
        hf_token=hf_token,
        sampling_rate=SAMPLING_RATE,
    )
    dataset = dataset_loader.load()
    logger.info(f"Dataset loaded. Total samples: {len(dataset)}")

    # Build indices
    logger.info("Building speaker index...")
    indexer = DatasetIndexer(
        dataset=dataset,
        speaker_id_feature=config.dataset.feature_speaker_id,
        text_feature=config.dataset.feature_text,
    )
    speaker_index = indexer.build_speaker_index()
    logger.info(f"Speaker index built. Found {len(speaker_index)} unique speakers")

    logger.info("Caching metadata...")
    metadata_cache = indexer.build_metadata_cache()
    logger.info("Metadata cache built")

    # Generate tracks in parallel
    num_workers = os.cpu_count() or 1
    logger.info(f"\nGenerating {config.track_count} tracks using {num_workers} worker process(es)...")
    
    # Prepare arguments for worker processes
    worker_args = []
    for track_id in range(config.track_count):
        args = (
            track_id,
            config,
            speaker_index,
            metadata_cache,
            config.conversation_patterns,
            config.random_seed,
        )
        worker_args.append(args)
    
    # Generate tracks in parallel and write incrementally
    output_dir = Path(config.output.path)
    writer = ParquetWriter(
        output_dir=output_dir,
        max_file_size_mb=config.output.parquet_file_size_mb,
    )
    
    def track_generator():
        """Generator that yields tracks as they are generated."""
        successful_count = 0
        failed_count = 0
        
        # Initialize worker processes with dataset pre-loaded
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(
                config.dataset.path,
                config.dataset.language,
                config.dataset.split_name,
                hf_token,
                SAMPLING_RATE,
            ),
        ) as executor:
            # Submit all tasks
            future_to_track_id = {
                executor.submit(_generate_track_worker, args): args[0]
                for args in worker_args
            }
            
            # Process results as they complete
            for future in as_completed(future_to_track_id):
                track_id, track_metadata, error = future.result()
                
                if track_metadata is not None:
                    successful_count += 1
                    yield track_metadata
                else:
                    failed_count += 1
                    logger.error(f"Failed to generate track {track_id}: {error}")
        
        logger.info(f"Track generation complete: {successful_count} successful, {failed_count} failed")
    
    # Write tracks incrementally as they are generated
    total_files = writer.write_tracks_incremental(track_generator())

    logger.info("\n" + "=" * 60)
    logger.info("Generation complete!")
    logger.info(f"Written {total_files} Parquet file(s)")
    logger.info(f"Parquet files: {output_dir.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

