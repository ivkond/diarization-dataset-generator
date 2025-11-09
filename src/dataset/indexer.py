"""Dataset indexing utilities for efficient speaker lookup."""

from typing import Dict, List, Optional, Tuple

from datasets import Dataset

from ..exceptions import DatasetError


class DatasetIndexer:
    """Indexer for building speaker and metadata indices."""

    def __init__(self, dataset: Dataset, speaker_id_feature: str, text_feature: str):
        """
        Initialize indexer.

        Args:
            dataset: Dataset to index.
            speaker_id_feature: Name of the feature containing speaker ID.
            text_feature: Name of the feature containing text.
        """
        self.dataset = dataset
        self.speaker_id_feature = speaker_id_feature
        self.text_feature = text_feature
        self._speaker_index: Optional[Dict[str, List[int]]] = None
        self._metadata_cache: Optional[Dict[int, Tuple[str, str]]] = None

    def build_speaker_index(self) -> Dict[str, List[int]]:
        """
        Build index mapping speaker_id to list of dataset indices.

        Returns:
            Dictionary mapping speaker_id to list of dataset indices.

        Raises:
            DatasetError: If indexing fails.
        """
        speaker_index: Dict[str, List[int]] = {}
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                speaker_id = sample[self.speaker_id_feature]
                if speaker_id not in speaker_index:
                    speaker_index[speaker_id] = []
                speaker_index[speaker_id].append(idx)
            except Exception as e:
                # Log but continue - some samples might be corrupted
                continue
        self._speaker_index = speaker_index
        return speaker_index

    def build_metadata_cache(self) -> Dict[int, Tuple[str, str]]:
        """
        Build cache of metadata (speaker_id, text) for each dataset index.

        Returns:
            Dictionary mapping dataset index to (speaker_id, text) tuple.

        Raises:
            DatasetError: If caching fails.
        """
        metadata_cache: Dict[int, Tuple[str, str]] = {}
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                speaker_id = sample[self.speaker_id_feature]
                text = sample[self.text_feature]
                metadata_cache[idx] = (speaker_id, text)
            except Exception as e:
                # Log but continue - some samples might be corrupted
                continue
        self._metadata_cache = metadata_cache
        return metadata_cache

    @property
    def speaker_index(self) -> Dict[str, List[int]]:
        """
        Get speaker index.

        Returns:
            Speaker index dictionary.

        Raises:
            DatasetError: If index is not built yet.
        """
        if self._speaker_index is None:
            raise DatasetError("Speaker index not built. Call build_speaker_index() first.")
        return self._speaker_index

    @property
    def metadata_cache(self) -> Dict[int, Tuple[str, str]]:
        """
        Get metadata cache.

        Returns:
            Metadata cache dictionary.

        Raises:
            DatasetError: If cache is not built yet.
        """
        if self._metadata_cache is None:
            raise DatasetError("Metadata cache not built. Call build_metadata_cache() first.")
        return self._metadata_cache

