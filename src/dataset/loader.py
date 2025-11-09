"""Dataset loading utilities."""

from typing import Optional

from datasets import Audio, Dataset, load_dataset

from ..constants import SAMPLING_RATE
from ..exceptions import DatasetError


class DatasetLoader:
    """Loader for HuggingFace datasets."""

    def __init__(
        self,
        dataset_path: str,
        language: str,
        split_name: str,
        hf_token: str,
        sampling_rate: int = SAMPLING_RATE,
    ):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to dataset on HuggingFace Hub.
            language: Dataset language identifier.
            split_name: Split name to load (e.g., "test", "train").
            hf_token: HuggingFace token for authentication.
            sampling_rate: Target sampling rate for audio.
        """
        self.dataset_path = dataset_path
        self.language = language
        self.split_name = split_name
        self.hf_token = hf_token
        self.sampling_rate = sampling_rate
        self._dataset: Optional[Dataset] = None

    def load(self) -> Dataset:
        """
        Load and prepare dataset.

        Returns:
            Loaded dataset with audio cast to specified sampling rate.

        Raises:
            DatasetError: If dataset loading fails.
        """
        try:
            dataset = load_dataset(
                path=self.dataset_path,
                name=self.language,
                split=self.split_name,
                token=self.hf_token,
            )
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
            self._dataset = dataset
            return dataset
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}") from e

    @property
    def dataset(self) -> Dataset:
        """
        Get loaded dataset.

        Returns:
            Loaded dataset.

        Raises:
            DatasetError: If dataset is not loaded yet.
        """
        if self._dataset is None:
            raise DatasetError("Dataset not loaded. Call load() first.")
        return self._dataset

