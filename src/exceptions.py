"""Custom exceptions for the diarization dataset generator."""


class DatasetGeneratorError(Exception):
    """Base exception for all dataset generator errors."""

    pass


class ConfigurationError(DatasetGeneratorError):
    """Raised when there's an error with configuration."""

    pass


class DatasetError(DatasetGeneratorError):
    """Raised when there's an error loading or processing the dataset."""

    pass


class AudioProcessingError(DatasetGeneratorError):
    """Raised when there's an error processing audio."""

    pass


class TrackGenerationError(DatasetGeneratorError):
    """Raised when there's an error generating a track."""

    pass

