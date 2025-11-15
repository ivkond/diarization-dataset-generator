"""Pydantic models for configuration validation."""

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    path: str
    language: str
    split_name: str
    feature_speaker_id: str
    feature_text: str
    feature_audio: str


class OutputConfig(BaseModel):
    """Output configuration."""

    path: str
    repo_id: str = ""  # Optional: if empty, save to Parquet files; if set, upload to Hub
    parquet_file_size_mb: float = Field(gt=0, description="Parquet file size limit in MB")


class PauseConfig(BaseModel):
    """Pause configuration."""

    min_ms: float = Field(gt=0)
    max_ms: float = Field(gt=0)

    @field_validator("max_ms")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max_ms is greater than min_ms."""
        if "min_ms" in info.data and v <= info.data["min_ms"]:
            raise ValueError("max_ms must be greater than min_ms")
        return v


class LongPauseConfig(BaseModel):
    """Long pause configuration."""

    probability: float = Field(ge=0, le=1)
    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max is greater than min."""
        if "min" in info.data and v <= info.data["min"]:
            raise ValueError("max must be greater than min")
        return v


class ShortSegmentConfig(BaseModel):
    """Short segment configuration."""

    probability: float = Field(ge=0, le=1)
    max_duration: float = Field(gt=0)


class TrackConfig(BaseModel):
    """Track structure configuration."""

    min_duration: float = Field(gt=0)
    max_duration: float = Field(gt=0)
    pause: PauseConfig
    long_pause: LongPauseConfig
    short_segment: ShortSegmentConfig

    @field_validator("max_duration")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max_duration is greater than min_duration."""
        if "min_duration" in info.data and v <= info.data["min_duration"]:
            raise ValueError("max_duration must be greater than min_duration")
        return v


class VolumeConfig(BaseModel):
    """Volume configuration."""

    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max is greater than min."""
        if "min" in info.data and v <= info.data["min"]:
            raise ValueError("max must be greater than min")
        return v


class SpeakerConfig(BaseModel):
    """Speakers configuration."""

    min: int = Field(gt=0)
    max: int = Field(gt=0)
    target_count: int = Field(gt=0)
    max_consecutive: int = Field(gt=0)
    volume: VolumeConfig

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: int, info) -> int:
        """Validate that max is greater than min."""
        if "min" in info.data and v <= info.data["min"]:
            raise ValueError("max must be greater than min")
        return v


class OverlapsConfig(BaseModel):
    """Overlaps configuration."""

    probability: float = Field(ge=0, le=1)
    min_percent: float = Field(ge=0, le=1)
    max_percent: float = Field(ge=0, le=1)

    @field_validator("max_percent")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max_percent is greater than min_percent."""
        if "min_percent" in info.data and v <= info.data["min_percent"]:
            raise ValueError("max_percent must be greater than min_percent")
        return v


class SimultaneousSpeechConfig(BaseModel):
    """Simultaneous speech configuration."""

    probability: float = Field(ge=0, le=1)
    min_duration: float = Field(gt=0)
    max_duration: float = Field(gt=0)

    @field_validator("max_duration")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max_duration is greater than min_duration."""
        if "min_duration" in info.data and v <= info.data["min_duration"]:
            raise ValueError("max_duration must be greater than min_duration")
        return v


class SNRConfig(BaseModel):
    """SNR configuration."""

    min: float
    max: float = Field(gt=0)

    @field_validator("max")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        """Validate that max is greater than min."""
        if "min" in info.data and v <= info.data["min"]:
            raise ValueError("max must be greater than min")
        return v


class NoiseConfig(BaseModel):
    """Background noise configuration."""

    probability: float = Field(ge=0, le=1)
    types: List[str]
    snr: SNRConfig


class Config(BaseModel):
    """Main configuration model."""

    random_seed: int
    track_count: int = Field(gt=0)
    dataset: DatasetConfig
    output: OutputConfig
    track: TrackConfig
    speakers: SpeakerConfig
    overlaps: OverlapsConfig
    simultaneous_speech: SimultaneousSpeechConfig
    noise: NoiseConfig
    conversation_patterns: Dict[str, float] = Field(
        description="Conversation pattern probabilities"
    )
    difficulty_distribution: Dict[str, float] = Field(
        description="Difficulty distribution probabilities"
    )

    @field_validator("conversation_patterns")
    @classmethod
    def validate_patterns_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that conversation pattern probabilities sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"conversation_patterns probabilities must sum to 1.0, got {total}"
            )
        return v

    @field_validator("difficulty_distribution")
    @classmethod
    def validate_difficulty_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that difficulty distribution probabilities sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"difficulty_distribution probabilities must sum to 1.0, got {total}"
            )
        return v

