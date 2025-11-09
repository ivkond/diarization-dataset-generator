"""Constants used throughout the application."""

# Audio constants
SAMPLING_RATE = 16000  # Fixed sampling rate: 16 kHz

# Pattern selection constants (used in conversation patterns)
INTERVIEW_QUESTION_PROBABILITY = 0.8  # Probability of interviewer asking after answer
GROUP_DISCUSSION_CONTINUE_PROBABILITY = 0.3  # Probability of same speaker continuing
STANDARD_CONTINUE_PROBABILITY = 0.35  # Standard probability of same speaker continuing

# Audio processing constants
AUDIO_TOLERANCE = 1e-6  # Tolerance for audio comparison
AUDIO_NORMAL_RANGE = 1.0  # Normal audio range [-1, 1]

# Validation constants
DEFAULT_MIN_SPEAKER_DURATION = 3.0  # Minimum duration for each speaker (seconds)
DEFAULT_MAX_IMBALANCE_RATIO = 0.4  # Maximum ratio of dominant speaker time to total

# Noise generation constants
PINK_NOISE_FILTER_COEFFS = (
    0.99886,
    0.99332,
    0.96900,
    0.86650,
    0.55000,
    0.7616,
    -0.00360,
)
PINK_NOISE_FEEDBACK_COEFFS = (-0.49448, 0.01681, 0.00972, 0.00001, -0.00360)
STATIC_NOISE_SPIKE_RATIO = 100  # One spike per 100 samples
STATIC_NOISE_SPIKE_MIN = 2.0
STATIC_NOISE_SPIKE_MAX = 5.0
HUM_FREQUENCY_BASE = 50.0  # Base frequency for hum noise (Hz)
HUM_FREQUENCY_VARIATION = 5.0  # Variation in frequency
HUM_HARMONIC_2_AMPLITUDE = 0.3
HUM_HARMONIC_3_AMPLITUDE = 0.1

# Metadata estimation
METADATA_OVERHEAD_BYTES = 1000  # Rough estimate for metadata overhead in Parquet

