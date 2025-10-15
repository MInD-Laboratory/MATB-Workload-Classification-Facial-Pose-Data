"""Configuration module for eye tracking processing pipeline.

Provides centralized configuration management for all processing parameters,
including directories, detection thresholds, and processing flags.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

# Load environment variables from .env file if present (for local development)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Check SciPy availability for filtering operations
try:
    from scipy.signal import butter, filtfilt, medfilt, resample_poly
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

@dataclass
class Config:
    """Eye tracking processing configuration.

    Attributes:
        RAW_DIR: Directory containing raw EyeLink CSV files
        OUT_BASE: Base directory for processed output
        PARTICIPANT_INFO_FILE: Filename of participant info CSV
        SCREEN_WIDTH: Screen width in pixels
        SCREEN_HEIGHT: Screen height in pixels
        SAMPLE_RATE: Eye tracker sampling rate in Hz
        WINDOW_SECONDS: Size of sliding window in seconds
        WINDOW_OVERLAP: Overlap fraction between consecutive windows (0-1)
        MISSING_MAX: Maximum proportion of missing data per window
        BLINK_Z_THRESH: Z-score threshold for blink detection
        BLINK_RAW_FLOOR: Minimum pupil size for valid data
        BLINK_MAX_DUR: Maximum blink duration in seconds
        FIXATION_MAX_DIST: Maximum distance for fixation (normalized units)
        FIXATION_MIN_DUR: Minimum fixation duration in seconds
        SACCADE_MIN_LEN: Minimum saccade length in samples
        SACCADE_VEL_THRESH: Velocity threshold for saccade detection
        SACCADE_ACC_THRESH: Acceleration threshold for saccade detection
        MEDIAN_FILTER_KERNEL: Kernel size for median filter
        OUTLIER_N_SD: Number of standard deviations for outlier clipping
        PUPIL_OUTLIER_THRESH: SD threshold for pupil size outliers
    """

    # Directory paths - can be overridden by environment variables
    _BASE_DIR: str = str(Path(__file__).parent.parent)
    RAW_DIR: str = os.getenv("EYELINK_RAW_DIR", str(Path(_BASE_DIR) / "data" / "eyelink_data"))
    OUT_BASE: str = os.getenv("EYELINK_OUT_BASE", str(Path(_BASE_DIR) / "data" / "processed"))

    # Participant info file - can be overridden by environment variable
    PARTICIPANT_INFO_FILE: str = os.getenv("PARTICIPANT_INFO_FILE", "participant_info.csv")

    # Screen parameters
    SCREEN_WIDTH: int = 2560
    SCREEN_HEIGHT: int = 1440

    # Sampling parameters
    SAMPLE_RATE: int = 1000  # Hz

    # Window parameters for metric extraction
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5
    MISSING_MAX: float = 0.25  # Max proportion missing data per window

    # Blink detection thresholds
    BLINK_Z_THRESH: float = -2.0
    BLINK_RAW_FLOOR: float = 30.0
    BLINK_MAX_DUR: float = 0.6  # seconds

    # Fixation detection thresholds
    FIXATION_MAX_DIST: float = 0.02  # normalized screen units
    FIXATION_MIN_DUR: float = 0.20  # seconds

    # Saccade detection thresholds
    SACCADE_MIN_LEN: int = 2  # samples
    SACCADE_VEL_THRESH: float = 0.5
    SACCADE_ACC_THRESH: float = 5.0

    # Filtering parameters
    MEDIAN_FILTER_KERNEL: int = 5
    OUTLIER_N_SD: float = 5.0
    PUPIL_OUTLIER_THRESH: float = 3.0  # SD for pupil outliers

    # Processing flags
    SAVE_NORMALIZED: bool = True
    SAVE_EVENTS: bool = False  # Save detected blinks/fixations/saccades
    OVERWRITE: bool = False

# Global configuration instance
CFG = Config()
