"""
Configuration for ECG data processing pipeline.

This module centralizes all configuration parameters for ECG processing,
including paths, processing parameters, and output flags.
"""

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for NeuroKit2 availability
try:
    import neurokit2 as nk
    NEUROKIT2_AVAILABLE = True
except ImportError:
    NEUROKIT2_AVAILABLE = False
    print("WARNING: NeuroKit2 not available. Install with: pip install neurokit2")


@dataclass
class Config:
    """ECG processing configuration.

    All parameters can be overridden via environment variables or by
    modifying this dataclass.
    """

    # === Path Configuration ===
    _BASE_DIR: str = str(Path(__file__).parent.parent)
    RAW_DIR: str = os.getenv(
        "ECG_RAW_DIR",
        str(Path(_BASE_DIR) / "data" / "ecg_data")
    )
    OUT_BASE: str = os.getenv(
        "ECG_OUT_BASE",
        str(Path(_BASE_DIR) / "data" / "processed")
    )

    # Participant info file (shared with pose/eye tracking pipelines)
    PARTICIPANT_INFO_FILE: str = os.getenv(
        "PARTICIPANT_INFO_FILE",
        "participant_info.csv"
    )

    # === ECG Processing Parameters ===
    # Sampling rate for Zephyr BioHarness
    SAMPLE_RATE: int = 250  # Hz

    # ECG signal cleaning method
    # Options: 'neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002',
    #          'elgendi2010', 'engzeemod2012', 'vg'
    CLEANING_METHOD: str = "engzeemod2012"

    # R-peak detection method
    # Options: 'neurokit', 'pantompkins1985', 'hamilton2002', 'zong2003',
    #          'martinez2004', 'christov2004', 'gamboa2008', 'elgendi2010',
    #          'engzeemod2012', 'manikandan2012', 'kalidas2017', 'nabian2018',
    #          'rodrigues2021', 'promac'
    PEAK_METHOD: str = "engzeemod2012"

    # Signal quality assessment method
    # Options: 'averageQRS', 'zhao2018'
    QUALITY_METHOD: str = "averageQRS"

    # Signal quality approach
    # Options: 'simple', 'fuzzy'
    QUALITY_APPROACH: str = "fuzzy"

    # Heart rate interpolation method
    # Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
    #          'previous', 'next', 'monotone_cubic'
    INTERPOLATION_METHOD: str = "monotone_cubic"

    # === Window Parameters ===
    # Window size and overlap for windowed HRV feature extraction
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5

    # === Output Flags ===
    SAVE_SIGNALS: bool = True      # Save cleaned signals, R-peaks, HR
    SAVE_FEATURES: bool = True     # Save HRV features
    PLOT_SIGNALS: bool = False     # Plot during processing (slow)
    BASELINE_CORRECTION: bool = False  # Apply baseline correction

    # === Validation ===
    # Required columns in ECG data
    REQUIRED_ECG_COLS: list = None

    def __post_init__(self):
        """Initialize computed fields and validate configuration."""
        if self.REQUIRED_ECG_COLS is None:
            self.REQUIRED_ECG_COLS = ['EcgWaveform']  # Main ECG waveform column

        # Validate NeuroKit2 availability
        if not NEUROKIT2_AVAILABLE:
            raise ImportError(
                "NeuroKit2 is required for ECG processing. "
                "Install with: pip install neurokit2"
            )


# Global configuration instance
CFG = Config()
