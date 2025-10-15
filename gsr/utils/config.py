"""
Configuration for GSR data processing pipeline.

This module centralizes all configuration parameters for GSR/EDA processing,
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
    """GSR processing configuration.

    All parameters can be overridden via environment variables or by
    modifying this dataclass.
    """

    # === Path Configuration ===
    _BASE_DIR: str = str(Path(__file__).parent.parent)
    RAW_DIR: str = os.getenv(
        "GSR_RAW_DIR",
        str(Path(_BASE_DIR) / "data" / "gsr_data")
    )
    OUT_BASE: str = os.getenv(
        "GSR_OUT_BASE",
        str(Path(_BASE_DIR) / "data" / "processed")
    )

    # Participant info file (shared with pose/eye tracking/ECG pipelines)
    PARTICIPANT_INFO_FILE: str = os.getenv(
        "PARTICIPANT_INFO_FILE",
        "participant_info.csv"
    )

    # === GSR Processing Parameters ===
    # Sampling rate for Shimmer device
    SAMPLE_RATE: int = 20  # Hz

    # EDA signal cleaning method
    # Options: 'neurokit', 'biosppy'
    CLEANING_METHOD: str = "neurokit"

    # Phasic/tonic decomposition method
    # Options: 'highpass' (default), 'cvxEDA', 'smoothmedian'
    PHASIC_METHOD: str = "highpass"

    # SCR peak detection method
    # Options: 'neurokit', 'gamboa2008', 'kim2004', 'vanhalem2020', 'nabian2018'
    PEAK_METHOD: str = "neurokit"

    # === Window Parameters ===
    # (for windowed analysis - optional)
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5

    # === Output Flags ===
    SAVE_SIGNALS: bool = True      # Save cleaned signals, SCR, SCL
    SAVE_FEATURES: bool = True     # Save EDA features
    PLOT_SIGNALS: bool = False     # Plot during processing (slow)

    # === Validation ===
    # Required columns in GSR data
    REQUIRED_GSR_COLS: list = None

    def __post_init__(self):
        """Initialize computed fields and validate configuration."""
        if self.REQUIRED_GSR_COLS is None:
            # Main GSR column from Shimmer device
            self.REQUIRED_GSR_COLS = ['Shimmer_AD66_GSR_Skin_Conductance_CAL']

        # Validate NeuroKit2 availability
        if not NEUROKIT2_AVAILABLE:
            raise ImportError(
                "NeuroKit2 is required for GSR processing. "
                "Install with: pip install neurokit2"
            )


# Global configuration instance
CFG = Config()
