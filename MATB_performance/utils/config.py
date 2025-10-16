"""
Configuration for MATB performance data processing pipeline.

This module centralizes all configuration parameters for MATB performance
metric extraction, including paths, processing parameters, and output flags.
"""

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """MATB performance processing configuration.

    All parameters can be overridden via environment variables or by
    modifying this dataclass.
    """

    # === Path Configuration ===
    _BASE_DIR: str = str(Path(__file__).parent.parent)
    RAW_DIR: str = os.getenv(
        "MATB_RAW_DIR",
        str(Path(_BASE_DIR) / "data" / "matb_data")
    )
    OUT_BASE: str = os.getenv(
        "MATB_OUT_BASE",
        str(Path(_BASE_DIR) / "data" / "processed")
    )

    # Participant info file (shared with pose/eye tracking/ECG/GSR pipelines)
    PARTICIPANT_INFO_FILE: str = os.getenv(
        "PARTICIPANT_INFO_FILE",
        "participant_info.csv"
    )

    # === MATB Processing Parameters ===
    # Window size and overlap for windowed metric extraction
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5
    TOTAL_TIME: int = 480  # 8 minutes per session

    # === Output Flags ===
    SAVE_INDIVIDUAL: bool = True   # Save per-participant files
    SAVE_COMBINED: bool = True     # Save combined file


# Global configuration instance
CFG = Config()
