# Data Setup Guide

This document explains how to configure data paths for local development vs. published/shared usage.

## Overview

The pipeline uses **environment variables** to specify data locations, which allows:
- **Local development**: Point to OneDrive or other custom locations without copying data
- **Published usage**: Users automatically use standard `data/` directory structure
- **No code changes** needed when switching between development and published versions

## For Local Development (Team Members)

### Option 1: Using `.env` File (Recommended)

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** with your actual paths:
   ```bash
   # macOS example
   POSE_RAW_DIR=/Users/yourname/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Complexity in Action - Research/Mind and Interaction Dynamics/PNAS-MATB/pose_data
   POSE_OUT_BASE=/Users/yourname/Projects/MATB/Pose/data/processed
   PARTICIPANT_INFO_FILE=participant_info.csv
   ```

3. **Install python-dotenv** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

4. **Run pipeline** - it will automatically use your custom paths:
   ```bash
   cd Pose
   python process_pose_data.py
   ```

**Note**: The `.env` file is in `.gitignore` and will NOT be committed to the repository.

### Option 2: Shell Environment Variables

Set environment variables in your terminal before running:

```bash
# macOS/Linux
export POSE_RAW_DIR="/path/to/your/data/pose_data"
export POSE_OUT_BASE="/path/to/your/output/processed"
export PARTICIPANT_INFO_FILE="participant_info.csv"

# Then run pipeline
cd Pose
python process_pose_data.py
```

```powershell
# Windows PowerShell
$env:POSE_RAW_DIR="C:/path/to/your/data/pose_data"
$env:POSE_OUT_BASE="C:/path/to/your/output/processed"
$env:PARTICIPANT_INFO_FILE="participant_info.csv"

# Then run pipeline
cd Pose
python process_pose_data.py
```

## For Published/Shared Usage (End Users)

Users who download the published code and data should:

1. **Download data** to the standard location:
   ```
   [project-root]/
   ├── Pose/
   │   └── data/
   │       ├── pose_data/          # Raw pose CSVs here
   │       └── processed/          # Pipeline outputs here
   └── data/
       ├── participant_info.csv    # Participant metadata
       ├── gsr_data/              # GSR data (future)
       └── ecg_data/              # ECG data (future)
   ```

2. **DO NOT create a `.env` file** - The pipeline will automatically use default paths

3. **Run pipeline**:
   ```bash
   cd Pose
   python process_pose_data.py
   ```

## Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSE_RAW_DIR` | `[root]/Pose/data/pose_data` | Directory containing raw pose CSV files |
| `POSE_OUT_BASE` | `[root]/Pose/data/processed` | Directory for processed outputs |
| `PARTICIPANT_INFO_FILE` | `participant_info.csv` | Filename of participant metadata CSV |

## Participant Info File Location

The `participant_info.csv` file is searched in the following order:
1. Parent directory of `POSE_RAW_DIR` (OneDrive scenario)
2. `[root]/data/` directory (published scenario)
3. Current working directory (fallback)

## Verifying Your Setup

To verify your configuration is working:

```python
# In Python console or script
from Pose.utils.config import CFG
print(f"RAW_DIR: {CFG.RAW_DIR}")
print(f"OUT_BASE: {CFG.OUT_BASE}")
print(f"PARTICIPANT_INFO_FILE: {CFG.PARTICIPANT_INFO_FILE}")
```

## Troubleshooting

**Problem**: Pipeline can't find data files

**Solutions**:
- Check that your `.env` file has the correct absolute paths
- Verify the data files actually exist at those paths
- Check for typos in environment variable names (must be exact: `POSE_RAW_DIR`, not `POSE_DATA_DIR`)

**Problem**: python-dotenv not found

**Solutions**:
- Install it: `pip install python-dotenv`
- OR: Use shell environment variables instead (Option 2 above)

**Problem**: `.env` file changes are committed to git

**Solutions**:
- Make sure `.env` is in `.gitignore` (it already is)
- If you accidentally committed it: Remove from git and use `.env.example` instead
- Only `.env.example` should be committed (as a template)

## Development Workflow

**During development**:
1. Keep `.env` file for your local paths (OneDrive)
2. Work normally - no need to copy data
3. `.env` is NOT committed to git

**Before publishing**:
1. No code changes needed!
2. Just document the expected data structure for users
3. Users will set up their own data directory
4. Pipeline automatically uses defaults when no `.env` exists

## Questions?

Contact the project maintainers if you have questions about data setup.
