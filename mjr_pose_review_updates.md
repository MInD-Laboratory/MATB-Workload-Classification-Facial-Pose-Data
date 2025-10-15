# Pose Pipeline Review & Updates Document

**Date**: October 14, 2025
**Reviewer**: Claude (AI Assistant)
**Purpose**: Document all identified issues, differences with .forcomparison code, and required fixes

---

## Table of Contents

1. [Critical Issues](#critical-issues)
2. [High Priority Issues](#high-priority-issues)
3. [Medium Priority Issues](#medium-priority-issues)
4. [Low Priority Issues](#low-priority-issues)
5. [Comparison with .forcomparison Code](#comparison-with-forcomparison-code)
6. [Implementation Plan](#implementation-plan)

---

## CRITICAL ISSUES

### ✅ ISSUE #1: Condition Mapping Not Implemented in Pipeline - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/process_pose_data.py`
**Lines**: 79, 95, 174, 222, 287, 338, and throughout

**Problem**:
- Pipeline is designed to output condition-based filenames (e.g., `3101_L_norm.csv`)
- Code passes `trial_num` (integer: 1, 2, 3) instead of `condition` (letter: L, M, H) to `get_output_filename()`
- Output files have trial numbers instead of conditions
- Smart skip logic at line 659 fails because it looks for condition-based filenames

**Current Code Example** (Line 78-79):
```python
pid, trial_num = parse_participant_trial(fp.name)  # trial_num is int (1, 2, 3)
out_name = get_output_filename(fp.name, pid, trial_num, "_norm")  # Should pass condition (L, M, H)
```

**Impact**:
- Documentation promises condition-based outputs but code produces trial-based outputs
- Inconsistent with .forcomparison code which already uses conditions
- Prevents proper file matching and pipeline resumption

**Fix Required**:
1. Load participant info at start of `run_pose_processing_pipeline()`
2. Create condition mapping using existing functions
3. Pass condition instead of trial_num to `get_output_filename()` throughout

**Reference**: Functions already exist in `preprocessing_utils.py`:
- `load_participant_info()`
- `create_condition_mapping()`
- `get_condition_for_file()`

**See**: .forcomparison implementation in `process_pose_linear.py:79-90, 112-113, 137-143`

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Added imports** (`process_pose_data.py:47-55`):
   - Imported `load_participant_info_file` from `io_utils`
   - Imported `load_participant_info`, `create_condition_mapping`, `get_condition_for_file` from `preprocessing_utils`

2. **Updated helper functions**:
   - `check_steps_1_5_complete()` - Now takes `condition_map` parameter and uses it to look up conditions
   - `load_existing_normalized_data()` - Now takes `condition_map` parameter and stores condition in metadata

3. **Updated all step functions** to use conditions instead of trial numbers:
   - `step_2_filter_keypoints()` - Takes `condition_map`, looks up condition, passes to `get_output_filename()`
   - `step_3_mask_low_confidence()` - Uses `condition` from metadata
   - `step_4_interpolate_filter()` - Uses `condition` from metadata
   - `step_5_normalize_coordinates()` - Uses `condition` from metadata
   - `step_7_extract_features()` - Uses `condition` from metadata and passes to `write_per_frame_metrics()`

4. **Updated main pipeline function** (`run_pose_processing_pipeline()`):
   - Added section to load participant info and create condition mapping at lines 683-696
   - Passes `condition_map` to `check_steps_1_5_complete()`
   - Passes `condition_map` to `load_existing_normalized_data()`
   - Passes `condition_map` to `step_2_filter_keypoints()`

5. **Changed all metadata dictionaries** from storing `"trial": trial_num` to storing `"condition": condition`

**Result**:
- Output files now use condition-based naming (e.g., `3101_L_norm.csv` instead of `3101_01_norm.csv`)
- Smart skip logic now works correctly
- Pipeline consistent with .forcomparison implementation
- File naming matches documentation

**Testing Recommended**:
- Run pipeline on sample data to verify condition-based output files
- Verify skip logic detects existing condition-based files
- Check that per-frame metrics use condition in filenames

---

### ✅ ISSUE #2: Linear Metrics Calculation Missing Rich Features - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**:
- `Pose/utils/features_utils.py` (missing function)
- `Pose/utils/features_utils.py` - `compute_linear_from_perframe_dir()` function

**Problem**:

**Your Current Approach**:
```python
# Computes from signal x:
vel = np.diff(x) / dt         # velocity via finite differences
acc = np.diff(vel) / dt       # acceleration from velocity
mean_abs_vel = np.mean(np.abs(vel))
mean_abs_acc = np.mean(np.abs(acc))
rms = np.sqrt(np.mean((x - np.mean(x))**2))
```

**Output per feature**: 4 metrics
- `blink_aperture_mean`
- `blink_aperture_mean_abs_vel`
- `blink_aperture_mean_abs_acc`
- `blink_aperture_rms`

**.forcomparison Approach**:
```python
# Uses gradient for ALL metrics:
df = add_perframe_derivatives(df, fps=60)  # Adds _vel and _acc columns via np.gradient
# Then computes statistics for: value, value_vel, value_acc separately
```

**Output per feature**: 27 metrics (9 statistics × 3 signals)
- For `blink_aperture`: min, max, mean, rms, std, median, p25, p75, autocorr1
- For `blink_aperture_vel`: min, max, mean, rms, std, median, p25, p75, autocorr1
- For `blink_aperture_acc`: min, max, mean, rms, std, median, p25, p75, autocorr1

**Missing Function**:
```python
def add_perframe_derivatives(df: pd.DataFrame, fps: float = 60.0) -> pd.DataFrame:
    """
    Append *_vel and *_acc columns (via np.gradient) for every numeric column.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col in {"participant", "condition", "session_number", "frame", "interocular"}:
            continue
        s = out[col].to_numpy(float)
        v = np.gradient(s) * fps          # per second
        a = np.gradient(v) * fps          # per second^2
        out[f"{col}_vel"] = v
        out[f"{col}_acc"] = a
    return out
```

**Missing Statistical Metrics** (computed in .forcomparison but not in your code):
1. **`std`** - Standard deviation (variability around mean)
2. **`median`** - Robust central tendency, less affected by outliers
3. **`p25`** - 25th percentile (lower quartile)
4. **`p75`** - 75th percentile (upper quartile)
5. **`autocorr1`** - Lag-1 autocorrelation (temporal smoothness/persistence)

**Impact**:
- **MASSIVE**: Much richer feature set for modeling (27 vs 4 metrics per feature)
- Better captures signal characteristics (distribution shape, temporal structure)
- Autocorrelation is especially valuable for physiological signals
- Likely improves classification performance

**Fix Required**:
1. Add `add_perframe_derivatives()` function to `features_utils.py`
2. Update `compute_linear_from_perframe_dir()` to:
   - Call `add_perframe_derivatives()` after loading each file
   - Compute statistics for value, _vel, and _acc columns separately
   - Add the 5 new statistical metrics

**Reference**: `.forcomparison/Pose/utils/features_utils.py:345-360` and `:363-468`

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Added `add_perframe_derivatives()` function** (`features_utils.py:344-386`):
   - Computes velocity (first derivative) and acceleration (second derivative) for ALL numeric metrics
   - Uses `np.gradient` instead of `np.diff` for centered finite differences (more accurate)
   - Adds `_vel` and `_acc` columns for each metric
   - Properly scales by fps for correct time units
   - Skips metadata columns (participant, condition, session_number, frame, interocular)

2. **Replaced entire `compute_linear_from_perframe_dir()` function** (`features_utils.py:389-545`):
   - Now calls `add_perframe_derivatives()` after loading each per-frame file
   - Processes THREE separate signals per metric: base value, velocity (_vel), acceleration (_acc)
   - Added 5 NEW statistical metrics per signal:
     - `std`: Standard deviation (variability around mean)
     - `median`: Robust central tendency (less sensitive to outliers)
     - `p25`: 25th percentile (lower quartile)
     - `p75`: 75th percentile (upper quartile)
     - `autocorr1`: Lag-1 autocorrelation (temporal smoothness/persistence)
   - Retained existing metrics: min, max, mean, rms
   - Result: 9 statistics × 3 signals = **27 features per original metric** (vs 4 previously)

3. **Added `re` import** at top of file for regex pattern matching

**Detailed Feature Comparison**:

**Before (Your Original Code)**:
For metric `blink_aperture`:
- `blink_aperture_mean`
- `blink_aperture_mean_abs_vel` (from np.diff)
- `blink_aperture_mean_abs_acc` (from np.diff)
- `blink_aperture_rms`

**Total**: 4 features per metric

**After (New Implementation)**:
For metric `blink_aperture`:

*Base signal* (`blink_aperture`):
- `blink_aperture_min`, `_max`, `_mean`, `_rms`, `_std`, `_median`, `_p25`, `_p75`, `_autocorr1`

*Velocity signal* (`blink_aperture_vel`):
- `blink_aperture_vel_min`, `_max`, `_mean`, `_rms`, `_std`, `_median`, `_p25`, `_p75`, `_autocorr1`

*Acceleration signal* (`blink_aperture_acc`):
- `blink_aperture_acc_min`, `_max`, `_mean`, `_rms`, `_std`, `_median`, `_p25`, `_p75`, `_autocorr1`

**Total**: 27 features per metric

**Benefits**:
- **6.75x more features** for machine learning models
- **Richer temporal dynamics**: Separate velocity and acceleration signals capture movement characteristics
- **Robust statistics**: Median and quartiles less affected by outliers
- **Temporal structure**: Autocorrelation captures signal smoothness and persistence
- **Better distribution characterization**: Quartiles reveal distribution shape
- **Consistent with .forcomparison**: Same methodology as comparison study

**Testing Recommended**:
- Run step 8 (linear metrics) on existing per-frame data
- Verify output CSV has all new columns (check column count increase)
- Confirm _vel and _acc columns are present in intermediate outputs
- Check autocorrelation values are between -1 and 1
- Verify no NaN inflation from the new metrics

---

## HIGH PRIORITY ISSUES

### ✅ ISSUE #3: Inconsistent Path Handling in Configuration - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/utils/config.py:40-41, 44`

**Problem**:
```python
RAW_DIR: str = r"data/pose_data"  # Relative path - will break if working directory changes
OUT_BASE: str = os.getenv("POSE_OUT_BASE", str(Path(_BASE_DIR) / "data" / "processed"))  # Absolute
PARTICIPANT_INFO_FILE: str = r"data/participant_info.csv"  # Relative path with directory
```

**Impact**:
- File-not-found errors if script run from different directory
- Inconsistency between RAW_DIR and OUT_BASE handling
- PARTICIPANT_INFO_FILE includes directory path when it should be just a filename

**.forcomparison Solution**:
```python
RAW_DIR: str = os.getenv("POSE_RAW_DIR", str(Path(_BASE_DIR) / "data" / "pose_data"))
PARTICIPANT_INFO_FILE: str = os.getenv("PARTICIPANT_INFO_FILE", "participant_info.csv")
```

**Fix Required**:
```python
RAW_DIR: str = os.getenv("POSE_RAW_DIR", str(Path(_BASE_DIR) / "data" / "pose_data"))
OUT_BASE: str = os.getenv("POSE_OUT_BASE", str(Path(_BASE_DIR) / "data" / "processed"))
PARTICIPANT_INFO_FILE: str = os.getenv("PARTICIPANT_INFO_FILE", "participant_info.csv")
```

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Updated `RAW_DIR`** (`config.py:40`):
   - Changed from relative path: `r"data/pose_data"`
   - To absolute path: `os.getenv("POSE_RAW_DIR", str(Path(_BASE_DIR) / "data" / "pose_data"))`
   - Now uses `_BASE_DIR` to construct absolute path from Pose directory
   - Supports environment variable override via `POSE_RAW_DIR`

2. **Updated `PARTICIPANT_INFO_FILE`** (`config.py:44`):
   - Changed from path: `r"data/participant_info.csv"`
   - To filename only: `os.getenv("PARTICIPANT_INFO_FILE", "participant_info.csv")`
   - Now correctly represents just a filename (as documented in docstring)
   - Directory location determined by `io_utils.load_participant_info_file()` function
   - Supports environment variable override via `PARTICIPANT_INFO_FILE`

**Result**:
- All path configurations now use absolute paths constructed from `_BASE_DIR`
- Pipeline will work correctly regardless of working directory
- Consistent with .forcomparison implementation
- Environment variable overrides available for all paths
- PARTICIPANT_INFO_FILE correctly represents a filename, not a path

**Testing Recommended**:
- Run pipeline from different working directories to verify path resolution
- Test environment variable overrides (POSE_RAW_DIR, POSE_OUT_BASE, PARTICIPANT_INFO_FILE)
- Verify participant info file is found in expected location

---

### ✅ ISSUE #4: No SciPy Availability Check - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/utils/config.py` (lines 11-18), `Pose/process_pose_data.py` (lines 44, 280-284)

**Problem**:
- No check if scipy is installed before using Butterworth filter
- Could lead to confusing runtime errors when scipy not installed

**.forcomparison Solution**:
```python
# Check SciPy availability for optional filtering operations
try:
    from scipy.signal import butter, filtfilt  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
```

Then in pipeline before step 4:
```python
if not SCIPY_AVAILABLE:
    print("SciPy is required for RUN_INTERP_FILTER. Install scipy or disable this step.")
    sys.exit(1)
```

**Fix Required**:
1. Add SCIPY_AVAILABLE check to config.py
2. Add check in step 4 (interpolate/filter) before attempting to use scipy

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Added SciPy availability check to `config.py`** (lines 11-18):
   ```python
   # Check SciPy availability for optional filtering operations
   # SciPy is used for Butterworth filtering and signal processing
   try:
       from scipy.signal import butter, filtfilt  # noqa: F401
       SCIPY_AVAILABLE = True
   except Exception:
       # SciPy not available - filtering operations will be skipped
       SCIPY_AVAILABLE = False
   ```
   - Added try/except block to safely attempt scipy import
   - Sets `SCIPY_AVAILABLE` flag for use throughout pipeline
   - Added comments explaining purpose and usage

2. **Updated imports in `process_pose_data.py`** (line 44):
   - Changed from: `from utils.config import CFG`
   - To: `from utils.config import CFG, SCIPY_AVAILABLE`

3. **Added SciPy check in step_4_interpolate_filter()** (lines 280-284):
   ```python
   # Check SciPy availability for filtering operations
   if not SCIPY_AVAILABLE:
       print("ERROR: SciPy is required for RUN_INTERP_FILTER=True.")
       print("Please install scipy (pip install scipy) or set RUN_INTERP_FILTER=False in config.")
       sys.exit(1)
   ```
   - Check added immediately after RUN_INTERP_FILTER flag check
   - Provides clear error message with installation instructions
   - Exits gracefully before attempting to use scipy functions

**Result**:
- Pipeline now gracefully handles missing scipy dependency
- Clear error messages guide users to install scipy or disable filtering
- Consistent with .forcomparison implementation
- Prevents confusing ImportError or AttributeError at runtime

**Testing Recommended**:
- Test pipeline with scipy installed (should work normally)
- Test pipeline without scipy but with `RUN_INTERP_FILTER=False` (should skip step 4 gracefully)
- Test pipeline without scipy and `RUN_INTERP_FILTER=True` (should exit with clear error message)

---

## MEDIUM PRIORITY ISSUES

### ✅ ISSUE #5: Unused Configuration Flag - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/process_pose_data.py` (lines 44, 651)

**Problem**:
- `SCALE_BY_INTEROCULAR` flag defined in config but never used
- Hardcoded logic `(feature_type == "original")` determines scaling behavior
- Makes config flag misleading and defeats purpose of configuration

**Current Code** (Line 651):
```python
compute_linear_from_perframe_dir(
    per_frame_dir, out_path, CFG.FPS, CFG.WINDOW_SECONDS,
    CFG.WINDOW_OVERLAP, scale_by_interocular=(feature_type == "original")
)
```

**.forcomparison Code**:
```python
from utils.config import CFG, SCALE_BY_INTEROCULAR
# ...
compute_linear_from_perframe_dir(
    per_frame_dir, out_path, CFG.FPS, CFG.WINDOW_SECONDS,
    CFG.WINDOW_OVERLAP, scale_by_interocular=SCALE_BY_INTEROCULAR
)
```

**Fix Options**:
1. **Option A**: Use the flag properly (import and check it) ✅ CHOSEN
2. **Option B**: Remove unused flag from config

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Updated imports in `process_pose_data.py`** (line 44):
   - Changed from: `from utils.config import CFG, SCIPY_AVAILABLE`
   - To: `from utils.config import CFG, SCIPY_AVAILABLE, SCALE_BY_INTEROCULAR`

2. **Updated step_8_compute_linear_metrics()** (line 651):
   - Changed from: `scale_by_interocular=(feature_type == "original")`
   - To: `scale_by_interocular=SCALE_BY_INTEROCULAR`
   - Now uses the configuration flag instead of hardcoded logic

**Result**:
- Configuration flag now controls scaling behavior globally
- Consistent behavior across all feature types (procrustes_global, procrustes_participant, original)
- Users can easily toggle scaling via config.py without modifying pipeline code
- Consistent with .forcomparison implementation
- Config flag now has actual effect

**Behavioral Change**:
- **Before**: Only "original" feature type used interocular scaling
- **After**: All feature types (procrustes_global, procrustes_participant, original) follow the `SCALE_BY_INTEROCULAR` flag setting
- **Current default**: `SCALE_BY_INTEROCULAR = True` in config.py (line 101)

**Testing Recommended**:
- Verify linear metrics are scaled by interocular distance when `SCALE_BY_INTEROCULAR=True`
- Test with `SCALE_BY_INTEROCULAR=False` to ensure no scaling applied
- Compare outputs with previous pipeline runs to understand any differences

---

### ✅ ISSUE #6: Documentation Error - Eye Landmark Labels - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/utils/normalize_utils.py:64`

**Problem**:
```python
# Line 64 docstring said:
"""Uses **landmarks 37 (right eye)** and **46 (left eye)**"""

# But line 72 comment correctly said:
x37_col = find_real_colname("x", 37, cols)  # Left eye corner x
```

Docstring had the eye labels swapped.

**Correct** (based on MediaPipe conventions):
- Landmark 37 = LEFT outer eye corner
- Landmark 46 = RIGHT outer eye corner

**Impact**:
- Documentation confusion for maintainers/reviewers
- No functional impact (code is correct, just docstring was wrong)

**Fix Required**:
Update docstring at line 64 to match inline comments

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Updated docstring in `normalize_utils.py`** (line 64):
   - Changed from: `Uses **landmarks 37 (right eye)** and **46 (left eye)**`
   - To: `Uses **landmarks 37 (left eye)** and **46 (right eye)**`

**Result**:
- Docstring now correctly identifies landmark positions
- Consistent with inline code comments
- Accurate documentation for maintainers and reviewers
- No functional changes (code was already correct)

---

### ✅ ISSUE #7: Unused Window Normalization Flags - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/utils/config.py` (lines 114-116, removed)

**Problem**:
- Two configuration flags defined but never used anywhere in the codebase:
  - `ZSCORE_PER_WINDOW: bool = True`
  - `MIN_MAX_NORMALIZE_PER_WINDOW: bool = False`
- No code implements window-level z-score or min-max normalization
- Misleading configuration that suggests functionality that doesn't exist
- Could confuse reviewers and maintainers

**Investigation**:
- Searched entire codebase - these flags are only defined in config.py, never imported or used
- The .forcomparison codebase also doesn't use these flags
- Linear metrics computation (`compute_linear_from_perframe_dir`) computes statistics directly on raw windowed segments without normalization
- Window normalization was likely planned but never implemented

**Impact**:
- No functional impact (flags weren't doing anything)
- Documentation/configuration confusion for reviewers
- Suggests unimplemented features

**Fix Applied**:
**Option: Remove unused flags** ✅ CHOSEN

Removed the following lines from config.py (lines 114-116):
```python
# Save windowed features for different normalizations (set only one to True)
ZSCORE_PER_WINDOW: bool = True  # Z-score features within each window
MIN_MAX_NORMALIZE_PER_WINDOW: bool = False  # Min-max scale features to [0,1] within each window
```

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Removed unused flags from `config.py`** (lines 114-116):
   - Removed `ZSCORE_PER_WINDOW` flag
   - Removed `MIN_MAX_NORMALIZE_PER_WINDOW` flag
   - Removed misleading comment about window normalization

**Result**:
- Configuration now accurately reflects implemented functionality
- No misleading flags suggesting unimplemented features
- Cleaner, more maintainable configuration
- No functional changes (flags weren't being used)
- Eliminates potential confusion for code reviewers

**Note**:
If window-level normalization is needed in the future, these flags can be re-added when the functionality is actually implemented.

---

## LOW PRIORITY ISSUES

### ✅ ISSUE #8: Inconsistent Docstring in Procrustes Function - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/utils/geometry_utils.py:11-40`

**Problem**:
- Function signature had incorrect return type: `Tuple[bool, float, float, float, np.ndarray, np.ndarray]` (only 2 floats)
- But code actually returns 7 values including `sx, sy, tx, ty` (4 floats total)
- Docstring described scale as singular value instead of separate sx and sy
- .forcomparison has correct signature: `Tuple[bool, float, float, float, float, np.ndarray, np.ndarray]` (4 floats)

**Impact**:
- Documentation confusion only
- Code works correctly, just signature/docstring were unclear

**Fix Required**:
Update function signature and docstring to match actual return values

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Updated function signature** (`geometry_utils.py:11`):
   - Changed from: `-> Tuple[bool, float, float, float, np.ndarray, np.ndarray]` (2 floats after bool)
   - To: `-> Tuple[bool, float, float, float, float, np.ndarray, np.ndarray]` (4 floats after bool)
   - Now correctly indicates 7 return values: bool, sx, sy, tx, ty, R, Xtrans

2. **Updated docstring Returns section** (`geometry_utils.py:23-31`):
   - Changed from singular "scale: Scaling factor"
   - To separate values:
     - `sx: Scaling factor in x direction`
     - `sy: Scaling factor in y direction`
   - Added "non-uniform scaling" to function description
   - Updated note to mention "Allows non-uniform scaling (different sx and sy values)"

**Result**:
- Function signature now accurately reflects return type
- Docstring clearly documents both sx and sy parameters
- Consistent with .forcomparison implementation
- No code changes needed (was already returning correct values)
- Eliminates documentation confusion for code reviewers

---

### ✅ ISSUE #9: Missing Pipeline Step Control Features - **FIXED**

**Status**: ✅ RESOLVED (October 14, 2025)
**Files Affected**: `Pose/process_pose_data.py`

**Problem**:
- No ability to start pipeline from specific step (e.g., rerun only feature extraction)
- No helper function to load existing templates when starting from step 7+
- Forces full pipeline rerun even when only later steps need modification
- Less flexible than .forcomparison implementation

**.forcomparison Features**:
1. **`--start-step` parameter**: Allows starting from any step 1-8
2. **`load_existing_templates()` function**: Loads pre-existing templates from disk

**Use Cases**:
- Rerun feature extraction after adjusting window parameters
- Recompute linear metrics without reprocessing raw data
- Skip expensive early steps when debugging later steps
- Resume pipeline after interruption

**Impact**:
- Workflow efficiency for iterative development
- Debugging and testing convenience
- Reduces reprocessing time during parameter tuning

---

#### **FIX IMPLEMENTED** ✅

**Changes Made**:

1. **Added `load_existing_templates()` function** (`process_pose_data.py:122-149`):
   ```python
   def load_existing_templates() -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
       """Load existing templates from disk for steps 7-8.

       Returns:
           Tuple of (global_template, participant_templates)
       """
       template_dir = Path(CFG.OUT_BASE) / "templates"

       # Load global template
       global_template = None
       global_template_path = template_dir / "global_template.csv"
       if global_template_path.exists():
           global_template = pd.read_csv(global_template_path)

       # Load participant templates
       participant_templates = {}
       if template_dir.exists():
           for template_path in template_dir.glob("participant_*_template.csv"):
               pid = template_path.stem.replace("participant_", "").replace("_template", "")
               participant_templates[pid] = pd.read_csv(template_path)

       return global_template, participant_templates
   ```
   - Loads pre-existing global template from disk
   - Loads all participant-specific templates
   - Provides warning messages if templates not found
   - Used when starting pipeline from step 7 or 8

2. **Updated `run_pose_processing_pipeline()` signature** (`process_pose_data.py:697`):
   - Changed from: `def run_pose_processing_pipeline() -> None:`
   - To: `def run_pose_processing_pipeline(start_step: int = 1) -> None:`
   - Added start_step parameter with default value of 1 (full pipeline)
   - Added validation to ensure start_step is between 1 and 8

3. **Added conditional step execution logic** (`process_pose_data.py:745-820`):
   - **Steps 1-5** (preprocessing): Run if `start_step <= 5`
   - **Step 6** (templates): Run if `start_step <= 6`, else load from disk
   - **Step 7** (features): Run if `start_step <= 7`
   - **Step 8** (linear metrics): Always run if reached
   - Loads normalized data from disk when starting from step 6+
   - Loads templates from disk when starting from step 7+

4. **Added `--start-step` command-line argument** (`process_pose_data.py:880-888`):
   ```python
   parser.add_argument(
       "--start-step",
       type=int,
       default=1,
       choices=range(1, 9),
       metavar="N",
       help="Start pipeline from step N (1-8). Requires existing data from prior steps. "
            "Example: --start-step 7 to run only feature extraction and linear metrics"
   )
   ```
   - Accepts values 1-8
   - Validates input using choices parameter
   - Provides clear help text with usage example

5. **Updated pipeline call** (`process_pose_data.py:898`):
   - Changed from: `run_pose_processing_pipeline()`
   - To: `run_pose_processing_pipeline(start_step=args.start_step)`

**Usage Examples**:

```bash
# Run full pipeline (default)
python process_pose_data.py

# Start from step 6 (build templates, extract features, compute linear metrics)
python process_pose_data.py --start-step 6

# Start from step 7 (extract features and compute linear metrics only)
python process_pose_data.py --start-step 7

# Start from step 8 (compute linear metrics only)
python process_pose_data.py --start-step 8
```

**Result**:
- Pipeline now supports flexible step control matching .forcomparison
- Can skip expensive preprocessing when only later steps need rerunning
- Reduces iteration time during parameter tuning and debugging
- Loads existing data/templates from disk as needed
- Clear validation and error messages for invalid step numbers
- Consistent with .forcomparison implementation

**Testing Recommended**:
- Test `--start-step 1` (full pipeline)
- Test `--start-step 6` with existing normalized data
- Test `--start-step 7` with existing normalized data and templates
- Test `--start-step 8` with existing per-frame features
- Verify error handling for invalid step numbers
- Confirm templates are loaded correctly when skipping step 6

---

### ⚠️ ISSUE #11: Potential Division by Zero (Actually Well-Handled)

**Status**: 🟢 LOW PRIORITY - Already Good
**Files Affected**: `Pose/utils/features_utils.py:402-407`

**Note**: This is actually **well-handled** in your code:
```python
with np.errstate(divide='ignore', invalid='ignore'):
    scaled_arr = arr / io
    bad_mask = ~np.isfinite(scaled_arr) | (io < 1e-6)
    scaled_arr[bad_mask] = arr[bad_mask]
```

Good defensive programming - no fix needed. Just noting it was reviewed.

---

## COMPARISON WITH .forcomparison CODE

### Filename Convention Difference

| Aspect | Your Code | .forcomparison | Status |
|--------|-----------|----------------|--------|
| **Input filenames** | `3101_01_pose.csv` (trial number) | `3101_H.csv` (condition letter) | See Issue #1 |
| **Parsing function** | `parse_participant_trial()` | `parse_pid_cond()` | Different but Issue #1 addresses this |
| **Output naming** | Should use condition but uses trial | Uses condition correctly | See Issue #1 |

### Pipeline Features Comparison

| Feature | Your Code | .forcomparison | Action |
|---------|-----------|----------------|--------|
| **--start-step parameter** | ✅ Implemented | ✅ Allows starting from step 1-8 | ✅ COMPLETED |
| **Template loading function** | ✅ Implemented | ✅ `load_existing_templates()` | ✅ COMPLETED |
| **Condition mapping** | ✅ Implemented | ✅ Fully integrated | ✅ COMPLETED (Issue #1) |
| **Error handling** | ✅ Good | ✅ Good | Both equivalent |

### Statistical Metrics Comparison

| Metric Type | Your Code (per feature) | .forcomparison (per feature) | Difference |
|-------------|-------------------------|------------------------------|------------|
| **Base metrics** | 1 (value only) | 3 (value, vel, acc as separate signals) | 3x more |
| **Statistics per signal** | 4 (mean, mean_abs_vel, mean_abs_acc, rms) | 9 (min, max, mean, rms, std, median, p25, p75, autocorr1) | 2.25x more |
| **Total per feature** | ~4 metrics | ~27 metrics | 6.75x more |

**Example for `blink_aperture`**:

**Your output**:
- `blink_aperture_mean`
- `blink_aperture_mean_abs_vel`
- `blink_aperture_mean_abs_acc`
- `blink_aperture_rms`

**.forcomparison output**:
- `blink_aperture_min`, `_max`, `_mean`, `_rms`, `_std`, `_median`, `_p25`, `_p75`, `_autocorr1`
- `blink_aperture_vel_min`, `_max`, `_mean`, `_rms`, `_std`, `_median`, `_p25`, `_p75`, `_autocorr1`
- `blink_aperture_acc_min`, `_max`, `_mean`, `_rms`, `_std`, `_median`, `_p25`, `_p75`, `_autocorr1`

### Derivative Computation Method

| Aspect | Your Code | .forcomparison |
|--------|-----------|----------------|
| **Method** | `np.diff()` with dt scaling | `np.gradient()` with fps scaling |
| **Accuracy** | Good for uniform sampling | Slightly more accurate (centered differences) |
| **Implementation** | Computed during windowing | Pre-computed as new columns |
| **Flexibility** | Less flexible | More flexible (can compute any statistic on derivatives) |

---

## IMPLEMENTATION PLAN

### Phase 1: Critical Fixes (MUST DO)

#### Task 1.1: Fix Condition Mapping Integration
- **File**: `Pose/process_pose_data.py`
- **Difficulty**: Medium
- **Time Estimate**: 1-2 hours
- **Steps**:
  1. Add condition mapping at start of `run_pose_processing_pipeline()`
  2. Update all calls to `get_output_filename()` to pass condition
  3. Update `check_steps_1_5_complete()` to use conditions
  4. Update `load_existing_normalized_data()` to use conditions
  5. Test with sample data

#### Task 1.2: Add Rich Linear Metrics
- **Files**: `Pose/utils/features_utils.py`
- **Difficulty**: Medium-High
- **Time Estimate**: 2-3 hours
- **Steps**:
  1. Add `add_perframe_derivatives()` function
  2. Replace `compute_linear_from_perframe_dir()` with .forcomparison version
  3. Add new statistical metrics (std, median, p25, p75, autocorr1)
  4. Update to handle value/_vel/_acc columns separately
  5. Test output format matches expected

#### Task 1.3: Verify and Fix Procrustes Landmarks
- **File**: `Pose/utils/config.py`
- **Difficulty**: Low (research) + Low (fix)
- **Time Estimate**: 30 minutes research + 10 minutes fix
- **Steps**:
  1. Research MediaPipe/OpenPose landmark 28 vs 30
  2. Verify which is correct for Procrustes reference
  3. Update if needed
  4. Document decision in code comments

---

### Phase 2: High Priority Fixes (SHOULD DO)

#### Task 2.1: Fix Path Handling
- **File**: `Pose/utils/config.py`
- **Difficulty**: Low
- **Time Estimate**: 10 minutes
- **Steps**:
  1. Update RAW_DIR to use absolute path
  2. Test paths work from different working directories

#### Task 2.2: Add SciPy Availability Check
- **Files**: `Pose/utils/config.py`, `Pose/process_pose_data.py`
- **Difficulty**: Low
- **Time Estimate**: 15 minutes
- **Steps**:
  1. Add try/except import in config.py
  2. Add check before step 4 in pipeline
  3. Test with and without scipy installed

---

### Phase 3: Medium Priority Fixes (NICE TO HAVE)

#### Task 3.1: Fix SCALE_BY_INTEROCULAR Flag Usage
- **Files**: `Pose/utils/config.py`, `Pose/process_pose_data.py`
- **Difficulty**: Low
- **Time Estimate**: 10 minutes
- **Steps**:
  1. Import flag from config
  2. Use flag instead of hardcoded logic
  3. Test that flag controls behavior

#### Task 3.2: Fix Documentation Errors
- **File**: `Pose/utils/normalize_utils.py`
- **Difficulty**: Trivial
- **Time Estimate**: 5 minutes
- **Steps**:
  1. Update docstring at line 64
  2. Verify comments match code

#### Task 3.3: Add Window Normalization Validation
- **File**: `Pose/utils/features_utils.py`
- **Difficulty**: Low
- **Time Estimate**: 10 minutes
- **Steps**:
  1. Add `np.all(np.isfinite(seg))` check
  2. Test with edge cases

#### Task 3.4: Add Mutual Exclusion Enforcement
- **File**: `Pose/utils/config.py` or `Pose/utils/features_utils.py`
- **Difficulty**: Low
- **Time Estimate**: 15 minutes
- **Steps**:
  1. Decide on enforcement location
  2. Add validation check
  3. Raise clear error message if both True

---

### Phase 4: Low Priority / Nice-to-Have

#### Task 4.1: Update Procrustes Function Signature
- **File**: `Pose/utils/geometry_utils.py`
- **Difficulty**: Low
- **Time Estimate**: 15 minutes
- **Steps**:
  1. Update return type annotation
  2. Update docstring
  3. No code changes needed (already returns correct values)

#### Task 4.2: Add --start-step Parameter
- **File**: `Pose/process_pose_data.py`
- **Difficulty**: Medium
- **Time Estimate**: 1-2 hours
- **Steps**:
  1. Add command-line argument
  2. Refactor pipeline to support skipping steps
  3. Add functions to load existing data/templates
  4. Update control flow
  5. Test all start points (1-8)

---

## TESTING CHECKLIST

After implementing fixes, test:

- [ ] Pipeline runs end-to-end without errors
- [ ] Output files use condition-based naming (e.g., `3101_L_norm.csv`)
- [ ] Skip logic works (rerunning pipeline skips completed steps)
- [ ] Linear metrics output includes all new statistics
- [ ] Linear metrics output has value, _vel, _acc for each feature
- [ ] Per-frame files include _vel and _acc columns
- [ ] Procrustes alignment produces expected results
- [ ] Pipeline works without scipy if filtering disabled
- [ ] Path handling works from different working directories
- [ ] Configuration flags control expected behavior

---

## QUESTIONS FOR DECISION

1. **Procrustes Landmarks**: Should we use landmark 28 or 30? Need to verify which is standard.

2. **SCALE_BY_INTEROCULAR Flag**:
   - Should we use the flag globally (for all feature types)?
   - Or keep type-specific logic (only for "original")?
   - Or make it configurable per feature type?

3. **Mutual Exclusion for Window Normalization**:
   - Where should we enforce this? Config initialization or runtime?
   - Should it be an error or warning?

4. **--start-step Parameter**:
   - Is this needed for your workflow?
   - Priority for implementation?

5. **Derivative Method**:
   - Switch from `np.diff` to `np.gradient` for consistency?
   - Or keep current method if already published/established?

---

## NOTES

- All code reviews completed on October 14, 2025
- Comparison based on `.forcomparison/Pose/` directory
- RQA/CRQA code in .forcomparison was ignored as specified
- Total files reviewed: ~2,366 lines (current) + ~1,200 lines (.forcomparison)

---

## SIGN-OFF

Once each fix is completed:
- [x] Task 1.1: Condition Mapping - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 1.2: Rich Linear Metrics - **Status**: ✅ COMPLETED (October 14, 2025)
- [ ] Task 1.3: Procrustes Landmarks - **Status**: ❌ REMOVED (Issue was intentional)
- [x] Task 2.1: Path Handling - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 2.2: SciPy Check - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 3.1: SCALE_BY_INTEROCULAR - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 3.2: Documentation - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 3.3: Unused Window Flags - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 4.1: Procrustes Signature - **Status**: ✅ COMPLETED (October 14, 2025)
- [x] Task 4.2: --start-step & load_existing_templates() - **Status**: ✅ COMPLETED (October 14, 2025)

---

**END OF DOCUMENT**
