# Facial Pose Analysis: Complete Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Data Input and Format](#data-input-and-format)
3. [Processing Pipeline Architecture](#processing-pipeline-architecture)
4. [Module-by-Module Analysis](#module-by-module-analysis)
5. [Feature Extraction Details](#feature-extraction-details)
6. [Quality Control System](#quality-control-system)
7. [Coordinate Normalization Methods](#coordinate-normalization-methods)
8. [Temporal Filtering](#temporal-filtering)
9. [Statistical Analysis Capabilities](#statistical-analysis-capabilities)
10. [Output Formats and Usage](#output-formats-and-usage)
---

## Data Input and Format

### Expected Input Format

The pipeline expects CSV files with OpenPose facial landmark data:

```
x0,y0,prob0,x1,y1,prob1,...,x69,y69,prob69
123.45,234.56,0.89,125.67,235.78,0.92,...
```

**Landmark Indexing (68 landmarks + 2 pupils = 70 total)**:
- Points 0-16: Jaw line
- Points 17-26: Eyebrows (left: 17-21, right: 22-26)
- Points 27-35: Nose bridge and tip
- Points 36-41: Left eye contour
- Points 42-47: Right eye contour
- Points 48-67: Mouth and lips
- Points 68-69: Pupil centers (if available)

---

## Processing Pipeline Architecture

### Pipeline Flow

```
Raw OpenPose CSV Files
          �
    Quality Control Analysis
          �
    Coordinate Normalization
          �
     Feature Extraction
          �
    Quality-Based Masking
          �
     Temporal Filtering
          �
    Statistical Analysis
          �
    Research-Ready Outputs
```

### Pipeline Stages

1. **Quality Control (QC)**: Identifies unreliable data periods
2. **Coordinate Normalization**: Removes head pose, position, and scale effects
3. **Feature Extraction**: Computes meaningful facial behavior measures
4. **Masking**: Applies QC results to remove unreliable data
5. **Filtering**: Reduces temporal noise while preserving signal
6. **Analysis**: Generates statistics and visualizations

### Configuration System

```python
# Quality Control Parameters
QC_WINDOW_SIZE = 1800          # 30 seconds at 60fps
CONFIDENCE_THRESHOLD = 0.3     # Minimum landmark confidence
MAX_INTERPOLATION = 60         # Maximum gap to interpolate

# Processing Parameters
COORDINATE_SYSTEM = "procrustes"  # Normalization method
SAMPLING_RATE = 60.0           # Data sampling rate
CUTOFF_FREQUENCY = 10.0        # Temporal filter cutoff
```

---

## Module-by-Module Analysis

### 1. `landmark_config.py` - Configuration Management

**Purpose**: Centralizes all landmark definitions and feature mappings for consistency across modules.

**Key Components**:
- **Landmark Indices**: Defines facial regions (eyes, mouth, nose, etc.)
- **Feature Mappings**: Links QC metrics to output feature columns
- **Helper Functions**: Column name generation and validation

**Critical Definitions**:
```python
EYES = {
    "L": [37, 38, 39, 40, 41, 42],  # Left eye landmarks
    "R": [43, 44, 45, 46, 47, 48],  # Right eye landmarks
}

METRIC_KEYPOINTS = {
    "eyes": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    "mouth_dist": [62, 66],  # Top and bottom lip centers
    "center_face": list(range(28, 36)),  # Nose bridge region
}
```

**Usage**: Referenced by all other modules to ensure consistent landmark indexing.

### 2. `quality_control.py` - Data Quality Assessment

**Purpose**: Automatically identifies time periods where landmark tracking is unreliable due to poor lighting, occlusion, or tracking failures.

**Algorithm**:
1. **Sliding Window Analysis**: Divides data into overlapping time windows
2. **Gap Detection**: Identifies consecutive missing/low-confidence frames
3. **Interpolation Assessment**: Determines if gaps are too long to reliably interpolate
4. **Regional Grouping**: Organizes results by facial region

**Key Functions**:
- `calculate_window_ranges()`: Generates sliding window positions
- `check_landmark_missing()`: Identifies poor-quality landmarks
- `analyze_file_quality()`: Complete QC analysis for single file
- `run_quality_control_batch()`: Batch processing for entire datasets

**Quality Metrics**:
- **Confidence Threshold**: Minimum acceptable landmark detection confidence
- **Gap Length**: Maximum consecutive missing frames before marking as "bad"
- **Regional Assessment**: If any landmark in a region is bad, entire region is flagged

**Output Files**:
- `keypoint_bad_windows.csv`: Per-landmark quality statistics
- `metric_bad_windows.csv`: Per-region quality summaries
- `metric_bad_window_indices.csv`: Specific time ranges of bad windows

### 3. `coordinate_normalization.py` - Pose Standardization

**Purpose**: Removes effects of head position, rotation, and scale to enable comparison across participants and time.

#### Original Method (Eye-Corner Based)
**Algorithm**:
1. Use outer eye corners (landmarks 37, 46) as reference points
2. Center coordinates at midpoint between eye corners
3. Rotate to make eye line horizontal
4. Scale by inter-ocular distance

**Mathematical Transform**:
```
# Center translation
x_centered = x_original - eye_midpoint_x
y_centered = y_original - eye_midpoint_y

# Rotation to horizontal
x_rotated = cos(�) * x_centered + sin(�) * y_centered
y_rotated = -sin(�) * x_centered + cos(�) * y_centered

# Scale normalization
x_normalized = x_rotated / inter_ocular_distance
y_normalized = y_rotated / inter_ocular_distance
```

#### Procrustes Method (Recommended)
**Algorithm**:
1. Select stable reference landmarks (temples, nose points)
2. Calculate average shape across all valid frames
3. For each frame, find optimal similarity transform to align with average
4. Apply transformation: translation + rotation + scaling

**Advantages**:
- More robust to individual landmark failures
- Uses information from multiple landmarks
- Better handles extreme head poses
- Statistically principled approach

**Implementation Details**:
- Uses SVD for optimal rotation calculation
- Handles degenerate cases (insufficient landmarks)
- Preserves original coordinates as backup
- Adds "_proc" suffix to aligned coordinates

### 4. `feature_extraction.py` - Behavioral Feature Computation

**Purpose**: Converts normalized landmark coordinates into meaningful behavioral and physiological measures.

#### Core Features

**Eye Aspect Ratio (EAR) - Estimate of Blink Behaviour**:
```python
# For each eye: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
left_ear = (dist(38,42) + dist(39,41)) / (2 * dist(37,40))
right_ear = (dist(44,48) + dist(45,47)) / (2 * dist(43,46))
blink_dist = (left_ear + right_ear) / 2
```

**Mouth Opening Distance**:
```python
mouth_dist = euclidean_distance(lip_top_center, lip_bottom_center)
```

**Head Rotation Angle**:
```python
head_rotation = arctan2(y_right_eye - y_left_eye, x_right_eye - x_left_eye)
```

#### Regional Features

**Face Center Movement**:
- Combines nose bridge landmarks (28-36)
- Provides overall head movement measure
- Frame-to-frame magnitude calculation

**Pupil/Gaze Features**:
- Individual pupil positions (if available)
- Combined average pupil position
- Gaze deviation magnitude from center

**Movement Magnitudes**:
For all coordinate features, calculates frame-to-frame movement:
```python
magnitude = sqrt((x[t] - x[t-1])� + (y[t] - y[t-1])�)
```

### 5. `temporal_filtering.py` - Signal Processing

**Purpose**: Reduces high-frequency noise while preserving meaningful signal characteristics.

#### Butterworth Low-Pass Filter (Primary Method)

**Design Characteristics**:
- Zero-phase filtering (no temporal shift)
- Configurable cutoff frequency and filter order
- Preserves NaN patterns exactly
- Handles edge effects gracefully

**Algorithm**:
1. Fill NaN gaps temporarily for filter stability
2. Apply second-order sections (SOS) Butterworth filter
3. Use `sosfiltfilt` for zero-phase response
4. Restore original NaN locations

**Parameter Selection**:
- **Cutoff Frequency**: Typically 5-15 Hz for facial movements
- **Filter Order**: Usually 4 (steeper rolloff with higher orders)
- **Sampling Rate**: Match original data collection rate

#### Alternative Filters

**Moving Average**:
- Simple temporal smoothing
- Configurable window size
- Good for initial data exploration

**Median Filter**:
- Effective for spike noise removal
- Preserves edges better than mean filters
- Useful for preprocessing before main filtering

**Adaptive Filter**:
- Adjusts filtering strength based on local signal characteristics
- Stronger filtering in noisy regions
- Lighter filtering in stable regions

### 6. `masking.py` - Quality-Based Data Exclusion

**Purpose**: Applies quality control results to remove unreliable data periods while preserving data structure.

**Masking Strategy**:
1. Load bad window locations from QC analysis
2. Convert window indices to frame ranges
3. Set corresponding feature values to NaN
4. Different facial regions can have different mask patterns

**Key Functions**:
- `load_bad_windows()`: Converts QC results to mask format
- `apply_bad_window_masks()`: Applies masks to feature data
- `interpolate_masked_regions()`: Optional gap filling after masking

**Interpolation Options**:
- Linear: Simple and fast
- Polynomial: Smoother curves
- Spline: Highest quality but computationally expensive

### 7. `statistics.py` - Statistical Analysis Tools

**Purpose**: Provides comprehensive statistical analysis and normalization capabilities for research applications.

#### Normalization Methods

**Z-Score Normalization**:
```python
z = (x - mean) / standard_deviation
```
- Centers data at zero
- Unit variance scaling
- Option for participant-specific normalization

**Percentile Normalization**:
```python
normalized = (x - percentile_min) / (percentile_max - percentile_min)
```
- Robust to outliers
- Configurable percentile range (e.g., 5th-95th percentile)
- Results in [0,1] range

**Robust Normalization**:
```python
robust_z = (x - median) / median_absolute_deviation
```
- Uses median and MAD instead of mean and SD
- Less sensitive to outliers
- Preferred for non-normal distributions

#### Statistical Measures

**Descriptive Statistics**:
- Mean, median, standard deviation
- Quartiles and percentiles
- Skewness and kurtosis
- Confidence intervals

**Correlation Analysis**:
- Pearson, Spearman, Kendall correlations
- Correlation matrix generation
- High correlation detection for multicollinearity

**Time Series Features**:
- Rolling statistics (mean, SD, range)
- Temporal autocorrelation
- Change point detection
- Trend analysis

### 8. `plotting.py` - Research Visualization

**Purpose**: Generates publication-quality visualizations for data exploration, quality assessment, and results presentation.

#### Quality Control Visualizations

**QC Summary Plots**:
- Bar charts showing bad window percentages by facial region
- Identifies systematic data quality issues
- Helps optimize QC parameters

**Timeline Visualizations**:
- Shows bad window locations over time
- Different colors for different facial regions
- Useful for understanding temporal patterns in data quality

#### Feature Visualizations

**Time Series Plots**:
- Multiple features on aligned time axes
- Highlights masked regions
- Configurable time ranges and feature selection

**Distribution Plots**:
- Histograms and density plots
- Grouped by experimental conditions
- Statistical overlays (mean, percentiles)

**Correlation Heatmaps**:
- Visualizes correlation matrices
- Color-coded correlation strength
- Hierarchical clustering for feature grouping

#### Comparison Plots

**Before/After Filtering**:
- Shows effect of temporal filtering
- Overlay of original and filtered signals
- Quantifies noise reduction

**Condition Comparisons**:
- Box plots comparing experimental conditions
- Statistical significance indicators
- Effect size visualizations

### 9. `pipeline.py` - Workflow Orchestration

**Purpose**: Coordinates all processing steps in a reproducible, configurable workflow.

**Complete Pipeline Function**:
```python
run_complete_pose_pipeline(
    raw_input_dir,           # Input data location
    output_base_dir,         # Output directory
    window_size=1800,        # QC window size
    coordinate_system="procrustes",  # Normalization method
    apply_temporal_filter=True,      # Enable filtering
    sampling_rate=60.0,      # Data sampling rate
    cutoff_frequency=10.0    # Filter cutoff
)
```

**Pipeline Outputs**:
- Processed feature files (CSV)
- Quality control reports
- Processing logs and metadata
- Statistical summaries
- Visualization figures

**Error Handling**:
- Graceful handling of corrupted files
- Detailed error logging
- Partial processing continuation
- Recovery recommendations

---

## Enhanced Flexible Feature Extraction System

### Overview of Enhanced Capabilities

The pipeline now includes advanced flexible feature extraction options that provide fine-grained control over processing methods. These enhancements address the need for different coordinate normalization approaches for different types of features within the same analysis.

#### Key Enhancements

1. **Per-Feature Procrustes Control**: Choose which features use Procrustes alignment independently
2. **Nose-Relative Gaze Features**: Calculate pupil positions relative to nose center for head-movement independence
3. **Flexible Feature Selection**: Extract only specific features needed for targeted analysis
4. **Research-Specific Configurations**: Pre-defined settings optimized for different research applications

### Per-Feature Procrustes Configuration

#### Configuration System

The enhanced system allows you to specify which features should use Procrustes-aligned coordinates and which should use original coordinates:

```python
feature_config = {
    'blink_dist': True,          # Use Procrustes for stable blink detection
    'mouth_dist': False,         # Use original for natural speech movements
    'pupils': True,              # Use Procrustes for accurate gaze tracking
    'head_rotation_angle': False # Use raw coordinates for head pose
}

features = extract_all_features(
    df,
    use_procrustes=False,  # Default setting
    feature_procrustes_config=feature_config,
    pupil_relative_to_nose=True
)
```

#### Scientific Rationale

**When to Use Procrustes for Specific Features**:
- **Blink Detection**: Procrustes alignment provides more stable eye measurements by removing head movement artifacts
- **Pupil/Gaze Tracking**: Alignment improves accuracy of relative gaze position calculations
- **Facial Expression Analysis**: Normalized coordinates better capture pure expression changes

### Nose-Relative Gaze Features

The nose-relative approach separates these components by calculating gaze position relative to a stable facial reference point.

#### Implementation

```python
# Calculate nose center from bridge landmarks (28-36)
nose_center_x = np.mean([df[f'x{i}'], df[f'x{i+1}'], ...])
nose_center_y = np.mean([df[f'y{i}'], df[f'y{i+1}'], ...])

# Pupil positions relative to nose
left_pupil_rel_nose_x = left_pupil_x - nose_center_x
left_pupil_rel_nose_y = left_pupil_y - nose_center_y

# Average gaze relative to nose
avg_pupil_rel_nose_magnitude = sqrt(
    (avg_pupil_rel_nose_x)² + (avg_pupil_rel_nose_y)²
)
```

#### Generated Features

- `left_pupil_rel_nose_x/y`: Left pupil relative to nose center
- `right_pupil_rel_nose_x/y`: Right pupil relative to nose center
- `avg_pupil_rel_nose_x/y`: Average gaze position relative to nose
- `avg_pupil_rel_nose_magnitude`: Gaze deviation independent of head movement
- `nose_center_x/y`: Computed nose reference position


### Flexible Feature Selection

#### Targeted Extraction

Extract only specific features needed for focused analyses:

```python
# Extract only gaze and blink features for attention study
gaze_features = extract_features_flexible(
    df,
    feature_list=['pupils', 'blink_dist'],
    procrustes_config={'pupils': True, 'blink_dist': True},
    pupil_relative_to_nose=True
)
```

### Pipeline Integration

#### Enhanced Pipeline Function

The main pipeline now supports flexible feature extraction:

```python
output_paths = run_complete_pose_pipeline(
    raw_input_dir="data/raw_pose",
    output_base_dir="data/processed",
    coordinate_system="procrustes",
    feature_procrustes_config={
        'blink_dist': True,
        'mouth_dist': False,
        'pupils': True
    },
    pupil_relative_to_nose=True
)
```

#### Backward Compatibility

All existing code continues to work without modification. New parameters are optional and default to previous behavior when not specified.

### Validation and Quality Assurance

#### Feature Validation

The system automatically validates feature extraction settings:
- Checks for conflicting configuration parameters
- Verifies required landmarks are available for requested features
- Warns about potential issues with feature combinations

#### Quality Metrics

Enhanced quality assessment for new features:
- Nose center stability validation
- Relative feature range checking
- Cross-validation of Procrustes vs. original results

### Performance Considerations

#### Computational Impact

- Per-feature Procrustes: Minimal additional computation (~5% overhead)
- Nose-relative calculations: Negligible impact (<1% overhead)
- Flexible extraction: Potential speedup when extracting fewer features

#### Memory Usage

- Configuration overhead: <1MB additional memory
- Feature-specific processing: Memory scales with number of requested features
- Overall impact: Minimal for typical use cases

---

## Feature Extraction Details

### Facial Expression Features

#### Eye Aspect Ratio (EAR)
**Scientific Basis**: Based on the geometric relationship between eye opening and facial landmark positions (Soukupov� & ech, 2016).

**Calculation**:
1. Identify 6 eye landmarks per eye (inner/outer corners, top/bottom points)
2. Calculate vertical distances: top-to-bottom separations
3. Calculate horizontal distance: corner-to-corner separation
4. Ratio: (vertical1 + vertical2) / (2 � horizontal)

**Interpretation**:
- Normal open eyes: EAR H 0.2-0.4
- Closed eyes: EAR H 0.0-0.1
- Blink detection: Sudden drops below threshold
- Fatigue: Gradual decrease in baseline EAR

**Research Applications**:
- Microsleep detection
- Attention monitoring
- Cognitive load assessment
- Driver fatigue analysis

#### Mouth Opening Distance
**Scientific Basis**: Mouth movements reflect speech, emotional expression, and stress responses.

**Calculation**:
1. Identify top lip center (landmark 62 or 63)
2. Identify bottom lip center (landmark 66 or 67)
3. Calculate Euclidean distance
4. Normalize by head size (if using original method)

**Interpretation**:
- Closed mouth: Distance H 0
- Speech: Variable, rapid changes
- Emotional expressions: Sustained changes
- Stress: Potential changes in baseline tension

**Research Applications**:
- Speech activity detection
- Emotional state classification
- Stress level assessment
- Social interaction analysis

#### Head Rotation Angle
**Scientific Basis**: Head pose reflects attention direction, engagement level, and postural control.

**Calculation**:
1. Identify left and right eye corners (landmarks 37, 46)
2. Calculate angle of line connecting eye corners
3. Reference to horizontal axis
4. Range: -� to � radians

**Interpretation**:
- 0 radians: Perfectly upright head
- Positive angles: Head tilted to participant's right
- Negative angles: Head tilted to participant's left
- Large angles: Potential fatigue or attention shifts

**Research Applications**:
- Attention direction tracking
- Engagement measurement
- Fatigue detection
- Postural stability assessment

### Movement Features

#### Regional Movement Magnitudes
**Scientific Basis**: Frame-to-frame movement quantifies facial dynamics and motor control.

**Calculation**:
For each facial region (eyes, mouth, face center):
1. Calculate position at frame t
2. Calculate position at frame t-1
3. Euclidean distance between positions
4. Magnitude = [(x_t - x_{t-1})� + (y_t - y_{t-1})�]

**Interpretation**:
- Low magnitude: Stable, controlled movement
- High magnitude: Active movement or tracking instability
- Patterns: Rhythmic vs. irregular movement
- Trends: Increasing magnitude may indicate fatigue

#### Gaze Deviation Features
**Scientific Basis**: Eye movements reflect attention allocation and cognitive processing.

**Calculation** (when pupil data available):
1. Calculate average pupil position (left + right) / 2
2. Measure deviation from face center
3. Calculate magnitude of deviation vector
4. Track changes over time

**Interpretation**:
- Central gaze: Low deviation magnitude
- Peripheral gaze: High deviation magnitude
- Attention shifts: Rapid changes in gaze position
- Sustained attention: Stable gaze patterns

#### Enhanced Nose-Relative Gaze Features (NEW)
**Scientific Basis**: By calculating pupil positions relative to nose center, these features isolate actual eye movement from head movement, providing more accurate measures of attention and gaze direction.

**Calculation**:
1. Calculate nose center from nose bridge landmarks (28-36)
2. Compute pupil positions relative to nose center:
   - `left_pupil_rel_nose_x = left_pupil_x - nose_center_x`
   - `left_pupil_rel_nose_y = left_pupil_y - nose_center_y`
3. Calculate average relative gaze position
4. Compute magnitude of relative gaze deviation

**Generated Features**:
- `left_pupil_rel_nose_x/y`: Left pupil relative to nose
- `right_pupil_rel_nose_x/y`: Right pupil relative to nose
- `avg_pupil_rel_nose_x/y`: Average gaze relative to nose
- `avg_pupil_rel_nose_magnitude`: Gaze deviation independent of head movement
- `nose_center_x/y`: Computed nose reference position

**Advantages**:
- **Head Movement Independence**: Separates eye movement from head movement
- **Better Attention Tracking**: More accurate measure of gaze direction
- **Workload Assessment**: Can distinguish attention shifts from postural changes
- **Cross-Participant Comparison**: Reduces individual differences in head pose

**Research Applications**:
- Attention and vigilance studies
- Workload classification
- Fatigue detection
- Human-computer interaction
- Driver monitoring systems

---

## Quality Control System

### Sliding Window Analysis

#### Window Configuration
**Window Size Selection**:
- Small windows (5-10 seconds): Detect brief tracking failures
- Medium windows (15-30 seconds): Balance sensitivity and stability
- Large windows (45-60 seconds): Focus on major quality issues

**Overlap Strategy**:
- No overlap (0%): Faster processing, risk of edge effects
- 50% overlap: Better temporal resolution, increased computation
- 75% overlap: Maximum sensitivity, highest computational cost

#### Quality Assessment Metrics

**Confidence Score Analysis**:
```python
# Landmark is considered "good" if:
confidence >= threshold AND
x_coordinate is not NaN AND
y_coordinate is not NaN
```

**Gap Length Analysis**:
- Consecutive missing frames counted
- Compared to maximum interpolation limit
- Window marked as "bad" if gap exceeds limit

**Regional Assessment**:
- Eyes: All eye landmarks must be good
- Mouth: Lip landmarks must be good
- Face center: Nose bridge landmarks must be good
- Head rotation: Eye corner landmarks must be good

### Quality Control Outputs

#### Keypoint Summary (`keypoint_bad_windows.csv`)
```
file,keypoint,bad_windows,total_windows,pct_bad
participant_001.csv,37,5,120,4.17
participant_001.csv,38,3,120,2.50
```

**Interpretation**:
- High percentages (>10%) indicate systematic tracking problems
- Specific landmarks consistently failing suggest environmental issues
- Temporal patterns may indicate lighting or pose changes

#### Metric Summary (`metric_bad_windows.csv`)
```
file,metric,bad_windows,total_windows,pct_bad
participant_001.csv,eyes,8,120,6.67
participant_001.csv,mouth_dist,2,120,1.67
```

**Usage**:
- Guides exclusion decisions for analysis
- Identifies problematic files or participants
- Informs data collection improvements

#### Bad Window Indices (`metric_bad_window_indices.csv`)
```
file,metric,window_index,start_frame,end_frame_exclusive
participant_001.csv,eyes,5,9000,10800
participant_001.csv,eyes,12,21600,23400
```

**Application**:
- Precise temporal masking during analysis
- Visualization of quality patterns
- Manual review of problematic periods

---

## Coordinate Normalization Methods

### Method Comparison

| Aspect | Original (Eye-Corner) | Procrustes |
|--------|----------------------|------------|
| Reference Points | 2 (eye corners) | 4+ (stable landmarks) |
| Robustness | Fails if eye corners missing | Handles individual landmark failures |
| Computational Cost | Low | Medium |
| Accuracy | Good for frontal poses | Better for varied poses |
| Research Use | Legacy compatibility | Recommended for new studies |

### When to Use Each Method

#### Original Method
**Advantages**:
- Simple, interpretable
- Fast computation
- Compatible with existing research
- Good for controlled environments

**Use Cases**:
- Replication of previous studies
- Real-time applications
- Limited computational resources
- Primarily frontal face recordings

#### Procrustes Method
**Advantages**:
- Statistically principled
- Robust to individual landmark failures
- Better handles pose variation
- Uses information from multiple landmarks

**Use Cases**:
- New research studies
- Varied head pose conditions
- Higher quality requirements
- Publications requiring methodological rigor

### Normalization Validation

#### Quality Checks
**Visual Inspection**:
- Overlay normalized landmarks on original images
- Verify alignment across different poses
- Check for systematic distortions

**Quantitative Validation**:
- Measure inter-subject variability after normalization
- Compare feature stability across methods
- Assess robustness to missing landmarks

**Statistical Validation**:
- Cross-validation with held-out data
- Comparison of analysis results between methods
- Effect size calculations for method differences

---

## Temporal Filtering

### Filter Selection Criteria

#### Signal Characteristics
**Facial Movement Frequencies**:
- Voluntary movements: 0.1-5 Hz
- Blinks: 2-8 Hz
- Micro-expressions: 1-15 Hz
- Tracking noise: >20 Hz

**Filter Requirements**:
- Preserve meaningful facial dynamics
- Remove tracking noise and artifacts
- Maintain temporal relationships
- Avoid introducing phase shifts

### Butterworth Filter Implementation

#### Mathematical Foundation
**Transfer Function** (nth-order low-pass):
```
H(s) = ə / (s^n + a���s^(n-1) + ... + a�s + ə)
```

Where ə is the cutoff frequency in rad/s.

**Digital Implementation**:
1. Design analog prototype
2. Bilinear transform to digital domain
3. Convert to second-order sections (SOS)
4. Apply forward and backward filtering

#### Parameter Selection Guidelines

**Cutoff Frequency**:
- Conservative (5 Hz): Preserves only slow movements
- Moderate (10 Hz): Good balance for most applications
- Liberal (15 Hz): Preserves more detail, less noise reduction

**Filter Order**:
- 2nd order: Gentle rolloff, minimal artifacts
- 4th order: Good balance (recommended)
- 6th order: Sharp rolloff, potential ringing artifacts

**Sampling Rate Considerations**:
- Nyquist frequency = sampling_rate / 2
- Cutoff must be < Nyquist frequency
- Higher sampling rates allow higher cutoffs

### Filter Validation

#### Signal Quality Assessment
**Before/After Comparison**:
- Signal-to-noise ratio improvement
- Preservation of meaningful features
- Removal of high-frequency artifacts

**Frequency Domain Analysis**:
- Power spectral density plots
- Verification of cutoff effectiveness
- Check for filter artifacts

**Time Domain Validation**:
- Preservation of event timing
- No phase shifts introduced
- Smooth transitions maintained

---

## Statistical Analysis Capabilities

### Normalization Strategies

#### Within-Subject Normalization
**Purpose**: Account for individual differences in baseline facial characteristics.

**Z-Score (Participant-Specific)**:
```python
z_score = (value - participant_mean) / participant_std
```

**Advantages**:
- Removes individual baseline differences
- Enables cross-participant comparison
- Focuses on within-subject changes

#### Across-Subject Normalization
**Purpose**: Create standardized measures across the entire sample.

**Global Z-Score**:
```python
z_score = (value - global_mean) / global_std
```

**Percentile Ranking**:
```python
percentile = rank(value) / total_count * 100
```

### Advanced Statistical Features

#### Time Series Analysis
**Temporal Features**:
- Moving averages and standard deviations
- Autocorrelation functions
- Trend analysis (linear, polynomial)
- Change point detection

**Frequency Domain Analysis**:
- Power spectral density
- Dominant frequency identification
- Periodicity detection
- Spectral centroid calculation

#### Multivariate Analysis
**Correlation Analysis**:
- Pearson (linear relationships)
- Spearman (monotonic relationships)
- Partial correlations (controlling for confounds)

**Principal Component Analysis**:
- Dimensionality reduction
- Feature importance ranking
- Multicollinearity detection

**Clustering Analysis**:
- Temporal pattern identification
- Participant grouping
- Behavior state classification

---

## Output Formats

### Feature Files Structure

#### Primary Feature File (`.csv`)
**Column Structure**:
```
blink_dist,mouth_dist,head_rotation_angle,center_face_x,center_face_y,
center_face_magnitude,left_eye_x,left_eye_y,left_eye_magnitude,...
```

**Row Structure**:
- Each row represents one time frame
- Frame rate matches original recording
- NaN values indicate masked/missing data

**Metadata Columns**:
- `participant_file`: Original filename
- `frame_number`: Sequential frame index
- `time_seconds`: Timestamp from recording start

#### Quality Control Files

**Keypoint Quality (`keypoint_bad_windows.csv`)**:
- Per-landmark quality assessment
- Percentage of bad windows for each landmark
- Identifies problematic landmarks

**Metric Quality (`metric_bad_windows.csv`)**:
- Per-region quality assessment
- Summary statistics for facial regions
- Overall data quality indicators

**Bad Window Details (`metric_bad_window_indices.csv`)**:
- Precise time ranges of unreliable data
- Used for temporal masking
- Enables selective data exclusion

### Analysis-Ready Datasets

#### Summary Statistics File
**Structure**:
```
participant_file,total_frames,duration_minutes,blink_dist_mean,
blink_dist_std,mouth_dist_mean,mouth_dist_std,...
```