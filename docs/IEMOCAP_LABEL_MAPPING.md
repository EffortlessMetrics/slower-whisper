# IEMOCAP Label Mapping and Preprocessing Guide

This document details how IEMOCAP emotion labels are mapped to slower-whisper's emotion models and the preprocessing steps required for accurate evaluation.

## Overview

IEMOCAP uses a **richer emotion taxonomy** than most pre-trained emotion recognition models. This creates a mapping challenge between:

1. **IEMOCAP labels** (10 categorical + 3 dimensional)
2. **Model predictions** (typically 6-8 categorical emotions + 3 dimensional)

This guide provides:
- Canonical label mappings
- Preprocessing recommendations
- Evaluation strategies for handling label mismatch

## IEMOCAP Emotion Taxonomy

### Categorical Emotions

IEMOCAP annotates 10 categorical emotions:

| Code  | Full Label   | Description | Frequency (approx) |
|-------|--------------|-------------|-------------------|
| `neu` | neutral      | Neutral affect, no strong emotion | 36% |
| `fru` | frustrated   | Frustration, annoyance (IEMOCAP-specific) | 19% |
| `exc` | excited      | Excitement, enthusiasm (IEMOCAP-specific) | 15% |
| `sad` | sad          | Sadness, sorrow, melancholy | 13% |
| `ang` | angry        | Anger, irritation, hostility | 11% |
| `hap` | happy        | Happiness, joy, contentment | 5% |
| `oth` | other        | Emotions not fitting above categories | 1% |
| `fea` | fearful      | Fear, anxiety, worry | <1% |
| `sur` | surprised    | Surprise, astonishment | <1% |
| `dis` | disgusted    | Disgust, revulsion | <1% |

**Key characteristics:**
- **Highly imbalanced:** Neutral dominates (36%)
- **IEMOCAP-specific emotions:** Frustrated and excited are unique to this dataset
- **Rare emotions:** Fear, surprise, disgust have <1% frequency each
- **Ambiguous "other":** Should be excluded from evaluation

### Dimensional Emotions

IEMOCAP provides 3 dimensional ratings (1-5 scale):

| Dimension | Description | Scale |
|-----------|-------------|-------|
| **Valence (VAL)** | Pleasantness/unpleasantness | 1 = very negative, 5 = very positive |
| **Activation (ACT)** | Energy/arousal level | 1 = very calm, 5 = very excited |
| **Dominance (DOM)** | Control/power | 1 = very submissive, 5 = very dominant |

**Annotation process:**
- Multiple annotators rate each utterance (typically 3-5 annotators)
- Categorical label determined by majority vote
- Dimensional ratings averaged across annotators

**Inter-annotator agreement:**
- Categorical: ~70% agreement
- Dimensional: Moderate correlation (r ~0.6-0.7)

## Pre-trained Model Taxonomies

### Categorical Model (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)

Trained on multiple datasets, outputs **7 emotions**:

| Model Label | Description |
|-------------|-------------|
| `angry` | Anger, irritation |
| `disgust` | Disgust, revulsion |
| `fear` | Fear, anxiety |
| `happy` | Happiness, joy |
| `neutral` | Neutral affect |
| `sad` | Sadness, sorrow |
| `surprise` | Surprise, astonishment |

**Notable gaps:**
- No `frustrated` (must map to `angry`)
- No `excited` (must map to `happy`)

### Dimensional Model (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)

Trained on MSP-Podcast, outputs **3 dimensions** (0-1 scale):

| Dimension | Scale |
|-----------|-------|
| `valence` | 0.0 = very negative, 1.0 = very positive |
| `arousal` | 0.0 = very calm, 1.0 = very excited |
| `dominance` | 0.0 = very submissive, 1.0 = very dominant |

**Important:** Note the **scale difference** (IEMOCAP: 1-5, Model: 0-1)

## Label Mapping Strategies

### Strategy 1: Conservative Mapping (Recommended for Publication)

Only evaluate on emotions present in both IEMOCAP and model:

```python
CONSERVATIVE_MAPPING = {
    # Direct matches
    "neu": "neutral",
    "ang": "angry",
    "sad": "sad",
    "hap": "happy",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",

    # Exclude IEMOCAP-specific and ambiguous
    "exc": None,  # Excluded (no clear mapping)
    "fru": None,  # Excluded (no clear mapping)
    "oth": None,  # Excluded (ambiguous)
}
```

**Pros:**
- No questionable mappings
- Clean evaluation on standard emotions

**Cons:**
- Discards ~35% of IEMOCAP data (frustrated + excited + other)
- Cannot evaluate model's handling of high-arousal emotions

### Strategy 2: Aggressive Mapping (Recommended for Development)

Map all IEMOCAP emotions to nearest model category:

```python
AGGRESSIVE_MAPPING = {
    # Direct matches
    "neu": "neutral",
    "ang": "angry",
    "sad": "sad",
    "hap": "happy",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",

    # IEMOCAP-specific mapped by arousal/valence
    "exc": "happy",  # Excited → happy (high arousal, positive valence)
    "fru": "angry",  # Frustrated → angry (high arousal, negative valence)

    # Exclude only truly ambiguous
    "oth": None,
}
```

**Pros:**
- Uses ~99% of IEMOCAP data
- Evaluates model's robustness to arousal variations

**Cons:**
- Penalizes model for distinguishing excited/happy or frustrated/angry
- May underestimate true accuracy

**Recommendation:** Use **aggressive mapping** for development and debugging, **conservative mapping** for publication.

### Strategy 3: Hierarchical Mapping (Best for Analysis)

Evaluate at multiple granularities:

```python
# Level 1: Valence only (3 classes)
VALENCE_MAPPING = {
    "neu": "neutral",
    "ang": "negative",
    "sad": "negative",
    "fru": "negative",
    "fea": "negative",
    "hap": "positive",
    "exc": "positive",
    # ... others excluded or mapped
}

# Level 2: Standard emotions (aggressive mapping)
# Level 3: Full IEMOCAP taxonomy (if future models support it)
```

**Use cases:**
- Debugging: Check if errors are valence-level or fine-grained
- Model comparison: Compare valence accuracy vs. granular accuracy
- Reporting: Report both coarse-grained and fine-grained performance

## Dimensional Mapping

### Scale Conversion

Convert IEMOCAP's 1-5 scale to model's 0-1 scale:

```python
def iemocap_to_model_scale(iemocap_value: float) -> float:
    """Convert IEMOCAP 1-5 scale to model 0-1 scale.

    Args:
        iemocap_value: Rating on 1-5 scale (1=low, 5=high)

    Returns:
        Normalized value on 0-1 scale
    """
    # Linear scaling
    return (iemocap_value - 1.0) / 4.0

# Examples:
# 1.0 → 0.0 (minimum)
# 3.0 → 0.5 (neutral)
# 5.0 → 1.0 (maximum)
```

**Important:** This assumes **linear relationship** between scales. In practice, annotators may not use scales uniformly (e.g., avoiding extremes).

### Dimension Name Mapping

IEMOCAP's "Activation" ≈ Model's "Arousal":

```python
DIMENSION_MAPPING = {
    "valence": "valence",  # Direct match
    "arousal": "activation",  # IEMOCAP calls this "activation"
    "dominance": "dominance",  # Direct match
}
```

## Preprocessing Steps

### 1. Filter Ambiguous Labels

```python
def should_include_sample(emotion_code: str, strategy: str = "aggressive") -> bool:
    """Determine if sample should be included in evaluation.

    Args:
        emotion_code: IEMOCAP 3-letter code (e.g., "ang")
        strategy: "conservative" or "aggressive"

    Returns:
        True if sample should be included
    """
    # Always exclude "other"
    if emotion_code == "oth":
        return False

    if strategy == "conservative":
        # Exclude IEMOCAP-specific emotions
        return emotion_code not in ["exc", "fru", "oth"]

    # Aggressive: include all except "other"
    return True
```

### 2. Handle Multiple Annotations

IEMOCAP provides multiple annotator ratings. For evaluation:

**Categorical:**
- Use **majority vote** (already provided in EmoEvaluation files)
- Optionally filter by **annotator agreement threshold**:

```python
def filter_by_agreement(annotations: list[str], min_agreement: float = 0.6) -> str | None:
    """Filter samples by inter-annotator agreement.

    Args:
        annotations: List of categorical labels from multiple annotators
        min_agreement: Minimum fraction agreeing on majority label

    Returns:
        Majority label if agreement >= threshold, else None
    """
    if not annotations:
        return None

    counter = Counter(annotations)
    majority_label, majority_count = counter.most_common(1)[0]
    agreement = majority_count / len(annotations)

    if agreement >= min_agreement:
        return majority_label
    return None
```

**Dimensional:**
- Use **mean** of all annotator ratings (standard approach)
- Optionally filter by **standard deviation**:

```python
def filter_by_variance(ratings: list[float], max_std: float = 1.0) -> float | None:
    """Filter samples by rating variance.

    Args:
        ratings: List of dimensional ratings from multiple annotators
        max_std: Maximum allowed standard deviation

    Returns:
        Mean rating if std <= threshold, else None
    """
    if not ratings:
        return None

    mean_rating = np.mean(ratings)
    std_rating = np.std(ratings)

    if std_rating <= max_std:
        return mean_rating
    return None
```

### 3. Handle Imbalanced Data

IEMOCAP is highly imbalanced (neutral 36%, happy 5%). Options:

**Option A: Stratified Sampling**
```python
def stratified_sample(samples: list, target_per_class: int = 100) -> list:
    """Sample equal numbers from each emotion class.

    Args:
        samples: List of evaluation samples
        target_per_class: Target samples per emotion class

    Returns:
        Balanced sample list
    """
    import random
    from collections import defaultdict

    by_emotion = defaultdict(list)
    for sample in samples:
        emotion = sample.reference_emotions[0]
        by_emotion[emotion].append(sample)

    balanced = []
    for emotion, emotion_samples in by_emotion.items():
        # Sample with replacement if needed
        n_to_sample = min(target_per_class, len(emotion_samples))
        balanced.extend(random.sample(emotion_samples, n_to_sample))

    return balanced
```

**Option B: Weighted Metrics**
- Report **weighted F1** (default in `eval_emotion.py`)
- Report **macro F1** for class-agnostic performance
- Report **per-class F1** for transparency

**Option C: Oversample Rare Classes**
- Duplicate samples from rare emotions (happy, fear, disgust, surprise)
- Only for training, **not** for evaluation

### 4. Audio Quality Filtering

Some IEMOCAP clips have issues:

```python
def should_include_audio(audio_path: Path, min_duration: float = 0.5, max_duration: float = 10.0) -> bool:
    """Filter audio clips by quality criteria.

    Args:
        audio_path: Path to WAV file
        min_duration: Minimum clip duration (seconds)
        max_duration: Maximum clip duration (seconds)

    Returns:
        True if audio meets quality criteria
    """
    import soundfile as sf

    try:
        info = sf.info(str(audio_path))

        # Check duration
        if info.duration < min_duration or info.duration > max_duration:
            return False

        # Check sample rate (IEMOCAP is 16kHz)
        if info.samplerate != 16000:
            return False

        # Check channels (should be mono)
        if info.channels != 1:
            return False

        return True

    except Exception:
        return False
```

## Evaluation Metrics

### Categorical Metrics

1. **Accuracy** (overall correctness)
   - Standard metric, but misleading for imbalanced data
   - Can be high even with poor rare-class performance

2. **F1-score (weighted)** (recommended)
   - Harmonic mean of precision and recall
   - Weighted by class frequency (accounts for imbalance)

3. **F1-score (macro)** (recommended)
   - Unweighted average across classes
   - Shows performance on rare emotions

4. **Confusion matrix**
   - Reveals systematic errors (e.g., frustrated → angry)

5. **Per-class precision/recall**
   - Granular analysis of strengths/weaknesses

### Dimensional Metrics

1. **Mean Absolute Error (MAE)** (primary metric)
   - Average absolute difference: |predicted - reference|
   - Range: 0-1 (lower is better)
   - Interpretable in original scale

2. **Root Mean Square Error (RMSE)**
   - Square root of mean squared error
   - Range: 0-1 (lower is better)
   - Penalizes large errors more than MAE

3. **Pearson Correlation (r)**
   - Linear correlation between predicted and reference
   - Range: -1 to 1 (higher is better, 1 = perfect)
   - Insensitive to scale/offset (tests ranking, not calibration)

4. **Concordance Correlation Coefficient (CCC)**
   - Combines correlation with calibration
   - Stricter than Pearson r
   - Preferred in emotion recognition research

### Baseline Comparison

Always report baseline performance:

```python
# Categorical baseline: majority class (neutral)
baseline_accuracy = max_class_frequency  # ~0.36 for neutral

# Dimensional baseline: mean predictor
baseline_mae = {
    "valence": std_dev_valence,  # Typically ~0.8-1.0 on 1-5 scale
    "arousal": std_dev_arousal,
    "dominance": std_dev_dominance,
}
```

## Common Pitfalls

### 1. Ignoring Label Mismatch

**Problem:** Directly comparing IEMOCAP labels to model predictions without mapping.

**Example:**
```python
# WRONG: Model never predicts "frustrated"
if iemocap_label == "frustrated" and model_prediction == "frustrated":
    correct += 1  # This will always be False!
```

**Solution:** Use explicit mapping (see Strategy 2 above).

### 2. Forgetting Scale Conversion

**Problem:** Comparing IEMOCAP's 1-5 scale directly to model's 0-1 scale.

**Example:**
```python
# WRONG: Different scales!
mae = abs(iemocap_valence - model_valence)  # 3.0 vs 0.5 → MAE = 2.5 (meaningless)
```

**Solution:** Convert to common scale first (see Dimensional Mapping above).

### 3. Not Handling "Other" Category

**Problem:** Including ambiguous "other" emotion in evaluation.

**Example:**
```python
# WRONG: "other" has no clear semantics
mapping["oth"] = "neutral"  # Arbitrary and wrong!
```

**Solution:** Always exclude `oth` from evaluation.

### 4. Overfitting to IEMOCAP Taxonomy

**Problem:** Tuning model/thresholds on IEMOCAP, then reporting those metrics.

**Example:**
```python
# WRONG: Optimizing on test set!
best_threshold = tune_threshold(iemocap_test_set)
accuracy = evaluate(iemocap_test_set, threshold=best_threshold)
```

**Solution:** Use proper train/dev/test splits (see IEMOCAP_SETUP.md).

### 5. Ignoring Session Effects

**Problem:** Mixing sessions in train/test splits → speaker leakage.

**Example:**
```python
# WRONG: Same speakers in train and test!
random.shuffle(all_clips)
train = all_clips[:8000]
test = all_clips[8000:]
```

**Solution:** Use session-based splits (Session 1-4 train, Session 5 test).

## Recommended Evaluation Protocol

For publication-quality results:

1. **Use session-based splits** (no speaker leakage)
   - Train: Sessions 1-3
   - Dev: Session 4
   - Test: Session 5

2. **Report both mapping strategies**
   - Conservative (direct matches only)
   - Aggressive (include frustrated/excited)

3. **Report multiple metrics**
   - Categorical: Accuracy, weighted F1, macro F1, confusion matrix
   - Dimensional: MAE, RMSE, Pearson r (or CCC if available)

4. **Include baseline comparison**
   - Majority class for categorical
   - Mean predictor for dimensional

5. **Provide per-class breakdown**
   - Show F1 for each emotion
   - Highlight rare-class performance

6. **Report sample sizes**
   - Total samples
   - Samples per emotion class
   - Samples excluded and why

## Implementation Example

See `/home/steven/code/Python/slower-whisper/benchmarks/eval_emotion.py` for a complete implementation following these guidelines.

Key features:
- Implements aggressive mapping strategy
- Handles scale conversion automatically
- Reports comprehensive metrics
- Saves detailed results for analysis

## References

1. Busso, C., et al. (2008). "IEMOCAP: Interactive emotional dyadic motion capture database." Language Resources and Evaluation.

2. Schuller, B., et al. (2018). "The INTERSPEECH 2009 emotion challenge." INTERSPEECH.

3. Ringeval, F., et al. (2017). "AVEC 2017: Real-life depression, and affect recognition workshop and challenge." ACM Multimedia.

4. Ekman, P. (1992). "An argument for basic emotions." Cognition & Emotion.
