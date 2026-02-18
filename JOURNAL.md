# Video Retrieval & Deduplication Research Journal

## Problem Statement

Video deduplication using semantic embeddings alone is insufficient. Two videos may share semantic content (e.g., Central Park) but represent entirely different recordings. We need to leverage **non-semantic signals** to disambiguate videos that are semantically similar but distinct.

### Motivating Example
- **Cyclist A**: Harlem → Central Park → Financial District
- **Cyclist B**: Chelsea → Central Park → Queens

Mid-frame semantic embeddings would capture "Central Park" for both, leading to false positive duplicate detection. We need signals that capture:
- Direction of travel
- Temporal sequence of scenes
- Motion patterns
- Unique trajectory signatures

---

## 2025-02-17: Initial Brainstorm

### Available Tools

**DINOv3 (Vision Transformer)**
- Self-supervised visual representations
- Patch-level attention maps
- CLS token captures global image semantics
- Patch tokens preserve spatial information

**v-JEPA (Video Joint Embedding Predictive Architecture)**
- Predicts future frame representations (not pixels)
- Captures temporal dynamics and motion patterns
- Learns video-specific structure beyond static frames
- Predictor network encodes "how scenes evolve"

---

### Non-Semantic Signals to Explore

#### 1. Temporal Ordering Signatures
The *sequence* of semantic content is non-semantic information.

**Idea**: Create a "trajectory embedding" by:
- Extracting per-frame embeddings (DINOv3 CLS tokens)
- Computing temporal derivatives: `d_embedding/d_frame`
- Using the sequence of transitions as a fingerprint

**Hypothesis**: Even if frames A and B both pass through Central Park, the *what comes before* and *what comes after* differs. The transition patterns are unique.

**Experiment**:
```
embedding_sequence = [e1, e2, ..., en]
transitions = [e2-e1, e3-e2, ..., en-e(n-1)]
trajectory_signature = hash(transitions) or learned_projection(transitions)
```

#### 2. Motion Field Patterns (v-JEPA)
v-JEPA's predictor learns to forecast future representations. The *prediction error patterns* may encode motion.

**Idea**: Use v-JEPA's prediction residuals as motion fingerprints:
- High residual = unexpected motion/scene change
- Pattern of residuals over time = unique motion signature

**Hypothesis**: Cyclist A traveling south through Central Park has different motion patterns than Cyclist B traveling east.

**Experiment**:
```
for each frame pair (t, t+k):
    predicted = vjepa.predict(frame_t, delta=k)
    actual = vjepa.encode(frame_t+k)
    residual = actual - predicted
    motion_signature.append(residual)
```

#### 3. Attention Flow Patterns
DINOv3 attention maps show *where* the model looks. The temporal evolution of attention is non-semantic.

**Idea**: Track attention center-of-mass over time:
- Extract attention maps from DINOv3
- Compute spatial attention centroid per frame
- The trajectory of attention centroids is a motion signal

**Hypothesis**: Camera motion, cyclist direction, and scene transitions create unique attention trajectories.

**Experiment**:
```
attention_trajectory = []
for frame in video:
    attn_map = dinov3.get_attention_map(frame)  # [H, W]
    centroid_y = sum(y * attn[y,x] for all y,x) / sum(attn)
    centroid_x = sum(x * attn[y,x] for all y,x) / sum(attn)
    attention_trajectory.append((centroid_x, centroid_y))
```

#### 4. Patch Token Variance
Not all information is in the CLS token. Patch tokens contain spatial detail.

**Idea**: Use patch token statistics as scene texture signatures:
- Variance across patches (spatial complexity)
- Entropy of patch token distribution
- PCA of patch tokens per frame

**Hypothesis**: Two different parks have similar semantics but different spatial textures.

#### 5. Temporal Frequency Analysis
Semantic content is "low frequency" (changes slowly). Non-semantic signals are "high frequency".

**Idea**: Apply temporal FFT to embedding sequences:
- Low frequencies = semantic (what's in the video)
- High frequencies = motion, camera shake, transitions

**Experiment**:
```
embedding_sequence = stack([embed(frame) for frame in video])  # [T, D]
fft_result = fft(embedding_sequence, axis=0)  # [T, D]
high_freq_signature = fft_result[T//2:, :]  # Upper half = high freq
```

#### 6. Scene Transition Graph
Build a graph of scene transitions rather than scene contents.

**Idea**:
- Segment video into scenes (using embedding similarity)
- Create transition matrix: P(scene_j | scene_i)
- Graph structure is non-semantic

**Hypothesis**: A→B→C→D is different from E→B→F→G even if B is Central Park in both.

#### 7. v-JEPA Predictor Weights as Video DNA
The v-JEPA predictor learns video-specific dynamics. Can we extract this?

**Idea**: Fine-tune or probe the predictor on each video, extract weights/activations:
- The predictor's internal state after seeing a video encodes its dynamics
- This is video-specific, not frame-specific

**Challenge**: Computationally expensive. May need lightweight proxy.

---

### Hybrid Approaches

#### Semantic + Non-Semantic Fusion
Don't abandon semantics—combine both signals:

```
final_similarity = α * semantic_sim(v1, v2) + (1-α) * nonsemantic_sim(v1, v2)

where:
  semantic_sim = cosine(mean_embedding(v1), mean_embedding(v2))
  nonsemantic_sim = trajectory_similarity(v1, v2)
```

#### Contrastive Learning on Trajectories
Train a model to distinguish video trajectories:
- Positive pairs: augmented versions of same video
- Negative pairs: different videos (even if semantically similar)
- Loss: contrastive on trajectory embeddings

---

## Experiment Priority Queue

1. **[HIGH] Temporal Derivative Fingerprints**
   - Simple to implement
   - Uses existing DINOv3 embeddings
   - Hypothesis is clear and testable

2. **[HIGH] v-JEPA Prediction Residuals**
   - Leverages v-JEPA's unique capability
   - Directly measures motion/dynamics
   - Needs v-JEPA implementation

3. **[MEDIUM] Attention Centroid Trajectories**
   - Novel approach
   - Interpretable
   - Needs attention extraction from DINOv3

4. **[MEDIUM] Temporal FFT of Embeddings**
   - Simple signal processing
   - May reveal interesting patterns
   - Easy to A/B test

5. **[LOW] Scene Transition Graphs**
   - More complex to implement
   - Requires scene segmentation
   - Good for longer videos

---

## Open Questions

1. How much temporal context is needed? Full video vs. sliding windows?
2. At what layer should we extract DINOv3 features? Earlier layers = more spatial, later = more semantic
3. Can we use v-JEPA's predictor without the full encoder? (efficiency)
4. What's the right distance metric for trajectory comparison?
5. How do we handle videos of different lengths?
6. Does camera motion (shake, pan, zoom) help or hurt?

---

## Related Work to Review

- [ ] "Self-Supervised Video Hashing" papers
- [ ] Video copy detection challenge methods (VCDB, FIVR)
- [ ] v-JEPA paper: what do they say about learned dynamics?
- [ ] Temporal segment networks
- [ ] Video transformers with temporal attention

---

## Implementation Progress

### 2025-02-17: Initial Implementation Complete

Created the experimental framework with the following components:

**Package Structure:**
```
video_retrieval/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── dinov3.py          # DINOv3Encoder with attention extraction
├── fingerprints/
│   ├── __init__.py
│   ├── temporal_derivative.py  # TemporalDerivativeFingerprint, MultiScaleDerivativeFingerprint
│   └── trajectory.py           # TrajectoryFingerprint, DTWTrajectoryMatcher
└── utils/
    ├── __init__.py
    └── video.py               # load_video, extract_frames, frames_to_tensor
```

**Key Classes Implemented:**

1. **`DINOv3Encoder`** (`models/dinov3.py`)
   - `encode_frames()` - Get CLS token embeddings for frames
   - `encode_video()` - Encode full video with batching
   - `get_attention_centroids()` - Extract attention center-of-mass trajectory
   - `get_patch_statistics()` - Variance and entropy of patch tokens
   - Uses RoPE2D positional encoding from entity_tracking

2. **`TemporalDerivativeFingerprint`** (`fingerprints/temporal_derivative.py`)
   - Computes d(embedding)/d(frame) at configurable window sizes
   - Captures magnitude and direction of embedding changes
   - Configurable derivative order (velocity, acceleration)
   - Multiple aggregation modes (mean, histogram, sequence)

3. **`MultiScaleDerivativeFingerprint`** (`fingerprints/temporal_derivative.py`)
   - Combines multiple window sizes (1, 5, 15, 30 frames)
   - Captures motion at different temporal scales
   - Fine-grained (camera shake) to coarse (narrative progression)

4. **`TrajectoryFingerprint`** (`fingerprints/trajectory.py`)
   - Uses attention centroid trajectories as motion signatures
   - Computes velocity, acceleration, curvature of attention path
   - Trajectory smoothing to reduce noise
   - Statistics-based fingerprint vector

5. **`DTWTrajectoryMatcher`** (`fingerprints/trajectory.py`)
   - Dynamic Time Warping for comparing trajectories of different lengths
   - Handles videos of different durations with similar motion patterns

**Experiments:**

1. **`experiments/test_fingerprints.py`**
   - Test with real video files
   - Compares temporal derivative, trajectory, and DTW methods
   - Includes semantic baseline (mean embedding) for comparison

2. **`experiments/test_synthetic.py`**
   - Synthetic trajectories (linear, circular, zigzag, stationary)
   - Validates fingerprints distinguish motion patterns
   - No video files needed - pure algorithm testing
   - Generates visualization of trajectories

---

## Next Steps

1. ~~Set up experimental framework with DINOv3 embeddings~~ ✓
2. ~~Implement temporal derivative fingerprinting~~ ✓
3. ~~Create synthetic test cases (same scene, different trajectories)~~ ✓
4. Get v-JEPA running for prediction residual experiments
5. Run experiments on real videos to validate hypotheses
6. Implement temporal FFT of embeddings (Experiment #4)
