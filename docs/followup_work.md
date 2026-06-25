# Follow-up Work on IMU-Optical Identity Alignment

## Context

This project established a baseline method for matching anonymous IMU wearables to optically-tracked football players. The method achieves 8/9 (89%) accuracy on a single 70-minute training session using activity-envelope correlation (IMU acceleration RMS vs optical speed RMS). The full findings, including 11 experiments and their results, are documented in `imu_optical_alignment_report.md` and `experiment_log.md`.

## Current limitations (in priority order)

1. **Minimum window length: 30 minutes.** 15-minute windows give 14-78% accuracy (highly variable). The activity envelope at 5-second RMS resolution operates at ~0.1 Hz, requiring long observation to accumulate individually distinctive patterns. Team-level synchrony (all players running/stopping together) is the main noise source.

2. **Single session, single device type.** Tested only on one Halmstadt U19 training session with WIMU devices. Needs validation on more sessions, different session types (match vs training), and Titan devices when available.

3. **Optical identity dependency.** The method requires the optical system to maintain per-player identity (currently via jersey number classification) to merge fragmented tracks. In training without numbered shirts, this falls to appearance-based ReID, which may be less reliable.

4. **Jersey 14 failure (upstream tracking quality).** One player was persistently misassigned due to confident jersey misclassification in the optical pipeline. This is not fixable at the alignment level but affects the practical accuracy ceiling.

5. **No minimum similarity threshold for assignment.** IMU-9 in the Halmstadt session has near-zero correlation with every optical player (best r < 0.05), yet still receives an assignment via the Hungarian algorithm. A minimum r threshold would allow the pipeline to emit an "unmatched" output instead of a spurious assignment. This breaks the strict 1-to-1 Hungarian guarantee, so the mechanism needs thought (e.g. post-hoc rejection vs modifying the cost matrix). The threshold value also needs to be set: we have no ground truth for what r constitutes a reliable match, so this requires either empirical calibration across more sessions or a principled noise floor estimate.

## Proposed upgrades

### Upgrade 1: Presence/absence augmentation (highest priority)

Use player on/off-pitch timing as a hard constraint before activity-envelope matching.

**Concept:** A two-stage pipeline:
1. Estimate each IMU device's and optical player's active interval `[onset, offset]`.
2. If a pair's intervals overlap less than ~70% of the shorter interval, veto that cell (set similarity to zero).
3. Compute Pearson correlation only over the overlapping active period.

**Why it helps:**
- Substitutions create strong binary constraints (6 per match = 6 free constraints regardless of window length)
- Partial sessions (jersey 16's late start, jersey 9's early shutdown) are handled explicitly
- Computing correlation only over the overlap avoids diluting the signal with absent periods
- May allow jersey 9 (currently excluded) to be included in the assignment

**Implementation sketch:**
```python
def compute_active_interval(signal, threshold=0.15):
    rms = pd.Series(signal).rolling(50, center=True).std().fillna(0)
    active = rms > threshold
    return active.idxmax(), len(active) - active[::-1].idxmax() - 1

for i, imu in enumerate(imu_devices):
    for j, optical in enumerate(optical_players):
        overlap = interval_overlap(imu_intervals[i], optical_intervals[j])
        if overlap < 0.70:
            similarity[i, j] = 0.0  # hard veto
        else:
            t0 = max(imu_intervals[i][0], optical_intervals[j][0])
            t1 = min(imu_intervals[i][1], optical_intervals[j][1])
            similarity[i, j] = pearson_on_window(imu_rms[i], optical_rms[j], t0, t1)
```

### Upgrade 2: HMM-based state sequence matching

Replace continuous RMS envelope correlation with discrete activity-state sequence comparison.

**Concept:** The current method correlates continuous RMS envelopes, but the discriminative information is concentrated in state transitions (when a player starts/stops running), not in the steady-state values (which are similar across the team). An HMM approach would:

1. Define shared activity states (e.g., standing, walking, jogging, sprinting) from pooled RMS values
2. Decode each device/player's state sequence independently (IMU observes acceleration, optical observes speed, both decode to shared state space)
3. Compare state sequences using edit distance, state-coincidence rate, or transition matrix similarity

**Why it helps:**
- Noise-robust state estimation (HMM transition priors prevent brief noise spikes from flipping state)
- Focuses comparison on informative transitions rather than uninformative steady-state periods
- Independent observation models abstract away the speed-vs-acceleration relationship
- Potentially works on shorter windows (a sequence of 5-10 distinctive transitions may suffice)

**Simpler first step:** Discretise RMS envelopes into activity states via k-means clustering (e.g., 4 clusters on pooled RMS), then compare state sequences with normalised edit distance. This captures much of the HMM benefit without the model fitting.

### Upgrade 3: Multi-session validation

- Run the method on additional training sessions with the same setup
- Test on match data (substitutions provide strong constraints for upgrade 1)
- Test with Titan wearable devices (different accelerometer hardware)
- Assess whether the 30-minute minimum is session-dependent (training with more varied drills may work in less time)

## Key technical context

- **Time alignment**: `optical_time = wimu_session_time + (-796.97)`. This is specific to the Halmstadt session. Any new session needs its own time alignment (Phase 1 in the original plan, or assumed pre-aligned).
- **IMU method (E04b)**: `|sqrt(ax^2 + ay^2 + az^2)| - 1g`, 5-second rolling RMS. Raw accelerometer only.
- **Optical method**: speed clipped to [0.5, 10.0] m/s, 5-second rolling RMS. Merged by jersey (team 0, conf > 0.1).
- **Assignment**: 10x10 Hungarian algorithm on Pearson correlation similarity matrix.
- **Gyro is redundant** at 5-second windows because the accelerometer captures rotational accelerations. Only matters at sub-second timescales.
- **10 FPS is below the minimum for gait analysis** (literature: 25-30 FPS). All stride/gait approaches failed. Don't retry without higher frame rate data.
- **Optical acceleration (ax/ay from the tracker) is unusable** (87% tracking jitter). Only speed is clean enough.
- **Jersey confidence is not calibrated** (0.95 does not mean 95% correct). Use conf > 0.1 as a noise floor, not as a quality filter.

## What NOT to retry

The experiment log documents these dead ends in detail:

- **Heading change rate** (Feature B): zero discriminative signal at 10 FPS
- **Ankle separation stride frequency** (E03): 10 FPS too coarse, image-space keypoints too noisy
- **Speed-cadence R2 matching** (E06): matches baseline but no short-window improvement
- **A-S fingerprinting** (E07): optical profiles indistinguishable between players
- **Autocorrelation stride frequency** (E08): worse than FFT, continuous estimates are noisier
- **Hip world-space oscillation** (E09): position noise (~10 cm) drowns the ~5-10 cm stride signal
- **Confidence threshold filtering** (E05): ghost-track detections are high-confidence
