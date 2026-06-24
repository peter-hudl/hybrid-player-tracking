# IMU-Optical Player Identity Alignment in Football

## Summary

We investigated whether anonymous IMU wearable devices can be matched to optically-tracked football players using only accelerometer data and optical speed, without GPS or UWB positioning. On a 70-minute U19 training session with 10 players, a simple activity-envelope correlation method correctly identified 8 of 9 evaluable players (89%) using the full session. The method requires only a 3-axis accelerometer (no gyroscope or onboard sensor fusion) and 30+ minutes of data. The sole failure was traced to an upstream optical tracking error, not a limitation of the alignment method. More sophisticated approaches (gait/stride frequency, acceleration-speed profiling, heading change rate) all failed to improve over the baseline, primarily because the 10 FPS optical frame rate is below the validated minimum for gait analysis and insufficient to resolve individual biomechanical signatures. This method assumes time-aligned optical and IMU data (+/- 0.2s) and reasonable quality optical tracking data. In our dataset, the sole failure was a player whose jersey number was systematically misclassified by the optical pipeline, contaminating the merged speed signal with another player's movement. This failure was not recoverable by confidence filtering.

## Background

Player tracking in football increasingly relies on combining wearable sensors with camera-based optical tracking. Current wearable systems (GPS, UWB) provide absolute position, making device-to-player assignment trivial: the GPS position matches the optical position. However, GPS and UWB add cost, bulk, and regulatory constraints. If the tracking problem could be solved with IMU-only wearables (accelerometers, gyroscopes), the hardware simplifies considerably: smaller, cheaper devices with longer battery life.

The challenge is identity alignment. An IMU device measures acceleration and rotation but not position. An optical tracking system observes player positions and velocities but does not know which player wears which device. Matching the two requires finding a shared signal observable from both modalities.

### Dataset

The Halmstadt U19 dataset (2025-08-16) provides redundant modalities for 10 players on one team during a training session:

- **Optical tracking**: Hudl aml-unified-inference pipeline from a Nexus FC500 camera (4 internal cameras, elevated stadium mount). Output at 10 FPS: world-space position (metres, pitch-centred), velocity, acceleration, jersey number classification, team classification, and 17 pose keypoints in image pixel coordinates. Tracking uses BoT-SORT with a constant-velocity Kalman filter; velocity outputs are Kalman state estimates.
- **WIMU wearables**: 10 devices (one per player), each recording raw accelerometer, gyroscope, and attitude (gravity-subtracted earth-frame acceleration) at 100 Hz, plus GPS at 10 Hz and UWB at 20 Hz.

GPS and UWB are used only for time alignment verification and ground truth identity establishment. These are later discarded; the identity alignment method itself uses only accelerometer/gyroscope and optical speed.

### Time alignment

The two recording systems started at different times. We established the session offset by cross-correlating GPS speed with optical speed for each player (using jersey number classification to establish identity for this step). After correcting for per-device start offsets, 7 of 10 players clustered within 0.5 seconds, giving a consensus offset of -796.97 +/- 0.20 seconds (optical time = WIMU time - 797 seconds). Three players failed GPS-based verification (one spurious correlation peak, one device shutdown, one late optical onset) but two of these were later confirmed correct through the identity alignment itself.

**N.B.** This project is not trying to solve time alignment, this was done as a one-off and to establish a ground-truth. For any implementation of this method, it is assumed that both Optical and IMU data streams will arrive time-aligned.

## Method

### Signal extraction

**IMU side.** We compute the total dynamic acceleration magnitude from the raw accelerometer: `dynamic_accel = |sqrt(ax^2 + ay^2 + az^2)| - 1g`, where the 1g subtraction removes the gravity component. This signal captures the total mechanical intensity of movement regardless of device orientation. We resample from 100 Hz to 10 Hz (to match the optical frame rate) and compute a 5-second rolling RMS to produce an activity envelope.

**Optical side.** We use the speed magnitude (`sqrt(vx^2 + vy^2)`) from the tracking output, clipped to [0.5, 10.0] m/s to remove stationary noise and tracking spikes. Speed is already Kalman-filtered by the tracker (BoT-SORT constant-velocity model), so no additional smoothing is applied. We compute a matching 5-second rolling RMS to produce the optical activity envelope.

To handle track ID fragmentation (the tracker assigns new IDs when a player is absent for >120 seconds), we merge all optical detections for each jersey number using the optical jersey classifier (team 0, confidence > 0.1). This collapses ~30 track fragments into 10 player-level time-series spanning the full session.

### Identity assignment

For each candidate pairing of WIMU device i with optical player j, we compute the Pearson correlation between their activity envelopes over the analysis window (after time-aligning using the known session offset). This produces a 10x10 similarity matrix. We solve the optimal assignment using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on the cost matrix `1 - similarity`), which finds the global one-to-one matching that maximises total similarity.

### Why the method works

The 5-second RMS window projects both signals onto the same latent variable: activity state. Over 5 seconds, a player who is actively running produces both high acceleration RMS (from footstrikes, direction changes, and speed adjustments) and high speed RMS. A stationary player produces low values in both. The Pearson correlation captures whether two signals share the same temporal pattern of active and inactive periods.

The discriminative information is in WHEN each player transitions between activity states, which differs by playing position and individual behaviour. A centre-back and a winger have different activity profiles even during the same drill. The method requires 30+ minutes because the activity envelope at 5-second resolution operates at ~0.1 Hz, and the individually distinctive patterns are at even lower frequencies. Team-level synchrony (all players running and stopping together) is the main source of noise.

Note that speed and acceleration are not the same physical quantity (acceleration is the derivative of speed). The method does not depend on their instantaneous relationship. The 5-second RMS collapses both to a common activity-level representation, abstracting away the kinematic differences between the two measurement modalities.

## Results

One player (jersey 9) was excluded due to data quality issues and low temporal overlap due to missing data.

### Accuracy

On the full session (~70 minutes), the method correctly assigns 8 of 9 evaluable players (89%). 

| Window length | Best accuracy | Mean accuracy (across windows) |
|---|---|---|
| 5 minutes | 7/9 (78%) | ~40% |
| 15 minutes | 7/9 (78%) | ~40% |
| 30 minutes | 8/9 (89%) | ~55% |
| Full session (~70 min) | 8/9 (89%) | 89% |

Short windows (5-15 minutes) are highly variable, ranging from 0% to 78% depending on which portion of the session is analysed. The second half of the session produces more discriminative activity patterns than the first half (warm-up). Stable accuracy requires at least 25-30 minutes.

### The jersey 14 failure

Jersey 14 is consistently assigned to optical player 9 across all window lengths, all feature types, and all IMU methods. This investigation suggests that this is likely an upstream optical tracking quality issue, not a limitation of the alignment method.

Ghost-track duplicates (two track IDs claiming the same jersey in the same frame, typically 10-30 metres apart) are present for all players in this dataset, caused by both teams sharing jersey numbers. However, for most players the deduplication step (keeping the highest-confidence detection per timestamp) reliably selects the correct detection. Jersey 14 is the exception because a significantly larger fraction of its detections are low-confidence:

| Metric | Jersey 14 (failed) | 8 successful players (range) |
|---|---|---|
| Detections below 0.95 confidence | 17.9% | 0.0 - 6.0% |
| Mean jersey confidence | 0.972 | 0.975 - 0.986 |
| Duplicate-frame rate | 31.2% | 14.4 - 50.8% |
| Median duplicate spatial separation | 15.4 m | 6.5 - 31.1 m |

The duplicate-frame rate and spatial separation are NOT what distinguishes J14 (jersey 7 has a 50.8% duplicate rate at 31.1 m median separation and is still correctly matched). Jersey 14 does have a notably higher low-confidence detection rate (17.9% below 0.95 confidence, vs 0-6% for successful players), indicating the classifier generally struggles with this jersey number. However, raising the confidence threshold to 0.95 did not fix the problem: the remaining high-confidence detections still produced a contaminated speed signal. The low-confidence fraction is a symptom of the classifier's difficulty with this jersey, not the direct cause of failure. The problematic detections are ones the classifier is confidently wrong about, not ones it is uncertain about.

The contaminated optical speed signal for jersey 14 correlates more strongly with jersey 9's IMU signal (r = 0.46) than with jersey 14's own IMU signal (r = 0.28).

We were unable to identify a single quantitative threshold that reliably predicts whether a player's optical data is good enough for identity alignment. The low-confidence fraction is correlated with failure but not causal, and ghost-track duplicate rates vary widely among successful players (14-51%). This remains an open question requiring more sessions to resolve.

### Alternative approaches evaluated

We tested 11 experiments across four categories. None improved over the baseline activity envelope method.

**Gait and stride frequency (E03, E06, E08, E09).** We attempted to extract stride frequency from both IMU (vertical acceleration FFT and autocorrelation) and optical (ankle separation oscillation, hip world-space residual, speed-cadence modelling) data. IMU-side stride frequency estimation is clean at 88 Hz, but all optical-side approaches failed. The literature consensus is 25-30 FPS minimum for stride frequency from pose; at 10 FPS, the Nyquist ceiling, position noise (~10 cm, comparable to the ~5-10 cm lateral hip displacement during running), and ankle keypoint quality are all limiting factors. Speed-cadence matching (pairing IMU stride frequency with optical speed via biomechanical priors) achieved 8/9 on the full session but did not improve short-window accuracy.

**Acceleration-speed fingerprinting (E07).** Per-player acceleration-speed profiles computed from optical data were indistinguishable between players (inter-player correlation > 0.95). The profile shape is dominated by universal human locomotion physics at 10 Hz resolution, not individual biomechanical signatures. Direct acceleration correlation (IMU vs optical tangential acceleration) achieved 6/9, below baseline, because optical tangential acceleration (the derivative of speed) amplifies tracking noise.

**Heading change rate (Feature B in E01-E03).** Gyroscope yaw rate (IMU) vs optical heading change rate produced zero discriminative signal (matched-pair r indistinguishable from mismatched-pair r). At 10 FPS, directional changes are smeared across ~5 samples, and the resulting RMS envelope captures the frequency of turns (shared across the team) rather than individual turning patterns.

## Discussion

### The 10 FPS constraint

The optical tracking frame rate is the binding constraint on what features can be extracted. At 10 FPS:

- Speed (first derivative of position) is usable. The tracker's Kalman filter provides adequate smoothing.
- Acceleration (second derivative) is dominated by tracking jitter. Not usable.
- Stride frequency (~1.2-3.5 Hz) is near the 5 Hz Nyquist limit. Not reliably estimable from pose or position data.
- Lateral body oscillation (~5-10 cm amplitude) is below the ~10 cm position noise floor. Not detectable.

The only optical signal with sufficient SNR for identity alignment is speed, and the only IMU signal that correlates with speed at the activity-envelope level is total acceleration magnitude. This explains why the simplest method is also the best: more sophisticated features require more optical signal quality than 10 FPS provides.

The cameras record at 30 FPS natively; the 10 FPS bottleneck is in the GPU encoding pipeline. At 30 FPS, stride frequency estimation from pose keypoints would become feasible (Nyquist at 15 Hz, 3x the position sample density), potentially enabling gait-based matching on shorter time windows.

### Minimum window length

The 30-minute minimum for reliable matching is a practical constraint. In training or match sessions (typically 60-90 minutes), this is acceptable provided real-time data is not required.

For real-time or shorter-window applications, two upgrades could reduce the minimum:

1. **Presence/absence augmentation.** Using player on/off-pitch timing as a hard constraint before activity-envelope matching. Each substitution provides a strong binary signal (a player's IMU goes inactive while their optical track disappears).

2. **HMM-based state sequence matching.** The current method correlates continuous RMS envelopes, but the discriminative information is concentrated in state transitions (when a player starts or stops running). Modelling the latent activity state with a Hidden Markov Model (shared states across sensors, independent observation models for acceleration and speed) could focus the comparison on the informative transitions rather than the uninformative steady-state periods. A simpler first step: discretise the RMS envelopes into activity states via clustering and compare state sequences with edit distance.

Both upgrades are complementary: presence augmentation handles edge cases (partial sessions, substitutions), while HMM-based matching improves the core signal quality.

### Limitations

This study tested one session with one device type (WIMU) and one optical tracking system (Nexus FC500 via aml-unified-inference). Key assumptions that need validation on additional data:

- **Single session.** The Halmstadt U19 training session may not be representative of all training formats or match scenarios. The balance of synchronised vs individualised activity (which determines the discriminative content) varies by session type.
- **Single IMU device type.** The WIMU device records at 100 Hz with a specific accelerometer sensitivity and noise floor. The method should generalise to any accelerometer with comparable specifications, but this needs verification on other devices (particularly Titan wearables, when data becomes available).
- **Single optical system.** The 10 FPS BoT-SORT pipeline produces Kalman-filtered velocity with a specific noise profile. Other optical tracking systems (different cameras, trackers, or frame rates) may produce different speed signal quality.
- **Team size.** With 10 players the assignment problem is tractable. Scaling to 22 players (both teams) would require either team classification as a pre-filter (available from the tracker) or a larger similarity matrix where the signal-to-noise margin becomes tighter.
- **No reliable optical quality metric.** The sole failure (jersey 14) correlates with higher jersey classifier uncertainty, but confidence filtering did not resolve it. We were unable to define a quantitative threshold that predicts whether a player's tracking data is sufficient. More sessions with varying tracking quality are needed.
- **Dependency on optical per-player identity.** The method merges fragmented optical tracks into per-player time-series using jersey number classification. In training scenarios without numbered shirts, this step would depend on the tracker's appearance-based ReID (EMA feature embeddings in BoT-SORT) to maintain stable per-player identities across the session. If ReID is unreliable, track fragmentation increases substantially and the per-fragment signal becomes too short for reliable correlation. The quality of optical identity maintenance (whether via jersey numbers or ReID) is a prerequisite for this method.
