# Optical Gait Estimation at 10 FPS: Research Summary

## Key finding

10 FPS is below the validated minimum (25-30 FPS) for stride frequency estimation from 2D pose. No published work validates stride frequency extraction at 10 FPS from pose keypoints. The 100 ms temporal resolution cannot resolve events shorter than 200 ms, and a full sprinting stride cycle is approximately 280-330 ms (only ~3 samples per cycle).

## Most promising alternatives (all feasible at 10 FPS)

### 1. Speed-conditioned stride frequency matching (highest feasibility)

Rather than estimating stride frequency optically, use biomechanical speed-cadence priors. Speed and stride frequency are tightly coupled: cadence = speed / stride_length, and stride length varies narrowly for a given athlete. At 10 FPS, world-space speed is high quality. Constrain the expected stride frequency range given instantaneous speed, then check which IMU device's measured stride frequency is consistent with each optical track's speed profile.

This reframes the problem from "estimate stride frequency optically" to "rank consistency of optical speed with IMU stride frequency given biomechanical priors."

### 2. Acceleration-speed fingerprinting (high feasibility)

The individual acceleration-speed (A-S) profile is an established player fingerprint in sports science. From world-space position at 10 Hz, compute instantaneous speed and build per-player characteristic A0 (max acceleration) and S0 (max speed) across multiple sprint efforts. These profiles differ significantly by position and individual. Studies in rugby and football validate A-S profiling at exactly our sampling rate.

### 3. Hip midpoint world-space oscillation (medium-high feasibility)

Project hip keypoint positions to world coordinates via the existing pitch homography (the system already does this for the main body position). Hip midpoint lateral displacement in world metres oscillates at stride frequency during locomotion. Better SNR than ankle-to-ankle pixel distance because: hips are closer to body centre, larger in image, less occluded from above, and world-space projection normalises for camera distance.

### 4. Pseudo-stereo in FC500 overlap zones (medium feasibility)

The FC500 has 4 cameras with overlapping fields of view. In overlap regions, triangulation could recover vertical (Z) ankle position if camera extrinsics are known or recoverable from pitch-plane features. Relative Z change over time (ankle rising/falling) would give clean stride detection from above. Main obstacle: internal camera geometry needs to be precisely known.

## Low feasibility approaches

- **Video frame interpolation** (RIFE/FILM from 10 FPS): artefacts dominate fast movements, not validated
- **Deep learning gait recognition**: all benchmarks use lateral/frontal views at 25+ FPS; no top-down athletic gait datasets exist

## Practical minimum frame rate

Literature consensus is 25-30 FPS for reliable temporal gait parameters. At 25 FPS, a one-frame error is 40 ms (~15% of a stride phase); at 10 FPS it's 100 ms (~40%). This is not a soft boundary.

## Sources

Key references: WorldPose (ECCV 2024), PMC11097739 (clinical gait from pose), PMC10784250 (GPS A-S profiling), Sports Engineering frame interpolation study, CASIA-B few-frame gait recognition.
