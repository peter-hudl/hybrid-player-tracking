# IMU-Optical Identity Alignment: Experiment Log

## Summary

Best result to date: **8/9 (89%)** on full session (E02, E04b, E05, E06). Sole failure: jersey 14 (ghost-track contamination in optical data, upstream issue). Jersey 9 excluded (device shutdown).

Minimum viable window: ~30 minutes. 15-minute windows are unreliable across all approaches.

## Experiments

| ID | Description | IMU method | Key params | Full session | Best 15m | Mean 15m | Notes |
|----|-------------|-----------|------------|-------------|----------|----------|-------|
| E01 | Activity envelope, 7 GT | s301 ac_earth | conf>0.3, 5s RMS, speed [0,12] | 6/7 (86%) | 5/7 (71%) | 40% | Phase 2a. 7 GT only. Optical accel failed; switched to speed RMS. |
| E02 | Activity envelope, 10x10 | s301 ac_earth | same | 8/9 (89%) | 7/9 (78%) | 39% | Phase 2b. J2/J16 added to GT. J14->opt9 persistent. |
| E03 | Activity + gait/stride | s301 ac_earth | ankle sep [0.5,4.5]Hz, 3s FFT | 8/9 (89%) | 5/9 (56%) | 33% | Phase 2c. Feature C alone 5/7. No improvement. 10 FPS too coarse for gait. |
| E04a | Complementary filter | raw accel+gyro | alpha=0.98 | 8/9 (89%) | 6/9 (67%) | 41% | Phase 2d. Matches reference. |
| E04b | Body-frame magnitude | raw accel only | \|accel\|-1g | 8/9 (89%) | 7/9 (78%) | 40% | Phase 2d. **Simplest viable approach.** Preferred going forward. |
| E04c | Gyro magnitude | raw gyro only | \|gyro\| | 8/9 (89%) | 5/9 (56%) | 36% | Phase 2d. Competitive but less stable on short windows. |
| E05 | High confidence threshold | s301 ac_earth | conf>0.95, speed [0.5,10] | 8/9 (89%) | 7/9 (78%) | 32% | Ghost-track detections are high-confidence. Doesn't fix J14. |
| E06 | Speed-cadence R2 matching | s301 ac_earth | R2 of speed-cadence fit, 3s FFT | 8/9 (89%) | 4/9 (44%) | 28% | No short-window gain. FFT frequency quantisation limits R2. |
| E07a | A-S fingerprint: accel corr | s301 ac_earth | IMU vs optical accel, 1s mean | 7/9 (78%) | 4/9 (44%) | 28% | Below baseline. |
| E07b | A-S fingerprint: binned profile | s301 ac_earth | per-speed-bin profiles | 0/9 (0%) | 0/9 (0%) | 3% | Complete failure. A-S profiles indistinguishable at 10 Hz. |
| | | | | | | | |
| *Re-runs with conf>0.1, E04b IMU, speed [0.5,10]:* | | | | | | |
| E04b' | Body-frame mag (re-run) | raw accel | conf>0.1, speed [0.5,10] | 8/9 (89%) | 7/9 (78%) | 40% | Unchanged from E04b. Confirms params don't affect result. |
| E06' | Speed-cadence R2 (re-run) | raw accel | conf>0.1, E04b IMU | 8/9 (89%) | 3/9 (33%) | 19% | Full session matches; short windows slightly worse. |
| E07a' | A-S accel corr (re-run) | raw accel | conf>0.1, E04b IMU | 6/9 (67%) | 6/9 (67%) | 32% | Dropped from 7/9. Extra low-conf data adds noise. |
| E08 | Autocorrelation stride freq | raw accel | autocorr + parabolic interp, 3s window | 0/9 (0%) | 4/9 (44%) | 11% | Worse than FFT (E06). Continuous estimates are noisier, not tighter. |
| E09 | Hip world-space oscillation | raw accel | position residual, bandpass [0.5,4.5]Hz | 0/9 (0%) | 1/9 (11%) | 4% | SNR < 1 at 10 FPS. Position noise (~10 cm) >= stride displacement (~5-10 cm). |

## Key findings

1. **Activity envelope (IMU accel RMS vs optical speed RMS) is the only feature that works.** Gait/stride, heading change rate, speed-cadence matching, and A-S fingerprinting all fail to improve over the simple activity envelope.
2. **Raw body-frame accel magnitude is sufficient (E04b).** `|accel_mag| - 1g` with a 5s RMS window matches sensor 301's preprocessed output. Only a 3-axis accelerometer is needed.
3. **Minimum viable window: ~30 minutes.** 15-minute windows give 28-78% accuracy (highly variable). 30-minute and full-session windows consistently reach 89%.
4. **Jersey 14 is an upstream tracking quality issue.** 31% of frames have two simultaneous high-confidence "jersey 14" detections 16-21m apart. No amount of feature engineering or confidence filtering fixes this. Documented, not pursued.
5. **Jersey 9 is unresolvable** due to early device shutdown (~53 min of data).
6. **Gyro adds nothing at 5s windows.** The accelerometer captures rotational accelerations, so at multi-second aggregation windows gyro is redundant.
7. **Optical speed filtering**: [0.5, 10.0] m/s clip, 5s RMS, no additional smoothing. The tracker's Kalman filter already smooths velocity.
8. **10 FPS is below the validated minimum for optical gait analysis** (literature: 25-30 FPS). This is the fundamental constraint on all stride/gait approaches.

## Closed investigations

- **E06**: Speed-conditioned stride matching. Matches baseline (8/9), no short-window improvement. FFT quantisation is not the bottleneck (E08 proved this).
- **E07**: A-S fingerprinting. Below baseline (7/9 best). Optical A-S profiles indistinguishable at 10 Hz.
- **E08**: Autocorrelation stride frequency. Worse than FFT (0/9 full session). Continuous estimates are noisier, not more discriminative.
- **E09**: Hip world-space oscillation. SNR < 1 at 10 FPS. Position noise drowns the stride signal.
- J14 is an upstream tracking quality issue (ghost tracks with miscalibrated confidence). Documented, not pursued.
- Optical gait research complete. See `gait_research_summary.md`.

## Future directions (not pursued)

- **Substitution/presence signal**: use player on/off-pitch timing as a binary matching feature. Strong for matches with subs (6 constraints for free). Not implemented.
- **Higher optical frame rate** (30 FPS): would push Nyquist to 15 Hz and reduce position noise. Hardware-constrained (FC500 encodes at 10 FPS on current Jetson/Orin hardware).
- **3D pose estimation**: stereo triangulation in FC500 overlap zones for vertical keypoint recovery. Requires camera extrinsics.

## Standard going forward

- **IMU method**: E04b (body-frame magnitude, `|accel_mag| - 1g`). Raw accelerometer only, no sensor fusion.
- **Optical**: merged by jersey (team 0, conf>0.1, speed [0.5, 10.0] m/s, 5s RMS.
- **Assignment**: 10x10 Hungarian, 9 verified GT (J9 excluded).

## Constants

- `SYSTEM_DELTA_S = -796.97` (optical_time = wimu_time - 796.97)
- Optical FPS: 10 Hz
- IMU sample rate: ~88 Hz (median dt = 10ms)
- RMS window: 5 seconds
- Jerseys: 2, 3, 6, 7, 8, 9, 12, 14, 16, 17
- Verified GT: 9 (all except J9)
- WIMU team: optical team 0
