# Optical Tracker Notes (BoT-SORT in aml-unified-inference)

Source: `aml-unified-inference/src/cpp/deepstream_lib/includes/tracker/algorithms/botsort.md` and `botsort_algorithm.hpp`.

## Algorithm

The optical tracking pipeline uses BoT-SORT (Aharon et al., 2022), a ByteTrack extension with a Kalman filter on `[cx, cy, w, h]` (constant-velocity model). Optionally includes a ReID branch (EMA appearance features) controlled by the `reidType` YAML field.

## Relevance to identity alignment

### Velocity is Kalman-filtered

The tracker maintains velocity as a Kalman state variable. The `_WORLD_VELOCITY_METERS_SECOND_X/Y` outputs in the chunkstream are smoothed state estimates, not raw finite differences of position. This means:
- Optical speed already has implicit low-pass smoothing from the Kalman update
- No additional pre-smoothing is needed before the 5s RMS envelope computation
- The speed signal quality is better than naive position differentiation would suggest

### Two independent confidence signals

| Column | Name | Source | What it means |
|--------|------|--------|---------------|
| 5 | `_CONFIDENCE` | Object detector (YOLOX) | How confident the detector is that this bounding box contains a player |
| 15 | `_JERSEY_CONFIDENCE` | Jersey number recognition model | How confident the classifier is about the assigned jersey number |

These are independent. A detection can have high detector confidence (definitely a player) but low jersey confidence (can't read the number). Our `JERSEY_CONF_MIN = 0.1` threshold applies to the jersey classifier output.

### Track lifecycle and fragmentation

1. **Spawning**: a new track is created from an unmatched high-confidence detection (`conf >= newTrackThresh`)
2. **Probation**: the track must accumulate `probationAge` consecutive matches before it emits output. Unconfirmed tracks that miss a detection are removed outright (not kept as lost).
3. **Active**: confirmed tracks participate in all three association stages
4. **Lost**: when a confirmed track has no matching detection, it enters a shadow/lost state and coasts on Kalman prediction for up to `maxShadowTrackingAge` frames (120s = 1200 frames at 10 FPS in our data)
5. **Retired**: lost tracks past `maxShadowTrackingAge` are dropped. If the same player is later detected, they get a new track ID.

This explains the track ID fragmentation in our data: players absent for >120s get new IDs. Within any 120s window, track continuity is preserved.

### Score-fused IoU

Stages 1 and 3 use score-fused IoU cost: `cost = 1 - (1 - iou_dist) * det_score`. This biases the Hungarian assignment toward higher-confidence detections when geometry is ambiguous. Stage 2 (low-confidence rescue) uses raw IoU without score fusion, since the whole point is to recover detections the detector was less certain about.

### No camera motion compensation

The implementation assumes a stationary camera (correct for the FC500 fixed stadium mount). Camera motion compensation (GMC/ECC from the original paper) is not implemented.
