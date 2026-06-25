# TODO

## Next steps

- **Minimum similarity threshold for assignment.** IMU-9 in the Halmstadt session has
  near-zero correlation with every optical player (best r < 0.05), yet still receives an
  assignment via the Hungarian algorithm. A minimum r threshold would allow the pipeline
  to emit an "unmatched" output instead of a spurious assignment. This breaks the strict
  1-to-1 Hungarian guarantee, so the mechanism needs thought (e.g. post-hoc rejection vs
  modifying the cost matrix). The threshold value also needs to be set: we have no ground
  truth for what r constitutes a reliable match, so this requires either empirical
  calibration across more sessions or a principled noise floor estimate.
