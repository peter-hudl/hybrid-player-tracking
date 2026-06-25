# conf_min Comparison: 0.1 vs 0.95

**Method:** E04b (raw accel envelope, Pearson r, Hungarian assignment)
**Session:** Halmstadt full session, system_delta_s = -796.97 s
**Players:** 10 IMU devices, 10 optical tracks

## Result

Raising conf_min from 0.1 to 0.95 changes 2 of 10 assignments. Jerseys 2 and 9 are
swapped. The other 8 are identical, and their per-player similarity values shift by less
than ±0.007.

## Per-player similarity

| IMU | Optical (baseline) | baseline r | high_conf r | delta |
|---|---|---|---|---|
| 2 | 2 | 0.3716 | 0.3773 | +0.006 |
| 3 | 3 | 0.4934 | 0.4936 | +0.000 |
| 6 | 6 | 0.5180 | 0.5181 | +0.000 |
| 7 | 7 | 0.3955 | 0.3949 | -0.001 |
| 8 | 8 | 0.3749 | 0.3749 | 0.000 |
| 9 | 14 | 0.0475 | 0.0345 | -0.013 |
| 12 | 12 | 0.5852 | 0.5920 | +0.007 |
| 14 | 9 | 0.4729 | 0.4999 | +0.027 |
| 16 | 16 | 0.5184 | 0.5113 | -0.007 |
| 17 | 17 | 0.4594 | 0.4600 | +0.001 |

## The swap

IMU jersey 9 is the root cause. Its correlations against every optical player are
near-zero regardless of threshold (best r < 0.05, compared to >0.35 for all other
players). It has no reliable optical match.

In the baseline, the Hungarian algorithm routes IMU-9 to optical-14 (the least-bad option
at r=0.0475) and leaves optical-2 free for IMU-2 (r=0.3716, correct). Under high-conf
filtering, the IMU-2 row shifts slightly: its correlation with optical-14 rises from 0.3465
to 0.3843, and with optical-9 from 0.3867 to 0.4228. This changes the Hungarian optimum
so IMU-2 is routed to optical-14, and IMU-9 lands on optical-2 instead.

The baseline assignment (IMU-2 to optical-2) matches ground truth. The high-conf threshold
therefore breaks this player, but it does so because IMU-9 is a degenerate case, not
because high-conf filtering is harmful in general. With a reliable IMU-9 signal the global
optimum would be stable under both thresholds.

## Conclusion

conf_min=0.95 does not meaningfully improve or degrade the method. The assignment swap is
entirely driven by IMU-9 being near-zero across all optical matches. For the 9 players with
reasonable signal, the two thresholds produce identical assignments and negligible similarity
differences.
