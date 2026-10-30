# Calibration

Scoring produces three numbers per peer, and only the last reaches the optimizer.

| Stage | Value | Range | Purpose |
|---|---|---|---|
| Points | `final_points` | 0-100 | absolute rubric score |
| Calibrate | `calibrated` | 0-100 | cohort-relative display score |
| Signal | `signal` | -1..1 | aggregation input |

## The transform

```
calibrated = TARGET_MEAN + SHRINK * (points - cohort_mean)
```

It is one affine map, so it preserves order for any positive `SHRINK` and pulls the tails
toward a common band without erasing the ranking. Defaults: `TARGET_MEAN = 62.0`,
`SHRINK = 0.35`, in settings under `NORMALIZATION`. `SHRINK` must be within `[0, 1]` and
`TARGET_MEAN` within `(0, 100)`; both are validated when settings load.

Worked example. Four peers score 20, 40, 60, 80 (mean 50). With the defaults they calibrate to
62 + 0.35 x (-30, -10, +10, +30) = **51.5, 58.5, 65.5, 72.5**. The 60-point spread
becomes 21, and the order is unchanged.

## Why void is excluded

A void episode is pinned to 0 points. Letting a 0 into the cohort mean would drag every honest
peer's calibrated score down by the amount one peer cheated, so the void peer is excluded from
the mean (`exclude_void`, on by default), stays at the floor itself, and is reported in
`NormalizationReport`. A cohort smaller than `min_cohort` (default 2) is not calibrated; the
report says why, instead of calibrating against a mean of one.

## Choosing `SIGNAL_SOURCE`

`COOPETITION.SIGNAL_SOURCE` decides what the aggregator and trainer consume.

- `"final_score"` (default): the continuous reward/penalty score of [scoring.md](scoring.md).
- `"points"`: the rubric points, calibrated then mapped to the signal range. Pick this when the
  rubric is what you want optimised and you accept that only guarded points count (a rubric
  line with missing evidence is denied and recorded, not granted).

Either way points are always computed and attached to the scorecard, and they never feed
`final_score`. A checkpoint records the source it was written under and refuses to resume under
the other.
