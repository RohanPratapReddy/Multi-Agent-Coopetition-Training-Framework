# conf/watchdog.yaml — owns: WATCHDOG. Unknown keys are REFUSED: a typo in a
# threshold is a watchdog that never fires, and silence is the one failure this
# component cannot report about itself.

WATCHDOG:
  # abort | rollback | ignore. A NaN loss that is ignored is a run that wastes
  # the remaining epochs.
  NAN_POLICY: abort
  INF_POLICY: abort

  LOSS_INCREASE_RATIO: 3.0
  LOSS_PATIENCE: 2

  MAX_GRAD_NORM: 100.0
  GRAD_NORM_PATIENCE: 1

  SCORE_REGRESSION_TOLERANCE: 0.10
  SCORE_PATIENCE: 3

  ENTROPY_COLLAPSE_RATIO: 0.05
  ENTROPY_PATIENCE: 2

  KAPPA_MIN: 1.0e-3
  KAPPA_MAX: 1000.0

  STALE_EPOCH_SECONDS: 1800.0
  MIN_THROUGHPUT_RATIO: 0.25
  MAX_STEP_TIME_DRIFT: 5.0

  MAX_RSS_GROWTH_RATIO: 2.0
  MAX_GPU_MEMORY_RATIO: 0.95

  MAX_CHECKPOINT_AGE_SECONDS: 3600.0
  WARNING_SPIKE_THRESHOLD: 10

  MAX_TREE_GROWTH_RATIO: 2.0
  DEAD_BRANCH_PATIENCE: 2
  CHILD_STARVATION_PATIENCE: 3
  DEPTH_COST_RATIO: 4.0

  # The watchdog's own cadence and response.
  SNAPSHOT_EVERY_EPOCHS: 1
  SNAPSHOT_EVERY_SECONDS: null
  SNAPSHOT_ON_DEGRADATION: true
  # none | log | snapshot | rollback | prune_branch | pause | abort
  ON_DEGRADATION: snapshot
  ON_CRITICAL: rollback
  KEEP_HISTORY: 512
