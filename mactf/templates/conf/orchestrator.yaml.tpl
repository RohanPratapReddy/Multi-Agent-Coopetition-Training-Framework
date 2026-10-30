# conf/orchestrator.yaml — owns: COOPETITION and ORCHESTRATOR.

COOPETITION:
  # Adaptive temperature. Kappa softens or sharpens the weight distribution.
  KAPPA_INIT: 1.0
  KAPPA_MIN: 0.1
  KAPPA_MAX: 10.0
  TARGET_ENTROPY_RATIO: 0.8
  BETA_MOMENTUM: 0.9
  ETA_KAPPA: 0.05

  # framework.md 10: no peer may take the whole blend or vanish from it.
  WEIGHT_FLOOR: 0.02
  WEIGHT_CAP: 0.6
  # Above this disagreement the blend falls back to the reference.
  DISAGREEMENT_MAX: 0.75
  FALLBACK_TO_REFERENCE: true
  TRIM_FRACTION: 0.0
  CONFIDENCE_FLOOR: 0.1
  REFERENCE_EMA: 0.05
  RESET_OPTIMIZERS: true
  AGGREGATE_BUFFERS: top

  # final_score | points. Which scorecard field drives aggregation.
  SIGNAL_SOURCE: final_score

  NORMALIZATION:
    ENABLED: true
    TARGET_MEAN: 62.0
    SHRINK: 0.35
    EXCLUDE_VOID: true
    MIN_COHORT: 2
    SIGNAL_SCALE: 50.0
    CLIP: [0.0, 100.0]

  # framework.md 9.2: report the cooperative component from a bounded
  # per-episode pool so R10/R11/R16 cannot be farmed.
  COOPERATIVE_POOL: true
  COOPERATIVE_BUDGET:
    PER_EPISODE: 6.0
    PER_PEER_CAP: 3.0
    CARRY_OVER: false

ORCHESTRATOR:
  TASK_SOURCE: null
  MAX_EPOCHS: null
  RESUME_FROM: null
  ANNOUNCE: true
  PAUSE_IS_FATAL: false
