# conf/distillation.yaml — owns: DISTILLATION. Every key here is optional; with
# ENABLED: false the trainer produces the numbers it produced before this block
# existed (document 14 section 14.1).

DISTILLATION:
  ENABLED: false

  # A teacher list is a project decision. ENABLED: true with an empty list is
  # refused at load, because a distilled run with no teacher is a typo.
  TEACHERS: []

  OUT_DIM: 64
  TAU: 0.07
  TAU_MIN: 0.01
  TAU_MAX: 1.0
  SYMMETRIC: true
  BETA: 1.0

  SIGMA_MODE: quality_softmax
  SIGMA_FLOOR: 0.0
  SIGMA_CAP: 1.0
  SIGMA_DECAY: 0.9

  LAMBDA_COOP: 0.0
  LAMBDA_IND: 1.0
  LAMBDA_REG: 0.0
  LAMBDA_CD: 1.0
  PEER_MARGIN: 0.25

  WARMUP_STEPS: 0
  RAMP_STEPS: 0

  REGULARIZERS: []
  VIEWS: {}
  STRICT: true

  CONFORMANCE:
    ENABLED: true
    MODE: hybrid
    TRUST_INIT: 1.0
    TRUST_FLOOR: 0.0
    TRUST_DECAY: 0.5
    TRUST_RESTORE: 0.05
    TRUST_MIN: 0.05
    STRICT_UNDECLARED: true
    AUDIT_ARTIFACTS: {}
    SIGMA_TRUST_MIN: true
