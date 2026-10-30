# conf/scorer.yaml — owns: SCORING, SEVERITY, WARNINGS, ENGINE.

SCORING:
  # framework.md 3.2: only the controlling tier scores. "layered" scores every
  # tier but still refuses to net across tiers.
  PRECEDENCE_MODE: strict

  # R = a*r_task + b*r_epi + g*r_collab + d*r_align
  REWARD_WEIGHTS:
    r_task: 1.0
    r_epi: 1.5
    r_collab: 1.2
    r_align: 1.0

  # P = sum(lambda_k * p_k) * Omega
  PENALTY_LAMBDAS:
    P1: 5.0
    P2: 5.0
    P3: 3.0
    P4: 0.5
    P5: 2.5
    P6: 1.0
    P7: 1.0
    P8: 0.75
    P9: 2.0
    P10: 2.0
    P11: 0.5
    P12: 0.5
    P13: 8.0
    P14: 8.0
    P15: 1.5
    P16: 3.0
    P17: 50.0
    P18: 1.5
    P19: 1.5
    P20: 1000.0
    P21: 50.0
    P22: 50.0
    P23: 2.0
  DEFAULT_LAMBDA: 1.0

  SANDBOX_MULTIPLIER: 1000.0
  OMEGA_RULES: [P20, P21, P22]
  OMEGA_RULE_OVERRIDES: {}

  # Removing any of these would silently permit netting across tiers. The
  # validator refuses a list that drops one.
  NON_OFFSETTABLE: [P1, P2, P17, P20, P21, P22]
  ALLOW_NEGATIVE_REWARDS: false

  F_CAP_RATIO: 0.5
  P_MIN: 0.5
  NOISE_THRESHOLD: 0.2

  # Doc 11: the 0-100 points economy. Reward weights sum to 100.
  POINTS:
    TABLE_VERSION: "1.0.0"
    WARNING_NOTIONAL: 2.0
    REWARD_WEIGHTS:
      R1: 25.0
      R2: 12.0
      R3: 2.0
      R4: 3.0
      R5: 3.0
      R6: 5.0
      R7: 5.0
      R8: 6.0
      R9: 3.0
      R10: 8.0
      R11: 5.0
      R12: 4.0
      R13: 6.0
      R14: 5.0
      R15: 3.0
      R16: 4.0
      R17: 1.0
    PENALTY_WEIGHTS:
      P1: 40.0
      P2: 40.0
      P3: 20.0
      P4: 5.0
      P5: 25.0
      P6: 6.0
      P7: 6.0
      P8: 5.0
      P9: 15.0
      P10: 15.0
      P11: 4.0
      P12: 4.0
      P13: 60.0
      P14: 60.0
      P15: 10.0
      P16: 25.0
      P17: 100.0
      P18: 10.0
      P19: 10.0
      P20: 100.0
      P21: 100.0
      P22: 100.0
      P23: 12.0
    SEVERITY_FRACTIONS:
      S0: 0.0
      S1: 0.05
      S2: 0.15
      S3: 0.4
      S4: 1.0

SEVERITY:
  # Must be strictly increasing S0..S4; the validator checks it.
  WEIGHTS:
    S0: 0.0
    S1: 0.25
    S2: 1.0
    S3: 4.0
    S4: 25.0
  MAX_TIER_ESCALATION: 3

WARNINGS:
  # K_ESC consecutive episodes escalate a warning.
  K_ESC: 3
  # N_WIN episodes form a window; N_DECAY episodes of clean behaviour decay it.
  N_WIN: 5
  N_DECAY: 3
  # W_CAP warnings in the window; must be <= K_ESC.
  W_CAP: 2
  P_WARN: 5.0
  ESCALATE: true

ENGINE:
  F_CAP: 3
  P_MIN: 0.5
  P12_NOISE_RATIO: 0.5
  R9_STEP_FACTOR: 1.25
