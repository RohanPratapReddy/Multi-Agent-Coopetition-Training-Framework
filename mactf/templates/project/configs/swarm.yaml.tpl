# Swarm composition and training-loop overlays.
# Merged over {{ project_name }}/settings.py at load time; only include what you change.

TRAINING:
  EPOCHS: 3
  BATCH_SIZE: 32
  STEPS_PER_EPOCH: 4
  LEARNING_RATE: 0.001
  CRITIQUE_ROUNDS: 1
  CHECKPOINT_EVERY: 1

COOPETITION:
  # docx Turn 5 adaptive temperature
  KAPPA_INIT: 1.0
  KAPPA_MIN: 0.1
  KAPPA_MAX: 10.0
  TARGET_ENTROPY_RATIO: 0.8
  BETA_MOMENTUM: 0.9
  ETA_KAPPA: 0.05
  # framework.md section 10 constraints
  WEIGHT_FLOOR: 0.02
  WEIGHT_CAP: 0.6
  DISAGREEMENT_MAX: 0.75
  REFERENCE_EMA: 0.05
