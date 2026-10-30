# conf/trainer.yaml — owns: TRAINING and TRAINER. TRAINER is folded into TRAINING
# at load (section 5.2), so a consumer reads one block.

TRAINER:
  EPOCHS: 3
  BATCH_SIZE: 32
  STEPS_PER_EPOCH: 4
  # Accumulate gradients over N steps before one optimizer step.
  GRAD_ACCUMULATION: 1
  MAX_GRAD_NORM: 1.0

  # A flat run: every peer trains independently and kappa-softmax blends.
  HIERARCHY: false
  PEER_STRATEGY: kappa_softmax
  WEIGHT_UPDATE: kappa_softmax
  # independent | shared_trunk | tied_weights. Siblings are averaged, so they must
  # not share parameters; shared_trunk shares a frozen trunk only.
  SHARING: independent

  TASK: synthetic_classification
  CHECKPOINT_EVERY: 1
  LOG_EVERY_STEPS: 1
  WATCHDOG_EVERY_EPOCHS: 1

# The model's input and output widths. They must match every peer's kwargs in
# peers.yaml; the trainer refuses a mismatch when it binds optimizers.
TRAINING:
  STATE_DIM: 16
  ACTION_DIM: 4
  LEARNING_RATE: 0.001

  GAMMA: 0.99
  GAE_LAMBDA: 0.95
  CRITIQUE_ROUNDS: 1

  # One bootstrap pass before the real loop: the reference policy is trained on a
  # small slice so the first contrast is not against random weights.
  BOOTSTRAP: true
  BOOTSTRAP_CONFIG:
    EPOCHS: 1
    BATCH_SIZE: 16
    LEARNING_RATE: 0.001
    WEIGHT_DECAY: 0.0
    MAX_GRAD_NORM: 1.0
    UPDATE_REFERENCE: true
    PATIENCE: 0
    BATCHES_PER_EPOCH: 0

  REDTEAM: false

  # Stop the run when the entropy ratio falls below this, after a snapshot.
  ROLLBACK_ON_ENTROPY_COLLAPSE: true
  ENTROPY_COLLAPSE_RATIO: 0.05
