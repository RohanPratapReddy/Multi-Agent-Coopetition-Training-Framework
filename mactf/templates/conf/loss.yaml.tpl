# conf/loss.yaml — owns: LOSS. These six coefficients are LossConfig's fields
# (mactf/rl/loss.py:53). LOSS wins over TRAINING as a block, not per key.

LOSS:
  # composite uses CompositeLoss over the coefficients below. Any other value is
  # a mactf.rl.functions registry name, and ARGS is passed to it.
  TYPE: composite

  VALUE_COEF: 0.5
  KL_COEF: 0.01
  ENTROPY_COEF: 0.0
  CLIP_EPSILON: 0.2
  COMPLIANCE_COEF: 1.0
  ADVANTAGE_NORMALIZATION: true

  ARGS: {}
