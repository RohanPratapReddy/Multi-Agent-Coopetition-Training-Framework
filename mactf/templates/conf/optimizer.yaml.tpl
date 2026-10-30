# conf/optimizer.yaml — owns: OPTIMIZER. Adapter is mactf.training.optimizer:
# from_spec accepts torch, scipy, and callable backends.

OPTIMIZER:
  # torch | scipy | callable
  TYPE: torch
  # A class on torch.optim. Unknown names are refused at construction with the
  # accepted set, never at epoch 4.
  CLASS: AdamW
  LR: 0.001
  WEIGHT_DECAY: 0.01
  # Adam/AdamW only. Set via BETAS so the mapping stays a plain YAML list.
  BETAS: [0.9, 0.999]
  EPS: 1.0e-8

  GRAD_ACCUMULATION: 1
  MAX_GRAD_NORM: 1.0

  # A scheduler is named, not constructed from YAML. Supported values:
  # null | step | cosine | exponential | plateau
  SCHEDULER: null
  SCHEDULER_KWARGS: {}
