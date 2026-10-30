# conf/optimizer.yaml — owns: OPTIMIZER. Adapter is mactf.training.optimizer:
# from_spec accepts torch, scipy and callable, plus any optimizer registered with
# @register_optimizer (and the built-in null and router).

OPTIMIZER:
  # torch | scipy | callable | null | router | a registered optimizer name
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

  # Per-node optimizers (document 17 section 9.4): a node-path glob maps to an OPTIMIZER
  # block. An override inherits LR, WEIGHT_DECAY, GRAD_ACCUMULATION, MAX_GRAD_NORM and
  # SCHEDULER one by one; setting TYPE or CLASS replaces the default's backend keys.
  OVERRIDES: {}
  # What a live swap does with the old optimizer's state: reset | carry | fail.
  MIGRATE: reset
  # Probe optimizers before use: custom (non-built-in only) | all | off.
  VERIFY: custom
  # An override that matches no node is an error (E050) when true, a warning when false.
  STRICT_GLOBS: true
