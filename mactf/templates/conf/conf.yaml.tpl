# conf/conf.yaml — owns: PATHS, DEBUG, DEVICE, DTYPE, AMP, SEED,
# DETERMINISTIC, DEFAULT_AUTO_FIELD. Loaded first; every other file's
# relative paths resolve against PATHS.ROOT.

DEBUG: false

# auto | cpu | cuda | cuda:1 | mps
DEVICE: auto
# auto | float32 | float16 | bfloat16 | float64
DTYPE: auto
# CUDA only; ignored elsewhere by mactf.runtime.
AMP: true

SEED: 1337
# true makes torch use deterministic kernels. Slower, and the only way to make a
# bitwise comparison across two runs meaningful.
DETERMINISTIC: false

DEFAULT_AUTO_FIELD: mactf.db.BigAutoField

PATHS:
  # "." means the project package directory, which is where settings.BASE_DIR
  # points. Leave it alone unless the project runs from outside its package.
  ROOT: "."
  VAR: var
  CHECKPOINTS: var/checkpoints
  CONFIGS: conf
  DATASETS: datasets
  LOGS: var/logs
  IMAGES: var/images
  RUNS: var/runs
  TEACHERS: var/teachers
  AUDITS: var/audits
  CACHE: var/cache
  STATE: var/state
