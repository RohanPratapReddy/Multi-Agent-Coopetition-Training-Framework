# conf/logger.yaml — owns: LOGGER. When this block is present it replaces
# TELEMETRY for the logger; TELEMETRY stays live for projects that never adopt it.

LOGGER:
  LEVEL: INFO
  RUNS_DIR: var/runs

  # Order matters: every record goes to each sink, in this order.
  # console | file | jsonl | tensorboard | wandb
  SINKS: [console, file]

  JSON: false
  LOG_FILE: var/logs/mactf.log
  JSONL_FILE: null

  ROTATE_BYTES: 10485760
  ROTATE_BACKUPS: 5

  BUFFER_SIZE: 1000
  ASYNC_WRITE: true
  QUEUE_SIZE: 8192

  # Empty means the seven defaults in mactf/training/logging.py:53. Naming
  # fragments REPLACES the list, so the shipped file spells them out: a project
  # that adds "session_cookie" should not also lose "authorization".
  REDACT:
    - token
    - secret
    - password
    - api_key
    - authorization
    - hf_token
    - wandb_key

  SCHEMA_VERSION: 1

  # Required when "tensorboard" is in SINKS.
  TENSORBOARD_DIR: null
  # Required when "wandb" is in SINKS.
  WANDB_PROJECT: null
