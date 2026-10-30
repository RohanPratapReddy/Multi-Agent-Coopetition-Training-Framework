# conf/api.yaml — owns: API and CORS. Read by mactf.orchestration.live and the
# mactf.contrib.api views.

API:
  ALLOW_ANONYMOUS: true

  # {"<token>": ["read", "run:write"]} or
  # {"<token>": {scopes: ["read"], name: "ci"}}. Scopes: read, run:write,
  # score:write, sandbox:exec, admin, worker.
  TOKENS: {}

  BASE_PATH: api/v1
  EVENT_BUFFER: 1024
  MAX_CONCURRENT_RUNS: 1
  CANCEL_GRACE_SECONDS: 10.0

  # Leave empty and use CORS.ORIGINS below; the validator refuses both.
  CORS_ORIGINS: []
  PRINCIPAL_RESOLVER: null
  IDEMPOTENCY_TTL_SECONDS: 86400

  LOG_TAIL_DEFAULT: 100
  LOG_TAIL_MAX: 2000
  HEALTH_HISTORY_LIMIT: 50
  PREVIEW_MAX_NODES: 64

  INFERENCE_MAX_WORKERS: 4
  INFERENCE_BUDGET_DEFAULT: null

CORS:
  ORIGINS: []
  ALLOW_CREDENTIALS: false
  ALLOW_HEADERS: [authorization, content-type, x-request-id]
  MAX_AGE: 600
