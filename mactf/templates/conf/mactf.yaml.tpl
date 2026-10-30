# conf/mactf.yaml — owns: HTTP, and records INSTALLED_APPS, MIDDLEWARE,
# ROOT_URLCONF, APPEND_SLASH for a reader. The apps list below is documentation:
# apps are imported before configuration is read, so settings.py is the source of
# truth for INSTALLED_APPS. A conf/ file that sets it is refused (reserved key).

# --- documented, not authoritative -----------------------------------------
# INSTALLED_APPS:
#   - mactf.contrib.scoring
#   - mactf.contrib.coopetition
#   - mactf.contrib.sandbox
#   - mactf.contrib.telemetry
#   - mactf.contrib.components
#   - "{{ project_name }}.peers"
#
# MIDDLEWARE:
#   - mactf.http.middleware.RequestIdMiddleware
#   - mactf.http.middleware.ExceptionMiddleware
#
# ROOT_URLCONF: null
# APPEND_SLASH: true

HTTP:
  # 8 MiB. The largest accepted request body; a score submission with 64 peer
  # records is under 200 KiB, so this is headroom, not a target.
  MAX_BODY_BYTES: 8388608
  # Worker threads for streaming responses.
  STREAM_WORKERS: 16
  # Seconds between SSE keepalives. Must be below the shortest proxy timeout in
  # front of the service.
  SSE_HEARTBEAT: 15.0

# Extra overlays, merged after this directory. A path outside the project is
# allowed; a path that does not exist is refused at load.
CONFIG_FILES: []
