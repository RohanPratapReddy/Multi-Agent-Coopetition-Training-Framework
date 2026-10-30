# conf/telemetry.yaml — owns: TELEMETRY. Read by MACTFLogger._default_sinks when
# LOGGER is absent. Keep this file if the project predates conf/logger.yaml.

TELEMETRY:
  LEVEL: INFO
  JSON: false
  CONSOLE: true
  LOG_FILE: var/logs/mactf.log
