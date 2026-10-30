# conf/distributed.yaml — owns: DISTRIBUTED. Read only when mactf.distributed is
# installed; harmless otherwise.

DISTRIBUTED:
  COORDINATOR_URL: http://127.0.0.1:8000
  BASE_PATH: worker/v1

  # null derives an id from the hostname. Set it when several workers share a
  # host, which is the only case where the derivation collides.
  WORKER_ID: null
  MAX_PEERS: 1

  COMPRESS: true
  WORKER_HEARTBEAT_TIMEOUT: 30.0
  WORKER_TASK_TIMEOUT: 600.0
  MAX_WORKERS: 16
