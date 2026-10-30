# conf/checkpoint.yaml — owns: CHECKPOINT. Mirrors MACTFCheckpoint.__init__
# (mactf/orchestration/checkpoint_v2.py:395).

CHECKPOINT:
  # null resolves to PATHS["CHECKPOINTS"]. Set an absolute path to put a run's
  # checkpoints on a different volume, which is what a large swarm does.
  DIR: null

  KEEP: 5
  KEEP_BEST: 1

  COMPRESS: true
  # sha256 | blake2b
  CHECKSUM: sha256
  ASYNC_WRITE: false

  # These five decide what a checkpoint can restore. Turning one off makes a
  # resume lossy in a way that is not detectable afterwards, so each is explicit.
  INCLUDE_OPTIMIZER: true
  INCLUDE_RNG: true
  INCLUDE_DATA_POSITION: true
  INCLUDE_BUFFERS: true
  INCLUDE_ADAPTERS: true
