"""Dataset and task loading.

``mactf.data`` is the only place that reads a dataset file. Everything upstream
(the orchestrator, the bootstrap trainer, the evaluators) receives a task source
that satisfies :class:`TaskSourceProtocol` or a validated
:class:`~mactf.data.codec.TaskRecord`, so no other module needs to know a file
format exists.

The historical ``mactf.orchestration.episode.TaskSource`` remains as a
compatibility shim over :class:`SyntheticTaskSource`; it is not re-exported here
to keep this package free of orchestration imports at module load time.
"""

from __future__ import annotations

from .codec import TaskCodec, TaskRecord
from .registry import (
    available_sources,
    build_source,
    register_source,
    source_names,
    unregister_source,
)
from .sampling import batch_slices, content_hash, epoch_indices, permutation, seed_from
from .sources import (
    JsonlTaskSource,
    MixedTaskSource,
    SyntheticTaskSource,
    TaskSourceProtocol,
    load_records,
)

__all__ = [
    "JsonlTaskSource",
    "MixedTaskSource",
    "SyntheticTaskSource",
    "TaskCodec",
    "TaskRecord",
    "TaskSourceProtocol",
    "available_sources",
    "batch_slices",
    "build_source",
    "content_hash",
    "epoch_indices",
    "load_records",
    "permutation",
    "register_source",
    "seed_from",
    "source_names",
    "unregister_source",
]
