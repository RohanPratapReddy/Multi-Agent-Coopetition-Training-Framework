from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mactf.exceptions import ImproperlyConfigured
from mactf.training.logging import (
    ConsoleSink,
    FileSink,
    JsonlSink,
    LogRecord,
    MACTFLogger,
    MemorySink,
)


def _logger(**kwargs):
    kwargs.setdefault("async_write", False)
    kwargs.setdefault("sinks", [])
    kwargs.setdefault("run_id", "run-1")
    return MACTFLogger(**kwargs)


def test_every_record_kind_is_emitted_with_schema_version():
    log = _logger()
    log.info("hello", a=1)
    log.epoch(SimpleNamespace(epoch=3, steps=2, mean_loss=0.5, completed=True, reason=""))
    log.step("peer-1", SimpleNamespace(loss_value=0.25, stepped=True, metrics={"grad_norm": 1.5}))
    log.loss({"total": 1.0, "policy": 0.5})
    log.aggregation({"kappa": 2.0})
    log.checkpoint(SimpleNamespace(path="a/ckpt-1.pt", episode=1), reason="periodic")
    log.health({"status": "critical", "epoch": 4})
    log.resume({"episode": 5})
    kinds = [record.kind for record in log.tail(0)]
    assert kinds == [
        "log", "epoch", "step", "loss", "aggregation", "checkpoint", "health", "resume",
    ] or kinds[1:] == ["epoch", "step", "loss", "aggregation", "checkpoint", "health", "resume"]
    for record in log.tail(0):
        assert record.schema_version == 1
        assert record.run_id == "run-1"
        json.loads(record.to_json())


def test_node_path_defaults_to_empty_and_step_carries_path():
    log = _logger()
    log.info("x")
    log.step("peer-1/child-a", SimpleNamespace(loss_value=1.0, stepped=True, metrics={}))
    first, second = log.tail(0)[-2:]
    assert first.node_path == ""
    assert second.node_path == "peer-1/child-a"
    assert second.fields["loss"] == 1.0


def test_health_level_follows_status():
    log = _logger()
    log.health({"status": "degraded"})
    log.health({"status": "critical"})
    log.health({"status": "healthy"})
    assert [r.level for r in log.tail(0)] == ["warning", "error", "info"]


def test_level_filter_and_unknown_level():
    log = _logger(level="WARNING")
    log.info("dropped")
    log.warning("kept")
    assert [r.message for r in log.tail(0)] == ["kept"]
    with pytest.raises(ImproperlyConfigured):
        _logger(level="LOUD")


def test_tail_filters_by_kind_and_node():
    log = _logger()
    log.step("a", SimpleNamespace(loss_value=1.0, stepped=True, metrics={}))
    log.step("b", SimpleNamespace(loss_value=2.0, stepped=True, metrics={}))
    log.info("noise")
    assert len(log.tail(10, kind="step")) == 2
    assert [r.node_path for r in log.tail(10, node_path="b")] == ["b"]
    assert len(log.tail(1)) == 1


def test_tensor_fields_are_jsonable():
    import torch

    log = _logger()
    log.info("t", small=torch.zeros(2), big=torch.zeros(100))
    data = json.loads(log.tail(1)[0].to_json())
    assert data["fields"]["small"]["values"] == [0.0, 0.0]
    assert "values" not in data["fields"]["big"]


def test_sinks_write_and_parse_back(tmp_path):
    jsonl = JsonlSink(tmp_path / "run.jsonl")
    plain = FileSink(tmp_path / "run.log")
    log = _logger(sinks=[jsonl, plain])
    for i in range(5):
        log.info(f"m{i}", i=i)
    log.flush()
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    assert [json.loads(line)["message"] for line in lines] == [f"m{i}" for i in range(5)]
    assert "m4" in (tmp_path / "run.log").read_text()
    assert jsonl.offset == (tmp_path / "run.jsonl").stat().st_size
    log.close()
    assert jsonl.closed and plain.closed


def test_file_sink_rotates(tmp_path):
    sink = FileSink(tmp_path / "r.log", max_bytes=200, backup_count=2)
    log = _logger(sinks=[sink])
    for i in range(30):
        log.info("x" * 40, i=i)
    log.close()
    assert (tmp_path / "r.log.1").exists()
    assert not (tmp_path / "r.log.3").exists()
    assert (tmp_path / "r.log").stat().st_size <= 400


def test_console_sink_writes_to_given_stream(capsys):
    import io

    stream = io.StringIO()
    log = _logger(sinks=[ConsoleSink(stream=stream, json_format=True)])
    log.info("hi")
    assert json.loads(stream.getvalue().splitlines()[0])["message"] == "hi"


def test_memory_sink_is_always_present_and_bounded():
    log = _logger(buffer_size=3)
    for i in range(10):
        log.info(str(i))
    assert [r.message for r in log.tail(0)] == ["7", "8", "9"]
    assert any(isinstance(s, MemorySink) for s in log.sinks)


def test_child_shares_buffer_and_suffixes_run_id():
    log = _logger()
    child = log.child("w1")
    child.info("from child")
    assert child.run_id == "run-1.w1"
    assert log.tail(1)[0].run_id == "run-1.w1"


def test_logrecord_matches():
    record = LogRecord(kind="step", node_path="p")
    assert record.matches(kind="step", node_path="p")
    assert not record.matches(kind="loss")
