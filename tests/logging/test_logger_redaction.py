"""test logger redaction (split from test_logger_async_redaction_failure.py)."""

from __future__ import annotations

import time


from mactf import signals


from mactf.training.logging import JsonlSink, LogRecord, MACTFLogger


def test_default_and_custom_redaction(tmp_path):
    sink = JsonlSink(tmp_path / "r.jsonl")
    log = MACTFLogger(run_id="r", sinks=[sink], async_write=False)
    log.info("x", api_key="sk-123", HF_TOKEN="hf_abc", ok="fine")
    log.close()
    text = (tmp_path / "r.jsonl").read_text()
    assert "sk-123" not in text and "hf_abc" not in text and "fine" in text

    custom = MACTFLogger(run_id="r", sinks=[], async_write=False, redact=["ssn"])
    custom.info("x", ssn="111-22", other="y")
    fields = custom.tail(1)[0].fields
    assert fields["ssn"] == "***" and fields["other"] == "y"


def test_signals_produce_records_and_logger_never_sends():
    log = MACTFLogger(run_id="s", sinks=[], async_write=False).connect()
    received: list[str] = []
    signals.warning_issued.connect(lambda sender, **kw: received.append("warn"))
    before = len(received)
    signals.warning_issued.send("test", peer="p", reason="r")
    assert len(received) == before + 1
    kinds = [r.kind for r in log.tail(0)]
    assert "warning" in kinds
    signals.phase_started.send("t", phase="rollout")
    signals.phase_finished.send("t", phase="rollout")
    assert any(r.kind == "step" and r.fields.get("phase") == "rollout" for r in log.tail(0))
    # connecting twice must not have made the logger emit into signals
    assert not hasattr(log, "send")


def test_state_dict_roundtrip_restores_offset(tmp_path):
    sink = JsonlSink(tmp_path / "s.jsonl")
    log = MACTFLogger(run_id="s", sinks=[sink], async_write=False)
    log.info("a")
    state = log.state_dict()
    other = MACTFLogger(
        run_id="", sinks=[JsonlSink(tmp_path / "s2.jsonl")], async_write=False
    )
    other.load_state_dict(state)
    assert other.run_id == "s"
    assert other.sinks[0].offset == state["sinks"][0]["offset"]
    assert isinstance(LogRecord(kind="x").as_dict()["kind"], str)
    time.sleep(0)
