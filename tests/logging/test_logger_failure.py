"""test logger failure (split from test_logger_async_redaction_failure.py)."""

from __future__ import annotations

from mactf.training.logging import JsonlSink, LogRecord, MACTFLogger


class Boom:
    name = "boom"

    def write(self, record):
        raise RuntimeError("disk full")

    def flush(self):
        raise RuntimeError("flush fail")

    def close(self):
        raise RuntimeError("close fail")

    def state_dict(self):
        return {"name": self.name}


def test_raising_sink_does_not_fail_and_is_reported():
    log = MACTFLogger(run_id="f", sinks=[Boom()], async_write=False)
    log.info("one")
    log.info("two")
    log.flush()
    log.close()
    assert any("disk full" in e for e in log.sink_errors)
    assert log.dropped_records >= 2
    assert [r.message for r in log.tail(0)] == ["one", "two"]
    assert log.describe()["sink_errors"]
