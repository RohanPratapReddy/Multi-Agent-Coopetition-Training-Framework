"""test logger async (split from test_logger_async_redaction_failure.py)."""

from __future__ import annotations

import threading


from mactf.training.logging import JsonlSink, LogRecord, MACTFLogger


class Slow:
    name = "slow"

    def __init__(self, gate: threading.Event):
        self.gate = gate
        self.seen: list[str] = []

    def write(self, record):
        self.gate.wait(5)
        self.seen.append(record.message)

    def flush(self):
        pass

    def close(self):
        pass

    def state_dict(self):
        return {"name": self.name}


def test_async_preserves_order_and_flush_drains():
    log = MACTFLogger(run_id="a", sinks=[], async_write=True)
    for i in range(200):
        log.info(str(i))
    log.flush()
    assert [r.message for r in log.tail(0)] == [str(i) for i in range(200)]
    log.close()


def test_close_drains_and_stops_thread():
    log = MACTFLogger(run_id="a", sinks=[], async_write=True)
    for i in range(50):
        log.info(str(i))
    log.close()
    assert len(log.memory.records) == 50
    assert not any(t.name.startswith("mactf-log-a") and t.is_alive() for t in threading.enumerate())
    log.info("after close")
    assert len(log.memory.records) == 50


def test_overflow_drops_non_errors_but_never_errors():
    gate = threading.Event()
    slow = Slow(gate)
    log = MACTFLogger(run_id="o", sinks=[slow], async_write=True, queue_size=2)
    for i in range(50):
        log.info(f"i{i}")
    assert log.dropped_records > 0
    # an error must block until space exists rather than drop
    threading.Timer(0.2, gate.set).start()
    log.error("must-survive")
    log.close()
    assert "must-survive" in slow.seen
