"""Document 15 section 7: every block a ``conf/`` file writes has a real reader.

Each test builds the consumer from a settings object, the way the framework does,
and asserts the value that the YAML named reached the object that acts on it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from mactf.conf import DEFAULTS, ImproperlyConfigured, Settings
from mactf.exceptions import ValidationError
from mactf.orchestration.checkpoint_v2 import MACTFCheckpoint
from mactf.training.logging import (
    DEFAULT_REDACT,
    ConsoleSink,
    FileSink,
    JsonlSink,
    MACTFLogger,
    MemorySink,
)
from mactf.training.optimizer import MACTFOptimizer, ScipyOptimizer, TorchOptimizer, optimizer_spec
from mactf.training.watchdog import MACTFWatchdog, WatchdogAction


def make_settings(root: Path, **blocks: dict[str, Any]) -> Settings:
    """Settings over ``DEFAULTS`` rooted at ``root`` with ``blocks`` merged per key."""
    values = copy.deepcopy(DEFAULTS)
    values["PATHS"]["ROOT"] = str(root)
    for name, override in blocks.items():
        values.setdefault(name, {}).update(override)
    return Settings(values)


# --------------------------------------------------------------------------- logger
class TestLoggerFromSettings:
    def test_default_sinks_are_console_file_and_one_memory(self, tmp_path):
        logger = MACTFLogger.from_settings(make_settings(tmp_path), run_id="r1")
        try:
            kinds = [type(sink) for sink in logger.sinks]
            assert kinds == [ConsoleSink, FileSink, MemorySink]
        finally:
            logger.close()

    def test_sinks_are_built_in_the_order_the_file_names_them(self, tmp_path):
        settings = make_settings(
            tmp_path,
            LOGGER={
                "SINKS": ["jsonl", "console"],
                "JSONL_FILE": "var/logs/events.jsonl",
                "ASYNC_WRITE": False,
            },
        )
        logger = MACTFLogger.from_settings(settings, run_id="r1")
        try:
            assert [type(sink) for sink in logger.sinks] == [JsonlSink, ConsoleSink, MemorySink]
        finally:
            logger.close()

    def test_memory_sink_present_exactly_once_even_when_named(self, tmp_path):
        settings = make_settings(
            tmp_path, LOGGER={"SINKS": ["console", "memory"], "ASYNC_WRITE": False}
        )
        with pytest.raises(ImproperlyConfigured, match="unknown LOGGER sink"):
            # ``memory`` is added by the logger itself, never listed in the file.
            MACTFLogger.from_settings(settings, run_id="r1")

    def test_memory_sink_added_when_not_listed(self, tmp_path):
        settings = make_settings(tmp_path, LOGGER={"SINKS": ["console"], "ASYNC_WRITE": False})
        logger = MACTFLogger.from_settings(settings, run_id="r1")
        try:
            assert sum(isinstance(sink, MemorySink) for sink in logger.sinks) == 1
        finally:
            logger.close()

    def test_relative_log_file_resolves_against_project_root(self, tmp_path):
        settings = make_settings(
            tmp_path,
            LOGGER={"SINKS": ["file"], "LOG_FILE": "var/logs/custom.log", "ASYNC_WRITE": False},
        )
        logger = MACTFLogger.from_settings(settings, run_id="r1")
        try:
            sink = next(item for item in logger.sinks if isinstance(item, FileSink))
            assert Path(sink.path) == tmp_path / "var" / "logs" / "custom.log"
        finally:
            logger.close()

    def test_level_and_buffer_size_come_from_the_block(self, tmp_path):
        settings = make_settings(
            tmp_path,
            LOGGER={"LEVEL": "debug", "BUFFER_SIZE": 7, "SINKS": ["console"], "ASYNC_WRITE": False},
        )
        logger = MACTFLogger.from_settings(settings, run_id="r1")
        try:
            assert logger.level == "DEBUG"
            assert logger.buffer_size == 7
        finally:
            logger.close()

    def test_redact_fragments_are_lowered_and_an_empty_list_keeps_the_defaults(self, tmp_path):
        custom = make_settings(
            tmp_path,
            LOGGER={"REDACT": ["Password", "TOKEN"], "SINKS": ["console"], "ASYNC_WRITE": False},
        )
        logger = MACTFLogger.from_settings(custom, run_id="r1")
        try:
            assert logger.redact == ("password", "token")
        finally:
            logger.close()

        default = make_settings(
            tmp_path, LOGGER={"SINKS": ["console"], "ASYNC_WRITE": False}
        )
        logger = MACTFLogger.from_settings(default, run_id="r1")
        try:
            assert logger.redact == tuple(DEFAULT_REDACT)
        finally:
            logger.close()

    def test_unknown_sink_is_refused_with_its_name(self, tmp_path):
        settings = make_settings(tmp_path, LOGGER={"SINKS": ["syslog"], "ASYNC_WRITE": False})
        with pytest.raises(ImproperlyConfigured, match="syslog"):
            MACTFLogger.from_settings(settings, run_id="r1")

    @pytest.mark.parametrize(
        ("sink", "missing_key"),
        [("jsonl", "JSONL_FILE"), ("tensorboard", "TENSORBOARD_DIR")],
    )
    def test_sink_without_its_path_names_the_missing_key(self, tmp_path, sink, missing_key):
        settings = make_settings(tmp_path, LOGGER={"SINKS": [sink], "ASYNC_WRITE": False})
        with pytest.raises(ImproperlyConfigured, match=missing_key):
            MACTFLogger.from_settings(settings, run_id="r1")

    def test_bad_level_is_refused(self, tmp_path):
        settings = make_settings(
            tmp_path, LOGGER={"LEVEL": "LOUD", "SINKS": ["console"], "ASYNC_WRITE": False}
        )
        with pytest.raises(ImproperlyConfigured, match="unknown log level"):
            MACTFLogger.from_settings(settings, run_id="r1")

    def test_a_project_without_logger_keeps_the_telemetry_path(self, tmp_path):
        """No ``LOGGER`` authored: ``TELEMETRY.CONSOLE: false`` still drops the console."""
        values = {
            "PATHS": {"ROOT": str(tmp_path)},
            "TELEMETRY": {"CONSOLE": False, "LEVEL": "WARNING"},
        }
        logger = MACTFLogger(values, run_id="r1")
        try:
            assert logger.level == "WARNING"
            assert not any(isinstance(sink, ConsoleSink) for sink in logger.sinks)
        finally:
            logger.close()


# ------------------------------------------------------------------------ checkpoint
class TestCheckpointFromSettings:
    def test_null_dir_resolves_through_paths(self, tmp_path):
        checkpoint = MACTFCheckpoint.from_settings(make_settings(tmp_path), run_id="r1")
        assert checkpoint.directory == tmp_path / "var" / "checkpoints"
        assert checkpoint.directory.is_dir()

    def test_changed_paths_checkpoints_moves_the_default(self, tmp_path):
        values = copy.deepcopy(DEFAULTS)
        values["PATHS"]["ROOT"] = str(tmp_path)
        values["PATHS"]["CHECKPOINTS"] = "elsewhere/ckpt"
        checkpoint = MACTFCheckpoint.from_settings(Settings(values), run_id="r1")
        assert checkpoint.directory == tmp_path / "elsewhere" / "ckpt"

    def test_explicit_block_dir_wins_over_paths(self, tmp_path):
        settings = make_settings(tmp_path, CHECKPOINT={"DIR": "named/dir"})
        checkpoint = MACTFCheckpoint.from_settings(settings, run_id="r1")
        assert checkpoint.directory == tmp_path / "named" / "dir"

    def test_explicit_argument_wins_over_the_block(self, tmp_path):
        settings = make_settings(tmp_path, CHECKPOINT={"DIR": "named/dir"})
        target = tmp_path / "argument"
        checkpoint = MACTFCheckpoint.from_settings(settings, run_id="r1", directory=target)
        assert checkpoint.directory == target

    def test_retention_and_content_keys_reach_the_writer(self, tmp_path):
        settings = make_settings(
            tmp_path,
            CHECKPOINT={
                "KEEP": 2,
                "KEEP_BEST": 0,
                "COMPRESS": False,
                "CHECKSUM": "SHA256",
                "ASYNC_WRITE": True,
                "INCLUDE_OPTIMIZER": False,
                "INCLUDE_RNG": False,
                "INCLUDE_DATA_POSITION": False,
                "INCLUDE_BUFFERS": False,
                "INCLUDE_ADAPTERS": False,
            },
        )
        checkpoint = MACTFCheckpoint.from_settings(settings, run_id="r1")
        assert (checkpoint.keep, checkpoint.keep_best) == (2, 0)
        assert checkpoint.compress is False
        assert checkpoint.checksum == "sha256"
        assert checkpoint.async_write is True
        assert not any(
            (
                checkpoint.include_optimizer,
                checkpoint.include_rng,
                checkpoint.include_data_position,
                checkpoint.include_buffers,
                checkpoint.include_adapters,
            )
        )

    def test_unknown_checksum_is_refused(self, tmp_path):
        settings = make_settings(tmp_path, CHECKPOINT={"CHECKSUM": "md-none"})
        with pytest.raises(Exception, match="(?i)checksum"):
            MACTFCheckpoint.from_settings(settings, run_id="r1")

    def test_mapping_without_a_path_resolver_needs_an_explicit_directory(self):
        with pytest.raises(ImproperlyConfigured, match="CHECKPOINT.DIR"):
            MACTFCheckpoint.from_settings({"CHECKPOINT": {}}, run_id="r1")


# -------------------------------------------------------------------------- watchdog
class _Sink:
    """A stand-in checkpoint/logger: the factory only stores them."""


class TestWatchdogFromSettings:
    def build(self, **block: Any) -> MACTFWatchdog:
        values = {"WATCHDOG": block}
        return MACTFWatchdog.from_settings(values, checkpoint=_Sink(), logger=_Sink())

    def test_defaults_match_the_shipped_block(self):
        watchdog = MACTFWatchdog.from_settings(
            copy.deepcopy(DEFAULTS), checkpoint=_Sink(), logger=_Sink()
        )
        assert watchdog.policy.max_grad_norm == 100.0
        assert watchdog.policy.loss_increase_ratio == 3.0
        assert watchdog.snapshot_every_epochs == 1
        assert watchdog.snapshot_every_seconds is None
        assert watchdog.keep_history == 512
        assert watchdog.on_degradation is WatchdogAction.SNAPSHOT
        assert watchdog.on_critical is WatchdogAction.ROLLBACK

    def test_policy_fields_reach_the_policy(self):
        watchdog = self.build(MAX_GRAD_NORM=5.0, LOSS_PATIENCE=9, NAN_POLICY="ignore")
        assert watchdog.policy.max_grad_norm == 5.0
        assert watchdog.policy.loss_patience == 9
        assert watchdog.policy.nan_policy == "ignore"

    def test_cadence_keys_reach_the_constructor(self):
        watchdog = self.build(
            SNAPSHOT_EVERY_EPOCHS=4,
            SNAPSHOT_EVERY_SECONDS=30.0,
            SNAPSHOT_ON_DEGRADATION=False,
            KEEP_HISTORY=3,
        )
        assert watchdog.snapshot_every_epochs == 4
        assert watchdog.snapshot_every_seconds == 30.0
        assert watchdog.snapshot_on_degradation is False
        assert watchdog.keep_history == 3

    @pytest.mark.parametrize("action", ["abort", "rollback"])
    def test_on_critical_is_lowered_into_an_action(self, action):
        watchdog = self.build(ON_CRITICAL=action.upper())
        assert watchdog.on_critical is WatchdogAction(action)

    def test_unknown_key_is_refused_by_name(self):
        with pytest.raises(ImproperlyConfigured, match="max_grad_normn"):
            self.build(MAX_GRAD_NORMN=100)

    def test_unknown_action_is_refused(self):
        with pytest.raises(ImproperlyConfigured, match="explode"):
            self.build(ON_CRITICAL="explode")

    def test_missing_logger_or_checkpoint_is_refused(self):
        with pytest.raises(ImproperlyConfigured):
            MACTFWatchdog.from_settings({"WATCHDOG": {}}, checkpoint=None, logger=_Sink())
        with pytest.raises(ImproperlyConfigured):
            MACTFWatchdog.from_settings({"WATCHDOG": {}}, checkpoint=_Sink(), logger=None)


# ------------------------------------------------------------------------- optimizer
class TestOptimizerFromBlock:
    def test_shipped_block_builds_adamw(self):
        optimizer = MACTFOptimizer.from_block(copy.deepcopy(DEFAULTS["OPTIMIZER"]))
        assert isinstance(optimizer.backend, TorchOptimizer)
        assert optimizer.backend._class_name == "AdamW"
        assert optimizer.backend.learning_rate == 0.001
        assert optimizer.backend.weight_decay == 0.01
        assert optimizer.max_grad_norm == 1.0
        assert optimizer.grad_accumulation == 1

    def test_betas_become_a_tuple_of_floats(self):
        optimizer = MACTFOptimizer.from_block({"CLASS": "Adam", "LR": 0.01, "BETAS": [1, 0.5]})
        betas = optimizer.backend.kwargs["betas"]
        assert betas == (1.0, 0.5)
        assert isinstance(betas, tuple)
        assert all(isinstance(item, float) for item in betas)

    def test_beta0_and_beta1_build_the_pair(self):
        spec = optimizer_spec({"CLASS": "Adam", "BETA0": 0.7, "BETA1": 0.8})
        assert spec["betas"] == (0.7, 0.8)
        assert "beta0" not in spec and "beta1" not in spec

    def test_null_lr_is_stripped_not_passed_as_none(self):
        """``LR: null`` means "the trainer supplies it" (``TRAINING.LEARNING_RATE``)."""
        spec = optimizer_spec({"CLASS": "Adam", "LR": None})
        assert "lr" not in spec
        # With no LR anywhere the backend refuses, naming the problem, rather than
        # constructing an optimizer that would step at a learning rate of zero.
        with pytest.raises(ValidationError, match="lr > 0"):
            MACTFOptimizer.from_block({"CLASS": "Adam", "LR": None})
    def test_scheduler_keys_never_reach_the_torch_optimizer(self):
        spec = optimizer_spec(
            {"CLASS": "SGD", "LR": 0.1, "SCHEDULER": "StepLR", "SCHEDULER_KWARGS": {"step_size": 2}}
        )
        assert "scheduler" not in spec and "scheduler_kwargs" not in spec
        optimizer = MACTFOptimizer.from_block(
            {"CLASS": "SGD", "LR": 0.1, "SCHEDULER": "StepLR", "SCHEDULER_KWARGS": {"step_size": 2}}
        )
        assert optimizer._pending_scheduler == ("StepLR", {"step_size": 2})

    def test_adam_defaults_are_dropped_for_a_class_that_rejects_them(self):
        """``CLASS: SGD`` in a file that still lists Adam's BETAS and EPS must build."""
        optimizer = MACTFOptimizer.from_block(
            {"CLASS": "SGD", "LR": 0.1, "BETAS": [0.9, 0.999], "EPS": 1e-08}
        )
        assert optimizer.backend._class_name == "SGD"
        assert "betas" not in optimizer.backend.kwargs
        assert "eps" not in optimizer.backend.kwargs

    def test_scipy_backend_takes_its_method(self):
        optimizer = MACTFOptimizer.from_block({"TYPE": "scipy", "METHOD": "L-BFGS-B"})
        assert isinstance(optimizer.backend, ScipyOptimizer)
        assert optimizer.backend.method == "L-BFGS-B"

    def test_scipy_drops_the_torch_only_keys(self):
        spec = optimizer_spec({"TYPE": "scipy", "CLASS": "AdamW", "EPS": 1e-08, "METHOD": "BFGS"})
        assert spec["type"] == "scipy"
        assert "class" not in spec and "eps" not in spec
        assert spec["method"] == "BFGS"

    def test_unknown_class_is_refused_with_its_name(self):
        with pytest.raises(ValidationError, match="Nope"):
            MACTFOptimizer.from_block({"CLASS": "Nope"})

    def test_unknown_backend_type_is_refused(self):
        with pytest.raises(ValidationError, match="unknown optimizer backend"):
            MACTFOptimizer.from_block({"TYPE": "quantum"})

    def test_node_path_is_an_argument_not_a_key(self):
        optimizer = MACTFOptimizer.from_block({"CLASS": "Adam", "LR": 0.01}, node_path="peer/a")
        assert optimizer.node_path == "peer/a"
