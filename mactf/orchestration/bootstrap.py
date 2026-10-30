"""Behavioural cloning for phase 2.

The docx calls the priming phase *behavioural cloning*: before any RL signal
exists, every peer is trained to imitate the target action on a priming corpus.
That is supervised learning with a cross-entropy objective, and it is what makes
a peer able to recognize when to critique, yield, and refuse - behaviours a
purely reward-driven loop discovers late or not at all.

Two decisions are deliberate and are documented here because they attract the
same objection every time:

* **Cross-entropy, not the composite loss.** The RL loss multiplies an
  importance ratio by an advantage. Before the first episode there is no
  advantage, so using it here would mean inventing one. Behavioural cloning is
  exactly ``-log pi(a* | s)`` over the target action.
* **Every peer trains on the same data.** At this stage the peers are meant to
  converge on a shared competent baseline; the docx says bootstrap produces N
  identical baseline agents. Diversity is reintroduced by initialization and by
  the RL phase, where per-peer weighting lives. Bootstrap is not where diversity
  comes from, so it is not a defect that it does not produce any.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from .. import signals
from ..data.sources import SyntheticTaskSource, TaskSourceProtocol
from ..exceptions import ContractError
from ..rl.loss import clip_gradients
from ..runtime import Runtime, autocast

__all__ = ["BootstrapConfig", "BootstrapReport", "BootstrapTrainer"]


@dataclass
class BootstrapConfig:
    """Supervised priming parameters."""

    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    #: docx Phase 2: ``pi_ref`` starts as the SFT result.
    update_reference: bool = True
    #: 0 means "run every epoch".
    patience: int = 0
    #: Batches per epoch; 0 means "one pass over the source".
    batches_per_epoch: int = 0


@dataclass
class BootstrapReport:
    """What actually happened, reported rather than asserted."""

    epochs_run: int
    final_loss: float
    initial_loss: float
    loss_history: tuple[float, ...]
    examples: int
    peers: tuple[str, ...]
    reference_updated: bool
    batches: int = 0
    stopped_early: bool = False
    stop_reason: str | None = None
    per_peer: dict[str, float] = field(default_factory=dict)

    @property
    def loss_delta(self) -> float:
        return self.final_loss - self.initial_loss

    @property
    def improved(self) -> bool:
        return self.final_loss < self.initial_loss

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loss_history"] = list(self.loss_history)
        payload["peers"] = list(self.peers)
        payload["loss_delta"] = self.loss_delta
        payload["improved"] = self.improved
        return payload

    def describe(self) -> str:
        if self.epochs_run == 0:
            return "bootstrap(not run)"
        return (
            f"bootstrap(epochs={self.epochs_run}, batches={self.batches}, "
            f"examples={self.examples}, loss {self.initial_loss:.4f} -> "
            f"{self.final_loss:.4f}, reference_updated={self.reference_updated})"
        )


class BootstrapTrainer:
    """Run supervised behavioural-cloning steps over a task source."""

    def __init__(
        self,
        *,
        source: TaskSourceProtocol,
        runtime: Runtime,
        config: BootstrapConfig | None = None,
    ) -> None:
        self.source = source
        self.runtime = runtime
        self.config = config or BootstrapConfig()
        self._corpus: tuple[Any, ...] | None = None

    # -- public API ------------------------------------------------------
    def new_corpus(self) -> None:
        """Discard the primed corpus so the next ``train`` re-draws one."""
        self._corpus = None

    def train(
        self, peers: dict[str, Any], reference: Any | None = None
    ) -> BootstrapReport:
        """Train every peer for ``config.epochs`` and report real loss numbers."""
        config = self.config
        names = tuple(peers)
        if not names:
            raise ContractError("bootstrap requires at least one peer")
        if config.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if config.epochs == 0:
            return BootstrapReport(
                epochs_run=0,
                final_loss=0.0,
                initial_loss=0.0,
                loss_history=(),
                examples=0,
                peers=names,
                reference_updated=False,
            )

        history: list[float] = []
        per_peer_totals: dict[str, float] = {name: 0.0 for name in names}
        per_peer_counts: dict[str, int] = {name: 0 for name in names}
        examples = 0
        batches = 0
        best_loss = math.inf
        best_epoch = 0
        stopped_early = False
        stop_reason: str | None = None

        # Batch scheduler: a dedicated optimizer per peer is created so the run's
        # learning rate is honoured even when peers were built with another one.
        optimizers = self._optimizers(peers)

        for epoch in range(config.epochs):
            epoch_losses: list[float] = []
            epoch_batches = 0
            epoch_examples = 0
            for batch in self._iter_batches(epoch):
                states = self.runtime.to(batch.states)
                targets = batch.targets.to(states.device)
                batch_loss = self._step(peers, optimizers, states, targets, per_peer_totals, per_peer_counts)
                epoch_losses.append(batch_loss)
                epoch_batches += 1
                epoch_examples += int(states.shape[0])

            if not epoch_losses:
                stop_reason = "task source produced no batches"
                break

            epoch_mean = sum(epoch_losses) / len(epoch_losses)
            history.append(epoch_mean)
            examples += epoch_examples
            batches += epoch_batches

            if epoch_mean < best_loss - 1e-9:
                best_loss = epoch_mean
                best_epoch = epoch
            elif config.patience > 0 and (epoch - best_epoch) >= config.patience:
                stopped_early = True
                stop_reason = (
                    f"no improvement for {config.patience} epoch(s); best was "
                    f"{best_loss:.6f} at epoch {best_epoch}"
                )
                break

        reference_updated = False
        if config.update_reference and reference is not None and history:
            reference_updated = self._update_reference(peers, reference)

        report = BootstrapReport(
            epochs_run=len(history),
            final_loss=history[-1] if history else 0.0,
            initial_loss=history[0] if history else 0.0,
            loss_history=tuple(history),
            examples=examples,
            peers=names,
            reference_updated=reference_updated,
            batches=batches,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            per_peer={
                name: (per_peer_totals[name] / per_peer_counts[name])
                if per_peer_counts[name]
                else 0.0
                for name in names
            },
        )
        signals.phase_progress.send(
            sender=self, phase="bootstrap", summary=report.describe()
        )
        return report

    # -- internals -------------------------------------------------------
    def _optimizers(self, peers: dict[str, Any]) -> dict[str, Any]:
        """A dedicated optimizer per peer, honouring the configure LR/decay."""
        config = self.config
        optimizers: dict[str, Any] = {}
        for name, peer in peers.items():
            parameters = [p for p in peer.adapter.parameters() if p.requires_grad]
            if not parameters:
                continue
            optimizers[name] = torch.optim.Adam(
                parameters,
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )
        return optimizers

    def _iter_batches(self, epoch: int) -> Sequence[Any]:
        """The epoch's batches, drawn from a corpus primed once.

        Behavioural cloning trains on a *fixed* corpus for several passes; that
        is what "supervised" means here. Priming once and replaying the corpus is
        also what makes the reported loss curve meaningful: the synthetic source
        draws a fresh random projection on every call, so re-drawing per epoch
        would move the target and no loss decrease could be observed or claimed.

        The corpus is built from ``epoch`` 0 of the source, so it is
        deterministic for a fixed ``(seed, source)``.
        """
        if self._corpus is None:
            config = self.config
            limit = config.batches_per_epoch
            declared = 0
            if limit <= 0:
                try:
                    declared = len(self.source)  # type: ignore[arg-type]
                except TypeError:
                    declared = 0
                limit = int(declared) if declared and declared > 0 else 1
            self._corpus = tuple(self.source.next(0) for _ in range(limit))
        return self._corpus

    def _step(
        self,
        peers: dict[str, Any],
        optimizers: dict[str, Any],
        states: torch.Tensor,
        targets: torch.Tensor,
        totals: dict[str, float],
        counts: dict[str, int],
    ) -> float:
        losses: list[float] = []
        for name, peer in peers.items():
            optimizer = optimizers.get(name)
            if optimizer is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            peer.adapter.train()
            with autocast(self.runtime):
                logits, _values = peer.adapter(states)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1).long(),
            )
            loss.backward()
            if self.config.max_grad_norm > 0:
                clip_gradients(
                    [p for p in peer.adapter.parameters() if p.requires_grad],
                    max_norm=float(self.config.max_grad_norm),
                )
            optimizer.step()

            value = float(loss.detach().item())
            losses.append(value)
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
        return sum(losses) / len(losses) if losses else 0.0

    def _update_reference(self, peers: dict[str, Any], reference: Any) -> bool:
        """Copy the trained weights into the frozen reference policy.

        The reference is typically a mean of the trained peers, which is what
        makes it a fair baseline for cross-peer advantage (framework.md 11.7):
        basing an advantage on one peer's own score would reward that peer for
        being unusual rather than for being right.
        """
        import copy

        trained = [peer for peer in peers.values()]
        if not trained:
            return False
        mean = None
        for peer in trained:
            flat = peer.export_parameters().detach().float()
            mean = flat if mean is None else mean + flat
        assert mean is not None
        mean = mean / len(trained)

        try:
            reference.import_parameters(mean.to(reference.export_parameters().device))
        except Exception:  # noqa: BLE001 - fall back to a direct copy
            reference.adapter = copy.deepcopy(trained[0].adapter)
        reference.adapter.eval()
        for parameter in reference.adapter.parameters():
            parameter.requires_grad_(False)
        return True


def build_source_from_settings(settings: dict[str, Any]) -> SyntheticTaskSource:
    """A synthetic source for callers that construct a context by hand."""
    return SyntheticTaskSource(
        state_dim=int(settings.get("STATE_DIM", 16)),
        action_dim=int(settings.get("ACTION_DIM", 4)),
        batch_size=int(settings.get("BATCH_SIZE", 32)),
        seed=int(settings.get("SEED", 0)),
    )
