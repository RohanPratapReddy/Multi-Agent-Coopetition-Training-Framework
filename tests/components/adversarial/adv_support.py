"""Harness of the adversarial compliance tests: a real trainer that is forced to see a breach.

The trainer's own task judge decides what a ledger holds; to attack the *compliance* boundary
the harness wraps ``scorer.score_episode`` and records a non-offsettable breach (``P1``) on
chosen peers just before scoring. Everything after that is production code.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from mactf.training import MACTFTrainer

from p2_support import configure, make_trainer, tune

#: A non-offsettable rule: its breach voids the node, floors its score and carries no weight.
BREACH = "P1"
HONEST_COST = -5.0          # the final score the built-in pipeline gives one P1 event of magnitude 1


def breach_trainer(settings, tmp_path, *, blocks: dict[str, Any] | None = None,
                   victims: Iterable[str] = ("peer-1",), peers: int = 3, epochs: int = 2,
                   seed: int = 0) -> MACTFTrainer:
    """A built trainer whose scorer records ``BREACH`` on every victim before each scoring."""
    tune(settings, EPOCHS=epochs)
    configure(settings, **(blocks or {}))
    trainer = make_trainer(settings, tmp_path, peers=peers, seed=seed)
    trainer.build()
    inject_breach(trainer, victims)
    return trainer


def inject_breach(trainer: MACTFTrainer, victims: Iterable[str], rule: str = BREACH) -> None:
    scorer = trainer.scorer
    names = tuple(victims)

    def plant(ledgers):
        for name in names:
            if name in ledgers and not ledgers[name].for_rule(rule):
                ledgers[name].record(rule, magnitude=1.0)

    if hasattr(scorer, "score_tree"):                       # hierarchical: (tree, ledgers, ...)
        tree_original = scorer.score_tree

        def wrapped_tree(tree, ledgers, *args, **kwargs):
            plant(ledgers)
            return tree_original(tree, ledgers, *args, **kwargs)

        scorer.score_tree = wrapped_tree
        return
    original = scorer.score_episode

    def wrapped(ledgers, *args, **kwargs):
        plant(ledgers)
        return original(ledgers, *args, **kwargs)

    scorer.score_episode = wrapped


def run_epochs(trainer: MACTFTrainer, epochs: int = 2) -> None:
    for epoch in range(epochs):
        trainer.run_epoch(epoch)


def victim_cards(trainer: MACTFTrainer, victim: str = "peer-1"):
    return [card.peers[victim] for card in trainer.scorecards if victim in card.peers]


def assert_breach_cost_intact(trainer: MACTFTrainer, victim: str = "peer-1") -> None:
    """The victim was voided every episode, scored no better than the built-in cost, and weighed 0."""
    cards = victim_cards(trainer, victim)
    assert cards, "the victim was never scored"
    for card in cards:
        assert card.is_void, "a void node was un-voided"
        assert BREACH in card.violated_non_offsettable
        assert card.final_score <= HONEST_COST + 1e-9, f"the breach cost was lowered: {card.final_score}"
    for report in trainer.reports:
        assert report.weights.get(victim, 0.0) == 0.0, f"a void peer carries weight: {report.weights}"


def reference_scores(settings_factory, tmp_path, *, epochs: int = 2) -> list[float]:
    """The final scores of the victim under the unmodified framework (the yardstick)."""
    trainer = breach_trainer(settings_factory(), tmp_path / "reference", epochs=epochs)
    run_epochs(trainer, epochs)
    return [card.final_score for card in victim_cards(trainer)]
