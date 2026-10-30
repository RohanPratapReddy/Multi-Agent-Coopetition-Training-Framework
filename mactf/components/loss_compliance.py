"""``TRAINER.LOSS_COMPLIANCE``: every loss carries the compliance term (document 17 section 7.8)."""

from __future__ import annotations

from typing import Any, Iterator

from ..exceptions import ValidationError

#: The ``spec_type`` of the two framework terms that carry the compliance charge (the classes
#: themselves, not these names, are what ``carries_compliance`` trusts).
CARRIES_COMPLIANCE = frozenset({"compliance_floor", "composite"})


def _carriers() -> tuple[type, ...]:
    from ..rl.functions.loss import ComplianceFloorLoss, CompositeLoss

    return (ComplianceFloorLoss, CompositeLoss)


def _containers() -> tuple[type, ...]:
    from ..rl.functions.loss import ChainedLoss, WeightedSumLoss

    return (ChainedLoss, WeightedSumLoss)


def _exact(value: Any, kinds: tuple[type, ...]) -> bool:
    """Identity on the class: neither ``spec_type`` nor a ``terms`` attribute can claim it."""
    return any(type(value) is kind for kind in kinds)


def iter_terms(loss: Any) -> Iterator[Any]:
    """Depth-first walk of a composed loss, entering only the framework's own containers.

    A custom loss may carry a ``terms`` attribute it never calls; it is not walked.
    """
    yield loss
    if _exact(loss, _containers()):
        for term in getattr(loss, "terms", ()) or ():
            yield from iter_terms(term)


def carries_compliance(loss: Any) -> bool:
    """True when the framework's own compliance term is in the loss tree.

    Decided by the class of each node, never by what it says about itself: a custom loss that
    sets ``spec_type = "compliance_floor"`` carries no compliance term.
    """
    carriers = _carriers()
    return any(_exact(term, carriers) for term in iter_terms(loss))


def ensure_compliance(loss: Any, *, mode: str, coefficient: float) -> Any:
    """Apply ``TRAINER.LOSS_COMPLIANCE`` to a resolved loss.

    ``mode``: ``"require"`` (default) adds ``compliance_floor`` when it is missing, so a
    custom objective is *extended*, never rejected; ``"enforce"`` raises instead, for an
    operator who wants the author to say so; ``"off"`` leaves the loss alone (a research
    run on a node tree with no non-offsettable rules).
    """
    if mode == "off" or carries_compliance(loss):
        return loss
    if mode == "enforce":
        raise ValidationError(
            f"loss {getattr(loss, 'spec_type', type(loss).__name__)!r} has no compliance_floor "
            "term and TRAINER.LOSS_COMPLIANCE is 'enforce'. Add {type: compliance_floor} or "
            "use TYPE: composite."
        )
    if mode != "require":
        raise ValidationError(f"TRAINER.LOSS_COMPLIANCE must be off, require or enforce; got {mode!r}")
    from ..rl.functions.loss import ComplianceFloorLoss, WeightedSumLoss

    return WeightedSumLoss(
        [loss, ComplianceFloorLoss(coefficient=coefficient)], weights=[1.0, 1.0]
    )
