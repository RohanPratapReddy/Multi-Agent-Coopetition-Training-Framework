"""The compatibility shim (document 07 section 7.3).

``mactf.scoring.pipeline.Scorer`` keeps constructing a working pipeline through
``0.2.x`` with a ``DeprecationWarning``; ``mactf.scoring.Scorer`` is the ABC.
"""

from __future__ import annotations

import warnings

import pytest

from mactf.scoring.base import ScoringKind
from mactf.scoring.pipeline import ScoringPipeline


def test_pipeline_scorer_emits_a_deprecation_warning():
    import mactf.scoring.pipeline as pipeline_module

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = pipeline_module.Scorer
    assert resolved is ScoringPipeline
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert "ScoringPipeline" in str(caught[0].message)


def test_pipeline_scorer_still_constructs_a_working_pipeline(
    score_config, severity_weights, warning_policy
):
    from mactf.scoring.ledger import Ledger
    from mactf.scoring.pipeline import Scorer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        scorer = Scorer(
            score_config=score_config,
            severity_weights=severity_weights,
            warning_policy=warning_policy,
        )

    assert isinstance(scorer, ScoringPipeline)
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=2.0)
    card = scorer.score_peer(ledger, peer="a", episode=1)
    assert card.final_score == pytest.approx(2.0)


def test_scoring_dot_scorer_is_the_abstract_base():
    import mactf.scoring as scoring

    assert scoring.Scorer is not ScoringPipeline
    assert scoring.Scorer.__name__ == "Scorer"
    assert getattr(scoring.Scorer, "__abstractmethods__", frozenset()) == frozenset({"score"})


def test_scoring_dot_scorer_cannot_be_instantiated():
    from mactf.scoring import Scorer

    with pytest.raises(TypeError):
        Scorer()  # type: ignore[abstract]


def test_scoring_dot_scorer_has_the_declared_class_attributes():
    from mactf.scoring import Scorer

    # ``kind`` is a ClassVar annotation, not a value: an implementation must
    # declare its own, so the ABC has no default to inherit by accident.
    assert "kind" in Scorer.__annotations__
    assert Scorer.order == 100
    assert Scorer.requires == frozenset()


def test_pipeline_is_a_scoring_dot_scorer():
    from mactf.scoring import Scorer

    assert issubclass(ScoringPipeline, Scorer)


def test_pipeline_scorer_shim_does_not_warn_on_other_attributes():
    import mactf.scoring.pipeline as pipeline_module

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pipeline_module.ScoringPipeline is ScoringPipeline
    assert caught == []


def test_unknown_attribute_raises_attribute_error():
    import mactf.scoring.pipeline as pipeline_module

    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        pipeline_module.Nope


def test_star_import_does_not_expose_the_shim():
    namespace: dict[str, object] = {}
    exec("from mactf.scoring.pipeline import *", namespace)
    # ``__all__`` deliberately omits the shim so a star import gets the real names.
    assert "Scorer" not in namespace
    assert "ScoringPipeline" in namespace


def test_shim_build_scorer_returns_the_pipeline(score_config, severity_weights, warning_policy):
    from mactf.scoring.pipeline import build_stages

    stages = build_stages(
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=warning_policy,
    )
    pipeline = ScoringPipeline(
        stages,
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=warning_policy,
    )
    assert pipeline.get(ScoringKind.WARNING).order == 5
