# Scoring

`ScoringPipeline` turns one peer's episode into a `PeerScorecard`. Stages are `Scorer`
subclasses (`RewardScorer`, `PenaltyScorer`, the warning regime, `TaskJudge`), registered in
`SCORER_TYPES` and assembled by `build_scorer`. A custom scorer subclasses `Scorer` and returns
a `ScorerOutput`; it is selected through settings, not by editing the pipeline.

## Two answers, two names

A scorecard carries two numbers that answer different questions. Do not compare them.

| Field | Question | Source |
|---|---|---|
| `final_score` | How well did this peer behave, continuously? | reward terms minus penalties, with the warning regime applied |
| `final_points` | How many rubric points were earned, with evidence? | `PointsEngine`, 0-100, guarded |

`final_score` is shaped by the rule ledger: each `RuleEvent` (rule id, description, magnitude)
is a violation, and repeated violations escalate through `WarningPolicy`. A void episode
(`is_void`) is a hard floor. `final_points` is the sum of rubric lines whose guards passed.
Each rubric line is bound to a guard (`GUARDS`, `BINDINGS`) that checks `EpisodeEvidence`; a line
whose evidence is missing is **denied**, and the denial is recorded in `scorecard.points`
(`DenialRecord`) instead of silently scoring zero. `card.points` is a dict that also reads as
attributes; the weights sum to exactly 100.

`final_points` can be identical across peers while `final_score` differs, and the reverse. The
aggregation signal is a third value, see [calibration.md](calibration.md).

## Evidence

When the orchestrator scores points, the default evidence observes only what the ledger
shows (`task_checks_passed` is true when rule `R1` fired), so rubric lines that need other
evidence are denied and the missing key is recorded. Supply real evidence in-process by
passing an `EpisodeEvidence` (with `ClaimAudit` entries) when scoring. The HTTP
`/evaluations` route scores a ledger only; the `evidence` field of each event there is the
event's free-text justification, not rubric evidence.

## Scorecard versions

v1 is the stored shape. v2 adds `points`, `normalization` and `guards_denied` and is requested
with `Accept: application/vnd.mactf.v2+json`. See [api.md](api.md).

## Reading rules

`client.rules.list()` (or `GET /rules`) lists every rule with its tier and severity;
`client.rules.validation()` reports rules referenced in a ledger but unknown to the policy.
