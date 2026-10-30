"""``mactf checkpoints`` (document 16, section 8.13) against the seeded nine-run registry."""

from __future__ import annotations


def items(seeded, *argv):
    out = seeded.invoke("checkpoints", "list", *argv, "--format", "json")
    assert out.code == 0, out.stderr
    return out.json()["items"]


def episodes(seeded, run_id):
    return sorted(int(row["episode"]) for row in seeded.registry.checkpoints(run_id))


def test_list_names_every_checkpoint_of_the_run(seeded):
    rows = items(seeded, "run-completed-a")
    assert [row["episode"] for row in rows] == [1, 2, 3]
    assert all(row["size_bytes"] > 0 and row["path"].endswith(".pt") for row in rows)


def test_list_defaults_to_the_latest_run(seeded):
    out = seeded.invoke("checkpoints", "list", "--format", "json")
    assert out.code == 0 and out.json()["run_id"] == "run-null-metrics"


def test_list_prints_a_table_in_text(seeded):
    out = seeded.invoke("checkpoints", "list", "run-paused")
    assert out.code == 0 and "run run-paused" in out.stdout and "EP" in out.stdout


def test_list_verify_flags_a_corrupt_file_and_valid_only_hides_it(seeded):
    rows = items(seeded, "run-completed-b", "--verify")
    assert [row["verified"] for row in rows] == [False]
    assert items(seeded, "run-completed-b", "--valid-only") == []
    assert all(row["verified"] for row in items(seeded, "run-completed-a", "--valid-only"))


def test_list_best_selects_nothing_when_none_is_marked_best(seeded):
    assert items(seeded, "run-completed-a", "--best") == []


def test_list_a_run_without_checkpoints_succeeds_empty(seeded):
    out = seeded.invoke("checkpoints", "list", "run-failed", "--format", "json")
    assert out.code == 0 and out.json()["count"] == 0
    assert "no checkpoints" in seeded.invoke("checkpoints", "list", "run-failed").stdout


def test_unknown_and_ambiguous_runs(seeded):
    assert seeded.invoke("checkpoints", "list", "nope").code == 3
    assert seeded.invoke("checkpoints", "list", "run-comp").code == 2


def test_show_describes_and_verifies_one_checkpoint(seeded):
    entry = seeded.invoke("checkpoints", "show", "run-completed-a", "2", "--format", "json").json()
    assert entry["episode"] == 2 and entry["verified"] is True and entry["exists"] is True
    assert seeded.invoke("checkpoints", "show", "run-completed-a", "latest", "--format", "json").json()["episode"] == 3
    text = seeded.invoke("checkpoints", "show", "run-completed-a", "1")
    assert text.code == 0 and "verified" in text.stdout


def test_show_missing_episode_is_not_found_and_bad_episode_is_usage(seeded):
    assert seeded.invoke("checkpoints", "show", "run-completed-a", "99").code == 3
    assert seeded.invoke("checkpoints", "show", "run-failed", "latest").code == 3
    assert seeded.invoke("checkpoints", "show", "run-completed-a", "abc").code == 2
    assert seeded.invoke("checkpoints", "show", "run-completed-a", "-1").code == 2


def test_verify_passes_a_healthy_run_and_fails_a_corrupt_one(seeded):
    out = seeded.invoke("checkpoints", "verify", "run-completed-a", "--format", "json")
    assert out.code == 0 and out.json()["ok"] is True and out.json()["count"] == 3
    bad = seeded.invoke("checkpoints", "verify", "run-completed-b", "--format", "json")
    assert bad.code == 1 and bad.json()["failed"] == 1
    assert seeded.invoke("checkpoints", "verify", "run-completed-b").code == 1


def test_verify_one_episode(seeded):
    out = seeded.invoke("checkpoints", "verify", "run-completed-a", "2", "--format", "json")
    assert out.code == 0 and [item["episode"] for item in out.json()["items"]] == [2]


def test_rollback_dry_run_changes_nothing(seeded):
    before = episodes(seeded, "run-completed-a")
    out = seeded.invoke("checkpoints", "rollback", "run-completed-a", "1", "--dry-run", "--format", "json")
    assert out.code == 0, out.stderr
    assert len(out.json()["would_archive"]) == 2
    assert episodes(seeded, "run-completed-a") == before
    assert len(list(seeded.checkpoint_dir.glob("run-completed-a-*.pt"))) == 3


def test_rollback_archives_newer_checkpoints(seeded):
    out = seeded.invoke("checkpoints", "rollback", "run-completed-a", "1", "--yes", "--format", "json")
    assert out.code == 0, out.stderr
    assert episodes(seeded, "run-completed-a") == [1]
    assert len(list(seeded.checkpoint_dir.glob("run-completed-a-*.pt"))) == 1
    assert len(list(seeded.checkpoint_dir.glob("run-completed-a-*.rolledback"))) == 2


def test_rollback_to_a_missing_episode_is_not_found(seeded):
    assert seeded.invoke("checkpoints", "rollback", "run-completed-a", "9", "--yes").code == 3
    assert episodes(seeded, "run-completed-a") == [1, 2, 3]


def test_rollback_will_not_prompt_without_a_terminal(seeded):
    assert seeded.invoke("checkpoints", "rollback", "run-completed-a", "1").code == 2
    assert episodes(seeded, "run-completed-a") == [1, 2, 3]


def test_rollback_refuses_a_run_being_driven(seeded):
    assert seeded.invoke("checkpoints", "rollback", "run-running-live", "1", "--yes").code in (3, 4)


def test_prune_dry_run_then_apply(seeded):
    dry = seeded.invoke("checkpoints", "prune", "run-completed-a", "--keep", "1", "--dry-run",
                        "--format", "json")
    assert dry.code == 0, dry.stderr
    assert len(dry.json()["runs"][0]["removed"]) == 2
    assert episodes(seeded, "run-completed-a") == [1, 2, 3]
    applied = seeded.invoke("checkpoints", "prune", "run-completed-a", "--keep", "1", "--yes",
                            "--format", "json")
    assert applied.code == 0, applied.stderr
    assert episodes(seeded, "run-completed-a") == [3]
    assert [p.name for p in seeded.checkpoint_dir.glob("run-completed-a-*.pt")] == ["run-completed-a-ep0003.pt"]


def test_prune_nothing_to_do(seeded):
    out = seeded.invoke("checkpoints", "prune", "run-paused", "--keep", "5", "--yes")
    assert out.code == 0 and "nothing to prune" in out.stdout


def test_prune_all_skips_the_live_run_and_trims_the_rest(seeded):
    out = seeded.invoke("checkpoints", "prune", "--all", "--keep", "1", "--yes", "--format", "json")
    assert out.code == 0, out.stderr
    assert episodes(seeded, "run-completed-a") == [3]
    assert episodes(seeded, "run-paused") == [2]
    assert episodes(seeded, "run-completed-b") == [1]            # already at the limit


def test_prune_needs_a_run_or_all_but_not_both(seeded):
    assert seeded.invoke("checkpoints", "prune", "--yes").code == 2
    assert seeded.invoke("checkpoints", "prune", "run-paused", "--all", "--yes").code == 2


def test_prune_refuses_to_prompt_without_a_terminal(seeded):
    assert seeded.invoke("checkpoints", "prune", "run-completed-a", "--keep", "1").code == 2
    assert episodes(seeded, "run-completed-a") == [1, 2, 3]
