# Publishing MACTF

The release pipeline is [`.github/workflows/deploy.yml`](workflows/deploy.yml).
It publishes with [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
which means **no API token and no repository secret is stored anywhere**. Instead
GitHub mints a short-lived OIDC token for the workflow run, and PyPI verifies it
against a publisher record you create once.

Everything below is one-time setup that cannot be done from a file in this repo.

---

## 1. Register the publisher on PyPI

The project `mactf` must exist, or be registered as a *pending* publisher.

**These are two separate services with two separate sets of records.** A publisher
registered on pypi.org does nothing for a TestPyPI upload, and vice versa. Register
both if you want both paths.

### TestPyPI (pre-release tags)

<https://test.pypi.org/manage/account/publishing/>

| Field | Value |
| --- | --- |
| PyPI Project Name | `mactf` |
| Owner | `RohanPratapReddy` |
| Repository name | `Multi-Agent-Coopetition-Training-Framework` |
| Workflow name | `deploy.yml` |
| Environment name | `testpypi` |

### PyPI (final tags)

<https://pypi.org/manage/account/publishing/>

| Field | Value |
| --- | --- |
| PyPI Project Name | `mactf` |
| Owner | `RohanPratapReddy` |
| Repository name | `Multi-Agent-Coopetition-Training-Framework` |
| Workflow name | `deploy.yml` |
| Environment name | `pypi` |

The environment name is part of the OIDC token subject, so it is matched literally:

```
sub: repo:RohanPratapReddy@68410459/Multi-Agent-Coopetition-Training-Framework@1425146762:environment:testpypi
```

A `v0.0.0rc1` tag therefore requires the **TestPyPI** record with environment
`testpypi`. If that record is missing, the upload fails with:

```
invalid-publisher: valid token, but no corresponding publisher
```

Setting up only the `pypi` record is enough for final releases and does nothing for
pre-releases. The workflow name must be exactly `deploy.yml`, not the full path.


## 2. Create the GitHub environments

Repository Settings -> Environments -> New environment:

- `testpypi` — no protection rules required.
- `pypi` — add **Required reviewers** so a human approves the production upload.

If you add required reviewers, the `publish-pypi` job pauses between the build and
the upload. The build artifact is still uploaded on every run, so nothing is lost.

## 3. Release

The version currently in `mactf/__init__.py` is `0.0.0rc1`, which is a
**pre-release**: tag it and it goes to TestPyPI, never PyPI. Use this flow first.

```powershell
# 1. tag the pre-release and push
git tag v0.0.0rc1
git push origin v0.0.0rc1
```

For a final release, bump the single source of truth and tag the same number:

```powershell
# 1. bump the single source of truth
#    mactf/__init__.py -> __version__ = "0.1.0"
git add mactf/__init__.py
git commit -m "Release 0.1.0"

# 2. tag and push
git tag v0.1.0
git push origin main --tags
```

The tag must match `mactf.__version__` exactly (a leading `v` is optional and
stripped). The `build` job fails loudly on a mismatch before anything is uploaded.

Pipeline for a final tag:

```
resolve -> test (3.10, 3.11, 3.12, 3.13) -> build -> publish-pypi
                                                        |-> verify-install (ubuntu, windows)
                                                        |-> github-release
```

## 4. Pre-release

A tag whose version has a letter suffix goes to TestPyPI and stops there. This is
what the current `0.0.0rc1` is for:

```powershell
git tag v0.0.0rc1
git push origin v0.0.0rc1
```

TestPyPI cannot resolve `torch` from its index, so this path validates packaging
metadata and the upload flow, not a full install. The version-format test is:

- `v0.1.0` -> numeric only -> PyPI
- `v0.0.0rc1`, `v2.0.0b3`, `v1.2.0.dev4` -> has a letter/`dev` suffix -> TestPyPI

## 5. Manual dry run

Actions -> **Deploy** -> Run workflow -> `target = build-only`.

That runs the test matrix and the build, validates with `twine check`, and uploads
the `dist` artifact without publishing. Use `testpypi` or `pypi` to exercise the
publish path without a tag.

---

## What the workflow guarantees

- `test` runs on Python 3.10 through 3.13 and installs CPU-only torch, so the suite
  is checked without a GPU and without an 800 MB CUDA download.
- `build` runs once; the same `dist/` bytes are uploaded to whichever index is
  selected. `twine check` validates both the wheel and the sdist.
- `id-token: write` is present only on `publish-testpypi` and `publish-pypi`. No
  other job can request an OIDC token.
- `concurrency` is keyed on the ref with `cancel-in-progress: false`, so a release
  is never cancelled mid-upload.
- `attestations: true` attaches PEP 740 provenance, which also populates the
  "Verified" badge on PyPI.
- `verify-install` installs from the public index on Linux and Windows, runs
  `mactf check`, scaffolds a project with `startproject`, and checks that project.

## Re-running a failed release

Re-run the failed job from the Actions UI. `skip-existing: true` on the TestPyPI job
makes that safe for pre-releases. The PyPI job deliberately omits it: a production
version can never be overwritten, so a duplicate there means you forgot to bump
`mactf/__init__.py` before tagging.

Prefer `gh run rerun <run-id> --failed` over deleting and re-pushing the tag. Pushing
the same tag twice creates two concurrent runs of an identical release; whichever
finishes second will collide on an already-uploaded filename. To restart from a clean
tag instead:

```powershell
git push origin :refs/tags/v0.0.0rc1    # delete the remote tag
git tag -d v0.0.0rc1                    # delete it locally too
git tag v0.0.0rc1                       # recreate at HEAD
git push origin v0.0.0rc1
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `invalid-publisher: valid token, but no corresponding publisher` | No publisher record on the index that the tag routes to, or one field disagrees. Read the `sub:` claim in the log and match its environment name. A pre-release tag needs the **TestPyPI** record. |
| Publishing works for `v0.1.0` but not `v0.1.0rc1` | Only the PyPI record exists. Add the TestPyPI one. |
| `File already exists` on PyPI | The version was already uploaded. Bump `mactf/__init__.py`; a published version can never be replaced. |
| `tag ... does not match mactf.__version__` | The git tag and the package version drifted. Both must be identical apart from the leading `v`. |
| Job skipped entirely | The `resolve` job sent this ref to the other index, or the `if` condition did not match. Check `publish target:` in the `Resolve publish target` log. |


## Known gaps

- **No `ruff` gate.** `ruff check .` currently reports findings across the package
  (23 unused imports, 22 unsorted imports, plus a handful of real defects including
  an undefined `os` in an annotation in `mactf/sandbox/channel.py`). A lint gate
  would make every release red until that cleanup lands, so it is intentionally not
  in this pipeline. Add `python -m ruff check .` to the `test` job once the tree is
  clean.
- **mypy is installed but not gated** for the same reason.
- **Python 3.14 is not in the matrix** even though it is a declared classifier,
  because the CPU torch wheel index does not publish 3.14 Linux wheels yet. Add it
  to the matrix once it does.
