# MACTF — Multi-Agent Coopetition Training Framework

A Django-shaped training framework for **any** `torch.nn.Module`, built for PyTorch
swarms that must be trained *and* audited.

The numeric objective comes from `docs/Designing_Multi_Agent_Incentive_Frameworks.docx`
(continuous reward, penalty, warning, softmax peer weighting, adaptive-κ momentum).
The structural guarantees come from `docs/framework.md` (precedence tiers T0–T6,
severity S0–S4, rules R1–R17 / P1–P23 / W1–W9, non-offsettable floors). Where the
two disagree, `framework.md` wins, and the deviation is visible in `mactf/compliance.py`.

---

## Install

```powershell
pip install mactf                # library + `mactf` CLI
pip install "mactf[shell]"       # + IPython/bpython for `manage.py shell`
```

CUDA builds of torch are not expressible as a PyPI extra. Install torch from the
matching index first, then the package:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install "mactf[gpu]"
```

A bare install has no project, so the CLI falls back to
`mactf.settings_default` (no peers, no overlays). `check` and `doctor` work
immediately; training needs a project:

```powershell
mactf startproject myswarm
cd myswarm
python manage.py check
python manage.py migrate
python manage.py run --epochs 3
```

`mactf` and `python -m mactf` are equivalent. `training/` in this repository is
the reference project; it ships in the sdist, not in the wheel.

---

## Quick start

From a checkout of this repository:

```powershell
git clone https://github.com/RohanPratapReddy/Multi-Agent-Coopetition-Training-Framework
cd Multi-Agent-Coopetition-Training-Framework

python training\manage.py check       # system checks: rules, scoring, peers, sandbox
python training\manage.py doctor      # environment, GPU, QEMU/isolation report
python training\manage.py migrate     # create the run registry (sqlite)
python training\manage.py run --epochs 3
```

GPU is on by default when one is present:

```powershell
python training\manage.py run --epochs 5 --device cuda
python training\manage.py run --epochs 1 --device cpu      # force CPU
```

Result:

```
run a3ef35594ca7 [completed]
  device   : cuda dtype=float32 amp=True
  peers    : peer_alpha, peer_beta, peer_gamma
  epoch 0: score=0.0938 kappa=1.001 entropy=1.089 [peer_alpha=0.398, peer_beta=0.301, peer_gamma=0.301]
  epoch 1: score=0.2501 kappa=1.003 entropy=1.096 [peer_alpha=0.365, peer_beta=0.317, peer_gamma=0.317]
```

---

## The one contract that matters

Your model is any `nn.Module`. Return one of:

```python
return logits                 # adapter attaches a value head
return logits, value          # preferred
return {"logits": ..., "value": ...}
```

`logits` is `(batch, action_dim)`; `value` is `(batch, 1)`. Everything else
(frozen layers, tied weights, dict/tuple/attribute outputs, 1-D logits, missing
value head) is normalized by `mactf/coopetition/adapter.py`.

```python
import torch, torch.nn as nn

class MyPeer(nn.Module):
    def __init__(self, state_dim=16, action_dim=4):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh())
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        h = self.body(state)
        return self.policy(h), self.value(h)
```

Register it, then list it in settings:

```python
PEERS = [
    {"name": "alpha", "class": "MyPeer", "kwargs": {"state_dim": 16, "action_dim": 4}},
    {"name": "beta",  "class": "MyPeer", "kwargs": {"state_dim": 16, "action_dim": 4}},
]
```

**Aggregation constraint:** reward-modulated aggregation averages parameter
vectors, so every peer in one swarm must share a layout. Use one class with
different initialisations for diversity. `require_same_layout()` fails loudly
rather than silently producing a meaningless average.

Scaffold a new peer:

```powershell
python training\manage.py startpeer peers\my_peer --state-dim 16 --action-dim 4
```

---

## Scoring, exactly as implemented

Reward (docx):

```
R = 1.0·r_task + 1.5·r_epi + 1.2·r_collab + 1.0·r_align
```

Penalty (docx), with Ω applied only to those penalties that trigger it:

```
P = (Σ λ_k · p_k) · Ω
```

Warning, then the docx net:

```
net = R − P − W
```

Then `framework.md` is enforced, in this order:

1. **Precedence** — only the highest represented tier scores (`strict` mode).
   A T0/T1/T2 breach therefore suppresses every reward.
2. **Non-offsettable floor** — P1, P2, P17, P20, P21, P22 become an absolute
   negative floor. No reward in the same episode may reduce it.
3. **Ω amplification** — bounded to configured multipliers, never infinity.
4. **Warning zero-score** — a rule that only produced a warning subtracts nothing;
   on escalation it costs `P_WARN` and rises exactly one tier (capped at S3).

Aggregation then post-processes `softmax(κ · score)` in a fixed, reported order:

participation gate → confidence gate → trim outliers → **floor and cap projected
onto the bounded simplex** → disagreement fallback → renormalize.

The cap is enforced by bisection projection, not clamp-and-renormalize, because
that naive cycle can push a weight above the cap again.

### Adaptive κ

```
ω_i   = softmax(κ · R_i)
H     = −Σ ω_i ln ω_i
Δ     = H − H_target          (H_target = ratio · ln N)
m     = β·m + (1−β)·Δ
κ     = clip(κ + η_κ·m, κ_min, κ_max)
```

`κ_max` is an addition to the docx formula: without it, runaway momentum collapses
the softmax onto one peer and violates the weight cap.

---

## Serve and client

The HTTP service is an installed app, selected the way `django.contrib.admin` is.
Add it to `INSTALLED_APPS` and `mactf serve` exists; remove it and the API, its
checks and its command are gone, and nothing else changes. It runs on the standard
library's WSGI server, so there is no web framework to install.

```python
# settings.py
INSTALLED_APPS = [
    "mactf.contrib.scoring", "mactf.contrib.coopetition",
    "mactf.contrib.sandbox", "mactf.contrib.telemetry",
    "mactf.contrib.api",
]
API = {"ALLOW_ANONYMOUS": False,
       "TOKENS": {"plain:s3cret": ["read", "run:write", "score:write"]}}
```

```powershell
python manage.py serve --port 8010
```

The client has two transports that return the same objects and raise the same
exceptions: `HTTPTransport` (standard-library `urllib`) and `InProcessTransport`,
which dispatches through `ROOT_URLCONF` with no socket.

```python
from mactf.client import MACTFClient

# against a running server
client = MACTFClient(transport="http", base_url="http://127.0.0.1:8010", token="s3cret")

run = client.runs.create({"epochs": 2, "seed": 1337, "device": "cpu"})
for event in client.runs.stream(run.id):          # server-sent events, resumable
    print(event.type, event.epoch)
for episode in client.runs.iter_episodes(run.id):
    for peer, card in episode.peers.items():       # PeerScorecard, the framework's class
        print(peer, round(card.final_score, 3))

card = client.score("peer-a", [("P1", "fabricated a citation", 1.0)])
assert card.is_void                                # POST /evaluations

# the same calls with no server at all, in this process
local = MACTFClient(transport="in_process", token="s3cret")
assert local.rules.list() == client.rules.list()
```

`client.scores` reads the 0-100 rubric (`scale()`, `points()`, `normalization()`);
see [docs/api.md](docs/api.md), [docs/client.md](docs/client.md) and
[docs/apps.md](docs/apps.md).

## Layout

```
mactf/
  conf.py          settings loading, YAML merge, env overrides, validation
  apps.py          AppConfig, app registry, peer/provider discovery
  checks.py        system check framework
  signals.py       episode/aggregation/checkpoint signals
  db.py            sqlite run registry with idempotent migrations
  runtime.py       device/dtype/AMP resolution, seeding, atomic checkpoints
  compliance.py    docx formulas reconciled with framework.md invariants
  rules/           catalog, severity, precedence
  scoring/         ledger, warning regime, continuous scorer, judge, scorecard
  rl/              policy, value, advantage, loss, rollout
  coopetition/     PeerAdapter, Peer, adaptive kappa, constrained aggregation
  sandbox/         ABC, detection, QEMU provider, local fallback, channel, provision
  orchestration/   episodes, five phases, checkpoints, orchestrator
  management/      command framework + check/doctor/migrate/run/inspect/shell/
                   startproject/startpeer/sandbox/redteam
  settings_default.py  built-in fallback settings for a bare install
  templates/project/   data files used by `startproject`
  testing/         harness for writing your own tests
training/          the reference project (sdist only): settings, configs, peers
tests/             141 tests
scripts/sandbox/   install, provision, launch (PowerShell and bash)
docs/              framework.md and the original .bak — never modified
```

---

## Commands

| Command | Purpose |
| --- | --- |
| `check` | Rules, scoring config, aggregation bounds, sandbox capability |
| `doctor` | Python, torch, GPU name/VRAM, QEMU, best isolation level |
| `migrate` | Create/upgrade the run registry |
| `run` | Training loop (`--epochs --device --seed --redteam --dry-run --resume`) |
| `components` | List, inspect, validate, scaffold and diff pluggable training components (scorers, policies, optimizers, strategies, RL functions) |
| `inspect` | `--runs --episodes --rules --checkpoints --json` |
| `shell` | Python shell with settings, rules, scoring, sandbox preloaded |
| `startproject` | Scaffold a runnable project (settings, peers, configs, manage.py) |
| `startpeer` | Scaffold an `nn.Module` peer |
| `sandbox` | `status`, `install`, `provision`, `launch`, `clean` |
| `redteam` | Adversarial pass; asserts P1 voids a score and P20 hits the Ω floor |

---

## Customising components

Every pluggable part of training is a registered *component* of a *kind*, chosen by a
layered spec in `COMPONENTS` / `RL` settings (or `conf/components.yaml`), wrapped in
compliance guards so a custom scorer, policy, optimizer or strategy cannot weaken the
rules, and recorded in checkpoints so a run resumes with the parts it started with. With
nothing configured nothing changes. See `docs/components.md` and
`mactf components --help`.

---

## Sandbox / hardware isolation

Isolation is a **scored property**, not a deployment detail, because P20 and P21
depend on the boundary being real.

```powershell
python training\manage.py sandbox status     # what this host can actually do
python training\manage.py sandbox install    # winget install QEMU (+ WHPX hint)
```

QEMU launches are air-gapped and hardened:

```
-net none                       no network device at all
-snapshot                       discard guest writes
-nographic                      no host display channel
q35 + virtio                    modern machine, paravirtualised disk
fw_cfg cloud-init seed          no genisoimage/cloud-localds on Windows
virtio-serial socket            newline-delimited JSON control channel
```

Acceleration is **probed**, not assumed: `kvm` on Linux, `whpx` on Windows, `tcg`
otherwise. The original design's `-enable-kvm` is not portable to Windows, so it
is never emitted.

Without QEMU the framework degrades to `LocalSandbox` (process isolation), records
the reason, emits an `integrity_incident` signal, and prints it in the run notes.
Set `REQUIRE_HARDWARE: true` or `ALLOW_FALLBACK: false` to refuse that degradation.

**Installing QEMU changes your machine** and WHPX needs elevation plus a reboot.
`manage.py sandbox install` prints the exact `dism` command; it does not run it
silently. The framework is fully functional before you do any of this.

---

## GPU support

GPU is a first-class requirement, not an afterthought:

- `DEVICE` accepts `auto`, `cpu`, `cuda`, `cuda:1`, `mps`
- `DTYPE` accepts `auto`, `float32`, `float16`, `bfloat16`, `float64`; half
  precision on CPU is promoted to `float32` with a recorded note
- `AMP` enables CUDA autocast + `GradScaler` in the local-update phase
- `doctor` reports GPU name, VRAM, and CUDA version
- checkpoints are saved on CPU and restored to the active device
- aggregation runs on flattened float32 CPU vectors, so peers can be spread across
  devices without changing the blend semantics

Verified on an RTX 4070 Laptop (8.6 GB), torch 2.10.0+cu126:

```
device  : cuda
dtype   : float32
amp     : True
gpu     : NVIDIA GeForce RTX 4070 Laptop GPU
```

Run `python scripts\check_cuda.py` to re-verify.

---

## Known limits

- **`sandbox exec` needs the guest agent in the image.** A spec with a
  `serial_socket` gets it installed through cloud-init; without a serial channel,
  or an image lacking the agent, `exec` answers exit code 127 with an explicit
  message. The capability matrix and the 127 case are in
  [docs/sandbox.md](docs/sandbox.md).
- **`PRECEDENCE_MODE: strict` is deliberately strict.** A T3 epistemic failure
  suppresses T5 task rewards entirely. Set `layered` for full visibility; the
  non-offsettable floor still applies either way.
- **`TaskJudge` is task-verifiable only, by design.** It records R1/R2/R14 from
  measured accuracy and calibration and no penalty for an honest wrong answer,
  because framework.md penalizes dishonesty rather than error. Subjective rules
  (R10 synthesis, P13 collusion, P16 loophole) enter through
  `mactf/scoring/evaluator.py` and `POST /evaluations`, the ingest path for
  structured evidence.
- **Trimming drops the dominant peer, not a statistical outlier.** `TRIM_FRACTION`
  is off by default; enabling it is only sensible with ≥3 peers.
- **A `trainer` spec on `POST /runs` is validated and stored, not executed.**
  Orchestrator runs follow the settings; drive `MACTFTrainer` directly for custom
  trainers ([docs/training.md](docs/training.md)).
- **The aggregator cap needs two participants.** A cohort of one cannot satisfy the
  per-peer weight cap, so a hierarchy needs at least two roots.
- **The API serves one process.** Live runs and the event bus are process-local, so
  `--workers` above 1 is ignored.

---

## Tests

```powershell
python -m pytest tests -q
```

141 tests cover catalog integrity against `framework.md`, precedence and the
non-offsettable floor, the docx equations, warning escalation/decay/cap, all four
adapter output styles, floor/cap/participation/fallback aggregation, κ bounds,
sandbox degradation, registry migrations, checkpoint round-trip, a full two-epoch
training run, and packaging (metadata, default settings, scaffolder output).

`docs/framework.md` and `docs/framework.md.bak` are read-only inputs and are never
written by the framework or the tests.

---

## Publishing

The distribution is a standard PEP 517/621 build with a setuptools backend, dynamic
version from `mactf.__version__`, and MIT licensing (PEP 639).

```powershell
pip install -e ".[dev]"
python -m pytest tests -q
python -m build                    # dist/mactf-<version>.whl + .tar.gz
python -m twine check dist/*       # both must say PASSED
python -m twine upload dist/*      # requires a PyPI API token
```

Before uploading:

- set `[project.urls]` in `pyproject.toml` (left commented rather than invented)
- confirm `LICENSE` holds the intended copyright holder
- bump `__version__` in `mactf/__init__.py`; it is the single source of truth
- TestPyPI dry run: `python -m twine upload --repository testpypi dist/*`

Build selection: the wheel ships only `mactf/`, including `py.typed` and the
`startproject` templates. The sdist additionally carries `training/`, `tests/`,
`scripts/`, and `docs/` so the normative rules can be audited from a source
archive. `MANIFEST.in` is the single place that decides this.

`requirements.txt` is a pinned, reproducible development environment, not the
runtime contract; runtime dependencies live in `pyproject.toml`.

