# Handoff

## Current state

- Current phase: Phase 1 — reuse Anew implementation
- Current task: T100 — create `third_party/anewomni`
- Status: IN_PROGRESS; Phase 0 gate passed
- Last updated: 2026-08-11
- Agent/model: Codex / GPT-5
- Container: `sihao-dev`, entered with `enter-container`
- Conda environment: torch-ito

## Repository locations

- Read-only PVB: `/workspace/PVB`
- Read-only AnewOmni: `/workspace/AnewOmni`
- Writable target: `/workspace/fuse_anew_pvb_hblock`

## Source revisions

| Source | Commit SHA | Dirty before work? | Modified by this task? |
|---|---|---:|---:|
| PVB | `c08e5e3cd49d45c6d748387e78224843bd356f50` | No | Must remain no |
| AnewOmni | `926e99818ea18cf9d9b2064ce0319fe691b7a1f1` | No | Must remain no |

## Target revision

- Branch: `main`
- Commit: Pending initial bootstrap commit
- Working tree status: PVB baseline, planning files, and Phase 0 diagnostic scripts ready to commit

## Completed tasks

- T000 — verified container entry and activated `torch-ito`.
- T001 — recorded Python, PyTorch, CUDA, GPU, torch-scatter, and xFormers status.
- T002 — recorded clean source revisions.
- T003 — confirmed source repositories are untouched.
- T004 — bootstrapped target from PVB with runtime-artifact exclusions.
- T005 — initialized an independent target Git repository.
- T006 — created the four live planning documents.
- T007 — passed the unchanged PVB import/CLI smoke suite.
- T008 — added the PVB train-step profiling script.
- T009 — added the deterministic one-batch overfit script.
- T010 — recorded a full-dimension 64-atom baseline profile.

## In-progress task

- Task ID: T100
- Intended outcome: vendor only the minimal Anew implementation required for faithful BlockEmbedding/EPT/edge/pooling reuse.
- Source files being reused: AnewOmni `models/IterVAE/model_edge.py`, `models/modules/nn.py`, `models/modules/EPT/`, `models/modules/GET/tools.py`, and required utility files.
- Target files: `third_party/anewomni/` and parity tests.
- Partial implementation: PVB baseline is reproducible and committed next; no Anew code has been copied yet.
- Remaining work: inspect Anew imports and source semantics, copy complete files with headers, namespace imports, and add parity checks. Stop if the actual source architecture differs materially from the plan.

## Commands already run

```bash
# Host/workspace audit
pwd
ls -la
command -v enter-container

# In sihao-dev, with torch-ito active
conda env list
conda activate torch-ito
python --version
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
python -c 'import importlib.util; print(importlib.util.find_spec("torch_scatter")); print(importlib.util.find_spec("xformers"))'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Source provenance
git -C /workspace/PVB rev-parse HEAD
git -C /workspace/PVB status --porcelain=v1
git -C /workspace/AnewOmni rev-parse HEAD
git -C /workspace/AnewOmni status --porcelain=v1

# Target bootstrap
rsync -a --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.cache/' --exclude='checkpoints/' --exclude='checkpoint/' --exclude='datasets/' --exclude='logs/' --exclude='results/' --exclude='outputs/' --exclude='wandb/' --exclude='ckpt/' --exclude='.venv/' --exclude='venv/' /workspace/PVB/ /workspace/fuse_anew_pvb_hblock/
git -C /workspace/fuse_anew_pvb_hblock init -b main
```

## Validation results

| Test or benchmark | Command | Result | Notes |
| --- | --- | --- | --- |
| PVB baseline | `python -m compileall -q .`; import/CLI smoke | Passed | Unmodified copied PVB tree |
| Vendor parity | Pending | Pending | |
| Block metadata | Pending | Pending | |
| Gate-zero parity | Pending | Pending | |
| One-batch overfit | Pending | Pending | |
| SE(3) | Pending | Pending | |
| Training smoke | Pending | Pending | |
| Inference smoke | Pending | Pending | |

## Performance results

| Mode | Atoms | Forward | Backward | Step | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| PVB baseline | 64 | 0.02282 s | 0.02611 s | 0.04893 s | 697,924,608 bytes |
| Legacy fusion | Pending | Pending | Pending | Pending | Pending |
| Anew H-block | Pending | Pending | Pending | Pending | Pending |

## Files changed

* `PLAN.md`
* `TASKS.md`
* `HANDOFF.md`
* `DECISIONS.md`
* PVB baseline files copied into the target from the clean source revision

## Decisions added or changed

* D001–D015 added from the requested architecture and stop-gate specification.

## Known problems

* xFormers is not installed in `torch-ito`; it is optional and not required for the faithful baseline.

## Blockers

* None yet.

## Exact next action

1. Inspect the target’s test files, import paths, README, and training entrypoint to identify the minimal unchanged PVB smoke suite.
2. Run that suite in `/workspace/fuse_anew_pvb_hblock` with `torch-ito` active.
3. If it fails, classify the failure as a reproducibility blocker and stop before Anew integration; if it passes, record evidence and commit the bootstrap baseline.

## Resume instructions

1. Enter the container.
2. Activate `torch-ito`.
3. Read `PLAN.md`.
4. Read `DECISIONS.md`.
5. Read this file.
6. Continue the single `IN_PROGRESS` task in `TASKS.md`.
7. Do not redo completed tasks unless their evidence is invalid.
