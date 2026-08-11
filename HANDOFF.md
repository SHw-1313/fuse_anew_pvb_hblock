# Handoff

## Current state

- Current phase: Phase 7 — final verification complete
- Current task: T706 — final changed-file and benchmark record
- Status: DONE; Phases 0–7 gates passed
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
- Commit: `62ceeb4` implementation; current `HEAD` also contains the documentation-only finalization commit
- Working tree status: clean; target has the vendored Anew path, explicit block metadata, fused encoder/decoder, checkpoint migration, staged training, benchmarks, tests, and live docs; source repositories remain clean

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
- T100–T112 — vendored the minimal Anew implementation and passed source parity.
- T200–T208 — added explicit block metadata, block-safe cropping, and protein-only rejection.
- T300–T311 — added the faithful Anew H-block encoder and passed pooling/SE(3)/gradient tests.
- T400–T409 — added zero-gated decoder conditioning and passed parity/gradient/overfit gates.
- T500–T506 — added explicit PVB/Anew/resume checkpoint migration, coverage reports, and trainer state saving.
- T600–T609 — added staged freezing, optimizer groups, BF16 option, n*n budgeting, and four-scale profiles.
- T700–T706 — passed the focused target suite, protein train/inference smoke tests, source/dependency audits, and final changed-file/benchmark review.

## In-progress task

- Task ID: None; all planned tasks are complete.
- Intended outcome: None.
- Source files being reused: None.
- Target files: None.
- Partial implementation: None.
- Remaining work: None for the first protein-only milestone.

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

# Phase 1–4 validation
python -m unittest -v tests.test_anew_vendor_parity tests.test_block_metadata tests.test_anew_block_encoder tests.test_fusion
python -m compileall -q module data tests third_party train.py
python -m scripts.overfit_one_batch --device cuda --atoms 16 --samples 2 --steps 5 --hidden-dim 32 --ffn-dim 64 --rbf-dim 8 --heads 4 --layers 2 --k-neighbors 8
python -m scripts.overfit_one_batch --device cuda --atoms 16 --samples 2 --steps 5 --hidden-dim 32 --ffn-dim 64 --rbf-dim 8 --heads 4 --layers 2 --k-neighbors 8 --fusion-mode anew_block
python -m unittest -v tests.test_checkpoints tests.test_training_stages
python -m scripts.profile_components --device cuda --atoms 256 --samples 1 --steps 2
python -m scripts.profile_components --device cuda --atoms 256 --samples 1 --steps 2 --fusion-mode anew_block
python -m unittest discover -s tests -v
python -m compileall -q .
python -m scripts.protein_smoke --device cuda --fusion-mode off
python -m scripts.protein_smoke --device cuda --fusion-mode anew_block
python train.py --help
python infer_prot.py --help
```

## Validation results

| Test or benchmark | Command | Result | Notes |
| --- | --- | --- | --- |
| PVB baseline | `python -m compileall -q .`; import/CLI smoke | Passed | Unmodified copied PVB tree |
| Vendor parity | `python -m unittest -v tests.test_anew_vendor_parity` | Passed | 2 source parity tests |
| Block metadata | `python -m unittest -v tests.test_block_metadata` | Passed | 4 metadata/cropping tests |
| Gate-zero parity | `python -m unittest -v tests.test_fusion` | Passed | decoder parity within `1e-6` |
| One-batch overfit | `python -m scripts.overfit_one_batch ...` | Passed | off `29.2191 → 3.1783`; fused `11.6374 → 0.8612` |
| SE(3) | `python -m unittest -v tests.test_anew_block_encoder` | Passed | atom state/coordinate equivariance |
| Checkpoint migration | `python -m unittest -v tests.test_checkpoints` | Passed | PVB/Anew/resume/coverage fixtures |
| Staged training | `python -m unittest -v tests.test_training_stages` | Passed | Stage A/B, groups, diagnostics, n*n budget |
| BF16 smoke | CUDA autocast forward/backward probe | Passed | finite fused loss and gradients |
| Training smoke | `python -m scripts.protein_smoke --device cuda --fusion-mode off`; `... --fusion-mode anew_block` | Passed | Synthetic protein train step; losses `13.7539` and `4.1669`, finite gradients |
| Inference smoke | Same protein smoke commands | Passed | Both modes returned finite `[16, 3]` generated coordinates |
| Focused target suite | `python -m unittest discover -s tests -v` | Passed | 22 tests in 15.297 s |
| Compile/import/CLI | `python -m compileall -q .`; `python train.py --help`; `python infer_prot.py --help` | Passed | Target compiles and both entrypoints import |

## Performance results

Stable post-warmup step from `scripts/profile_components.py` on one A100; graph/encoder/decoder values are forward components.

| Mode | Atoms | Graph | Encoder | Decoder | Forward | Backward | Step | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PVB baseline | 64 | — | — | — | 0.02282 s | 0.02611 s | 0.04893 s | 697,924,608 B |
| PVB baseline | 256 | 0.03172 s | 0.00497 s | 0.01953 s | 0.08040 s | 0.06572 s | 0.14612 s | 2,361,583,616 B |
| PVB baseline | 512 | 0.02972 s | 0.02405 s | 0.04456 s | 0.10814 s | 0.08315 s | 0.19129 s | 4,593,067,008 B |
| PVB baseline | 1024 | 0.02801 s | 0.01021 s | 0.05976 s | 0.12035 s | 0.12455 s | 0.24490 s | 9,034,408,960 B |
| PVB baseline | 2000 | 0.00789 s | 0.01513 s | 0.04912 s | 0.07482 s | 0.09180 s | 0.16661 s | 17,511,942,656 B |
| Anew H-block | 256 | 0.03078 s | 0.10455 s | 0.04235 s | 0.18486 s | 0.07258 s | 0.25744 s | 2,610,635,776 B |
| Anew H-block | 512 | 0.00359 s | 0.03939 s | 0.01917 s | 0.06556 s | 0.07295 s | 0.13851 s | 4,715,552,256 B |
| Anew H-block | 1024 | 0.00272 s | 0.02794 s | 0.02646 s | 0.05959 s | 0.08203 s | 0.14162 s | 8,943,741,440 B |
| Anew H-block | 2000 | 0.00326 s | 0.05010 s | 0.04776 s | 0.10420 s | 0.15587 s | 0.26007 s | 17,345,196,544 B |

## Files changed

* `PLAN.md`
* `TASKS.md`
* `HANDOFF.md`
* `DECISIONS.md`
* PVB baseline files copied into the target from the clean source revision
* `third_party/anewomni/`
* `data/block_metadata.py`
* `module/anew_block_encoder.py`
* `utils/checkpoint.py`
* `utils/fusion_training.py`
* tests and profiling scripts

## Decisions added or changed

* D001–D021; D017 records the vendored source map, D018 the verified vocabulary mapping, D019 target-local NumPy compatibility, D020 shared decoder gating, and D021 n*n batching/performance decision.

## Known problems

* xFormers is not installed in `torch-ito`; it is optional and not required for the faithful baseline.
* Root-level `python -m unittest discover -v` also discovers two unrelated optional PVB packages: `ept.models` expects its legacy `utils.register` import layout and `simulation` requires `openmm`. The authoritative target suite is `python -m unittest discover -s tests -v`, which passes all 22 fused-repository tests.
* The copied optional PVB converters/simulation retain their original local `sys.path.append('..')` compatibility lines; no target runtime code imports `/workspace/PVB` or `/workspace/AnewOmni`, and Anew parity tests invoke the source only in isolated subprocesses.

## Blockers

* None yet.

## Exact next action

1. Run `git log -1 --oneline` in `/workspace/fuse_anew_pvb_hblock` to verify the final local target revision; expected result is a clean documentation-finalized `main` commit after `62ceeb4`.

## Resume instructions

1. Enter the container.
2. Activate `torch-ito`.
3. Read `PLAN.md`.
4. Read `DECISIONS.md`.
5. Read this file.
6. Continue the single `IN_PROGRESS` task in `TASKS.md`.
7. Do not redo completed tasks unless their evidence is invalid.
