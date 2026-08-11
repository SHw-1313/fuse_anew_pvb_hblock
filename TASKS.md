# Tasks

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`.

Only one task may be `IN_PROGRESS`.

## Phase 0 — Repository and baseline

- [x] T000 Verify `enter-container` and activate `torch-ito`.

  Evidence:
  - Source files: `/usr/local/bin/enter-container`
  - Target files: None
  - Commands: `enter-container`; `conda env list`; `conda activate torch-ito`
  - Tests: Interactive container prompt and environment activation
  - Result: Passed; container prompt is `/workspace` and `torch-ito` activated.
  - Commit: Not applicable

- [x] T001 Record Python, PyTorch, CUDA, GPU, torch-scatter and xFormers versions.

  Evidence:
  - Source files: None
  - Target files: `HANDOFF.md`
  - Commands: `python --version`; PyTorch/CUDA probe; import-spec probe; `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`
  - Tests: Environment probe
  - Result: Python 3.11.15; PyTorch 2.5.1+cu121; CUDA 12.1; 8× NVIDIA A100-SXM4-80GB; torch_scatter installed; xFormers absent.
  - Commit: Not applicable

- [x] T002 Record PVB and Anew commit SHAs and worktree status.

  Evidence:
  - Source files: `/workspace/PVB/.git`; `/workspace/AnewOmni/.git`
  - Target files: `HANDOFF.md`; `DECISIONS.md`
  - Commands: `git -C /workspace/PVB rev-parse HEAD`; `git -C /workspace/AnewOmni rev-parse HEAD`; `git -C ... status --porcelain=v1`
  - Tests: Clean-worktree check
  - Result: PVB `c08e5e3cd49d45c6d748387e78224843bd356f50`; AnewOmni `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`; both clean.
  - Commit: Not applicable

- [x] T003 Confirm both source repositories remain read-only.

  Evidence:
  - Source files: `/workspace/PVB`; `/workspace/AnewOmni`
  - Target files: None
  - Commands: Source worktree status before bootstrap
  - Tests: No source modifications observed
  - Result: Passed; all project writes are directed to the target.
  - Commit: Not applicable

- [x] T004 Bootstrap target from `/workspace/PVB`, excluding runtime artifacts.

  Evidence:
  - Source files: `/workspace/PVB`
  - Target files: PVB source tree in `/workspace/fuse_anew_pvb_hblock`
  - Commands: `rsync -a` with exclusions for `.git`, caches, checkpoints, datasets, logs, results, outputs, wandb, ckpt, and virtual environments
  - Tests: Target tree inspection
  - Result: Passed; target was inspected first and populated from PVB without copying its `.git` or excluded runtime artifacts.
  - Commit: Pending bootstrap commit

- [x] T005 Initialize target git history without copying PVB’s `.git`.

  Evidence:
  - Source files: None copied from PVB `.git`
  - Target files: `/workspace/fuse_anew_pvb_hblock/.git`
  - Commands: `git -C /workspace/fuse_anew_pvb_hblock init -b main`
  - Tests: `git status --short --branch`
  - Result: Passed; independent `main` repository initialized.
  - Commit: Pending bootstrap commit

- [x] T006 Create `PLAN.md`, `TASKS.md`, `HANDOFF.md`, and `DECISIONS.md`.

  Evidence:
  - Source files: User-provided planning specification
  - Target files: `PLAN.md`; `TASKS.md`; `HANDOFF.md`; `DECISIONS.md`
  - Commands: `apply_patch`
  - Tests: File existence and content inspection
  - Result: Passed; live planning, task, handoff, and decision records created.
  - Commit: Pending bootstrap commit

- [x] T007 Run the unmodified PVB test/import smoke suite.

  Evidence:
  - Source files: PVB `module/`, `data/`, `trainer/`, `train.py`, `infer_prot.py`
  - Target files: unchanged copied PVB tree
  - Commands: `python -m compileall -q .`; core PVB import probe; `python train.py --help`; `python infer_prot.py --help`
  - Tests: Bytecode compilation, imports, training CLI help, inference CLI help
  - Result: Passed; no upstream test suite exists in the copied repository, and all available baseline smoke checks succeeded.
  - Commit: Pending bootstrap commit

- [x] T008 Add `scripts/profile_train_step.py`.

  Evidence:
  - Source files: PVB `module/model.py`; PVB `module/graph.py`; PVB `utils/bio_utils.py`
  - Target files: `scripts/profile_train_step.py`; `scripts/__init__.py`
  - Commands: `python -m compileall -q scripts`; `python -m scripts.profile_train_step --steps 1 --atoms 16 --samples 2 --hidden-dim 32 --ffn-dim 64 --layers 1 --heads 4 --rbf-dim 8 --k-neighbors 8 --device cuda:0 --using-ode`
  - Tests: Synthetic PVB `_train` forward/backward smoke
  - Result: Passed; finite loss and gradients with forward/backward/peak-memory measurements.
  - Commit: Pending bootstrap commit

- [x] T009 Add `scripts/overfit_one_batch.py`.

  Evidence:
  - Source files: PVB `module/model.py`; PVB graph and bridge path
  - Target files: `scripts/overfit_one_batch.py`
  - Commands: `python -m scripts.overfit_one_batch --steps 10 --atoms 16 --samples 2 --device cuda:0 --using-ode`
  - Tests: Deterministic one-batch regression
  - Result: Passed; loss decreased from `4.601564` to `0.907342` with finite gradients.
  - Commit: Pending bootstrap commit

- [x] T010 Record baseline time, memory, loss, and gradient results.

  Evidence:
  - Source files: PVB commit `c08e5e3cd49d45c6d748387e78224843bd356f50`
  - Target files: `HANDOFF.md`
  - Commands: `python -m scripts.profile_train_step --steps 3 --atoms 64 --samples 2 --device cuda:0`
  - Tests: Full PVB-dimension synthetic profile
  - Result: Finite losses `[43.899323, 42.206913, 83.708847]`; excluding first-step warmup, forward `0.02282 s`, backward `0.02611 s`, step `0.04893 s`; peak `697,924,608` bytes; gradients finite with norms `42,723.9`, `50,820.4`, `79,491.6`.
  - Commit: Pending bootstrap commit

Gate P0: stop if the copied PVB baseline is not reproducible.

## Phase 1 — Reuse Anew implementation

- [ ] T100 Create `third_party/anewomni`. **IN_PROGRESS**
- [ ] T101 Copy Anew license.
- [ ] T102 Copy `models/modules/EPT/ept.py`.
- [ ] T103 Copy `models/modules/EPT/radial_basis.py`.
- [ ] T104 Copy `models/modules/GET/tools.py`.
- [ ] T105 Copy `models/modules/nn.py`.
- [ ] T106 Copy required `utils/nn_utils.py`.
- [ ] T107 Copy required `utils/gnn_utils.py`.
- [ ] T108 Copy `utils/register.py` or remove registration with a documented minimal patch.
- [ ] T109 Convert copied imports to package-relative imports.
- [ ] T110 Record every copied source and SHA in `DECISIONS.md`.
- [ ] T111 Add import tests for the vendored package.
- [ ] T112 Add parity tests against the corresponding Anew source implementation.

Gate P1: vendored EPT and pooling must match the source implementation.

## Phase 2 — Block data contract

- [ ] T200 Inspect every PVB dataset used by the requested training stage.
- [ ] T201 Define explicit block metadata schema.
- [ ] T202 Add preprocessing support for local block IDs.
- [ ] T203 Make graph cropping preserve complete residue blocks.
- [ ] T204 Offset block IDs correctly in `data/collate.py`.
- [ ] T205 Produce `atom_block_id`, `block_type`, `block_batch`, `block_lengths`.
- [ ] T206 Add compatibility conversion for old datasets, on CPU only.
- [ ] T207 Add block ordering, length, batch-isolation, and repeated-residue tests.
- [ ] T208 Reject unsupported ligand/molecular block mappings clearly.

Gate P2: no model code may infer block membership from `b0` coordinate equality.

## Phase 3 — Anew block encoder

- [ ] T300 Add `module/anew_block_encoder.py`.
- [ ] T301 Reuse Anew `BlockEmbedding`.
- [ ] T302 Reuse Anew EPT.
- [ ] T303 Reuse Anew edge utilities.
- [ ] T304 Adapt the encoding sequence from Anew `model_edge.py`.
- [ ] T305 Implement Anew-equivalent `H_atom → H_block` pooling.
- [ ] T306 Implement `X_atom → X_block` pooling.
- [ ] T307 Return a typed/documented output dictionary.
- [ ] T308 Keep PVB coordinate mean equal to `x0`.
- [ ] T309 Ensure `X_atom`/`X_block` are diagnostic-only in milestone one.
- [ ] T310 Remove unused PVB encoder-edge construction in `anew_block` mode.
- [ ] T311 Add shape, pooling, batch-isolation, gradient, and SE(3) tests.

Gate P3: block outputs must be correct before modifying the decoder.

## Phase 4 — PVB decoder conditioning

- [ ] T400 Add fusion configuration with `off` and `anew_block`.
- [ ] T401 Add block projection `LayerNorm(512) → Linear(512, hidden_dim)`.
- [ ] T402 Add scalar zero-initialized `block_gate`.
- [ ] T403 Broadcast block features with `H_block[atom_block_id]`.
- [ ] T404 Inject conditioning into the PVB decoder input.
- [ ] T405 Apply identical conditioning to both cross-attention branches.
- [ ] T406 Preserve `fusion.mode=off` behavior.
- [ ] T407 Add gate-zero decoder parity test.
- [ ] T408 Verify nonzero projector and gate gradients.
- [ ] T409 Run one-batch overfit for `off` and `anew_block`.

Gate P4: do not proceed if gate-zero parity or overfit fails.

## Phase 5 — Checkpoints

- [ ] T500 Replace whole-object checkpoint loading with state-dict loading.
- [ ] T501 Implement PVB key migration.
- [ ] T502 Implement Anew key migration.
- [ ] T503 Implement full fused resume.
- [ ] T504 Print key coverage and mismatch reports.
- [ ] T505 Add configurable minimum coverage thresholds.
- [ ] T506 Add checkpoint-loading tests with synthetic and real checkpoints.

## Phase 6 — Training and performance

- [ ] T600 Add frozen-Anew training stage.
- [ ] T601 Add separate optimizer parameter groups.
- [ ] T602 Add selective EPT unfreezing.
- [ ] T603 Log per-module gradient norms and gate value.
- [ ] T604 Profile 256/512/1024/2000-atom batches.
- [ ] T605 Separate graph, encoder, decoder, and backward timings.
- [ ] T606 Add BF16 autocast behind a config flag.
- [ ] T607 Add attention-aware batch budgeting.
- [ ] T608 Compare PVB, legacy fusion if retained, and block fusion.
- [ ] T609 Decide whether a separate block-sparse encoder is necessary.

## Phase 7 — Final verification

- [ ] T700 Run full unit tests.
- [ ] T701 Run protein training smoke test.
- [ ] T702 Run protein inference smoke test.
- [ ] T703 Confirm source repositories are unchanged.
- [ ] T704 Confirm no runtime dependency on sibling repositories.
- [ ] T705 Update all four planning/handoff files.
- [ ] T706 Record final changed-file list and benchmark table.
