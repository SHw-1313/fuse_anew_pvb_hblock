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

- [x] T100 Create `third_party/anewomni`.

  Evidence:
  - Source files: Anew `models/modules/`, `utils/`, and `LICENSE`
  - Target files: `third_party/anewomni/`
  - Commands: complete-file vendor from Anew SHA `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`
  - Tests: `python -m unittest -v tests.test_anew_vendor_parity`
  - Result: Passed; namespaced vendored package imports without the sibling repository.
  - Commit: Pending phase commit

- [x] T101 Copy Anew license.

  Evidence:
  - Source files: `AnewOmni/LICENSE`
  - Target files: `third_party/anewomni/LICENSE`
  - Commands: complete-file copy
  - Tests: source/target license inspection
  - Result: Passed; license retained.
  - Commit: Pending phase commit

- [x] T102 Copy `models/modules/EPT/ept.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/EPT/ept.py`
  - Target files: `third_party/anewomni/models/modules/EPT/ept.py`
  - Commands: complete-file copy with import-only namespace patch
  - Tests: EPT numerical parity test
  - Result: Passed; output matches the source implementation.
  - Commit: Pending phase commit

- [x] T103 Copy `models/modules/EPT/radial_basis.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/EPT/radial_basis.py`
  - Target files: `third_party/anewomni/models/modules/EPT/radial_basis.py`
  - Commands: complete-file copy
  - Tests: vendored EPT parity subprocess
  - Result: Passed; source implementation reused.
  - Commit: Pending phase commit

- [x] T104 Copy `models/modules/GET/tools.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/GET/tools.py`
  - Target files: `third_party/anewomni/models/modules/GET/tools.py`
  - Commands: complete-file copy
  - Tests: vendored encoder forward
  - Result: Passed; Anew block/unit edge utility is available locally.
  - Commit: Pending phase commit

- [x] T105 Copy `models/modules/nn.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/nn.py`
  - Target files: `third_party/anewomni/models/modules/nn.py`
  - Commands: complete-file copy
  - Tests: BlockEmbedding numerical parity test
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T106 Copy required `utils/nn_utils.py`.

  Evidence:
  - Source files: `AnewOmni/utils/nn_utils.py`
  - Target files: `third_party/anewomni/utils/nn_utils.py`
  - Commands: complete-file copy
  - Tests: vendored EPT import and encoder tests
  - Result: Passed; stable normalization/batching dependency is present.
  - Commit: Pending phase commit

- [x] T107 Copy required `utils/gnn_utils.py`.

  Evidence:
  - Source files: `AnewOmni/utils/gnn_utils.py`
  - Target files: `third_party/anewomni/utils/gnn_utils.py`
  - Commands: complete-file copy
  - Tests: Anew pooling formula test
  - Result: Passed; variance-preserving pooling matches reference semantics.
  - Commit: Pending phase commit

- [x] T108 Copy `utils/register.py` or remove registration with a documented minimal patch.

  Evidence:
  - Source files: `AnewOmni/utils/register.py`
  - Target files: `third_party/anewomni/utils/register.py`
  - Commands: complete-file copy
  - Tests: namespaced imports
  - Result: Passed; registration dependency is isolated under the vendored namespace.
  - Commit: Pending phase commit

- [x] T109 Convert copied imports to package-relative imports.

  Evidence:
  - Source files: vendored EPT imports
  - Target files: `third_party/anewomni/models/modules/EPT/ept.py`
  - Commands: changed only sibling utility imports to relative imports
  - Tests: source/target subprocess parity without `PYTHONPATH` or `sys.path` injection
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T110 Record every copied source and SHA in `DECISIONS.md`.

  Evidence:
  - Source files: pinned Anew source tree
  - Target files: `DECISIONS.md`
  - Commands: source path/SHA audit
  - Tests: source worktree status
  - Result: Passed; D017 records the complete vendor map.
  - Commit: Pending phase commit

- [x] T111 Add import tests for the vendored package.

  Evidence:
  - Source files: vendored package modules
  - Target files: `tests/test_anew_vendor_parity.py`
  - Commands: `python -m unittest -v tests.test_anew_vendor_parity`
  - Tests: import and numerical parity tests
  - Result: Passed; 2 tests.
  - Commit: Pending phase commit

- [x] T112 Add parity tests against the corresponding Anew source implementation.

  Evidence:
  - Source files: Anew `BlockEmbedding` and `XTransEncoderAct`
  - Target files: `tests/test_anew_vendor_parity.py`
  - Commands: `python -m unittest -v tests.test_anew_vendor_parity`
  - Tests: source-vendored output comparisons
  - Result: Passed; numerical parity confirmed.
  - Commit: Pending phase commit

Gate P1: PASSED; vendored EPT, embedding, edge, and pooling dependencies match the source implementation.

## Phase 2 — Block data contract

- [x] T200 Inspect every PVB dataset used by the requested training stage.

  Evidence:
  - Source files: `data/*_dataset.py`, `data/subgraph.py`, `data/collate.py`
  - Target files: same PVB paths
  - Commands: graph-crop and block-field call-site audit
  - Tests: all relevant graph crop paths reviewed
  - Result: Passed; protein and complex preprocessors were identified and updated where applicable.
  - Commit: Pending phase commit

- [x] T201 Define explicit block metadata schema.

  Evidence:
  - Source files: PVB item/collate contracts
  - Target files: `data/block_metadata.py`
  - Commands: schema implementation
  - Tests: metadata contract tests
  - Result: Passed; schema is `atom_block_id`, `block_type`, `block_batch`, `block_lengths`.
  - Commit: Pending phase commit

- [x] T202 Add preprocessing support for local block IDs.

  Evidence:
  - Source files: `data/mmap_dataset.py`, `data/data_init.py`
  - Target files: same paths; `data/block_metadata.py`
  - Commands: annotate records during preprocessing and batch creation
  - Tests: explicit preprocessing annotation test
  - Result: Passed; local IDs are materialized before model forward.
  - Commit: Pending phase commit

- [x] T203 Make graph cropping preserve complete residue blocks.

  Evidence:
  - Source files: PVB graph crop call sites
  - Target files: `data/subgraph.py`, protein/complex dataset preprocessors
  - Commands: pass explicit `block_id` to `graph_cut`
  - Tests: partial outer block expansion test
  - Result: Passed; selected blocks are complete.
  - Commit: Pending phase commit

- [x] T204 Offset block IDs correctly in `data/collate.py`.

  Evidence:
  - Source files: `data/collate.py`
  - Target files: `data/collate.py`, `data/block_metadata.py`
  - Commands: global offset implementation
  - Tests: two-sample offset test
  - Result: Passed; IDs remain contiguous and sample-isolated.
  - Commit: Pending phase commit

- [x] T205 Produce `atom_block_id`, `block_type`, `block_batch`, `block_lengths`.

  Evidence:
  - Source files: PVB collate/data-init paths
  - Target files: `data/collate.py`, `data/data_init.py`
  - Commands: explicit metadata update in returned batches
  - Tests: metadata contract test
  - Result: Passed; all four fields are emitted for explicit and legacy records.
  - Commit: Pending phase commit

- [x] T206 Add compatibility conversion for old datasets, on CPU only.

  Evidence:
  - Source files: legacy `b0`/`btype` records
  - Target files: `data/block_metadata.py`
  - Commands: CPU fallback derivation with warning
  - Tests: legacy warning/fallback test
  - Result: Passed; fallback is outside model GPU forward.
  - Commit: Pending phase commit

- [x] T207 Add block ordering, length, batch-isolation, and repeated-residue tests.

  Evidence:
  - Source files: metadata/collate implementation
  - Target files: `tests/test_block_metadata.py`
  - Commands: `python -m unittest -v tests.test_block_metadata`
  - Tests: ordering, lengths, offsets, repeated residue IDs, and crop behavior
  - Result: Passed; 4 tests.
  - Commit: Pending phase commit

- [x] T208 Reject unsupported ligand/molecular block mappings clearly.

  Evidence:
  - Source files: verified PVB/Anew vocabularies
  - Target files: `module/anew_block_encoder.py`
  - Commands: explicit protein vocabulary assertion
  - Tests: non-protein block rejection test
  - Result: Passed; unsupported element/ligand blocks raise a clear protein-only error.
  - Commit: Pending phase commit

Gate P2: PASSED; model code consumes explicit integer metadata and does not infer blocks from `b0` equality.

## Phase 3 — Anew block encoder

- [x] T300 Add `module/anew_block_encoder.py`.

  Evidence:
  - Source files: Anew `models/IterVAE/model_edge.py`
  - Target files: `module/anew_block_encoder.py`
  - Commands: implementation and compileall
  - Tests: `tests.test_anew_block_encoder`
  - Result: Passed; explicit-metadata encoder is available.
  - Commit: Pending phase commit

- [x] T301 Reuse Anew `BlockEmbedding`.

  Evidence:
  - Source files: Anew `models/modules/nn.py`
  - Target files: `module/anew_block_encoder.py`, `third_party/anewomni/models/modules/nn.py`
  - Commands: direct vendored import
  - Tests: vendor parity and encoder shape tests
  - Result: Passed; no replacement embedding was reimplemented.
  - Commit: Pending phase commit

- [x] T302 Reuse Anew EPT.

  Evidence:
  - Source files: Anew `models/modules/EPT/ept.py`, `radial_basis.py`
  - Target files: `module/anew_block_encoder.py`, `third_party/anewomni/models/modules/EPT/`
  - Commands: direct vendored import
  - Tests: EPT parity and finite-gradient tests
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T303 Reuse Anew edge utilities.

  Evidence:
  - Source files: Anew `models/modules/GET/tools.py`
  - Target files: `module/anew_block_encoder.py`, vendored GET package
  - Commands: `knn_edges` and vendored edge embedding path
  - Tests: encoder batch-isolation test
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T304 Adapt the encoding sequence from Anew `model_edge.py`.

  Evidence:
  - Source files: Anew `models/IterVAE/model_edge.py`
  - Target files: `module/anew_block_encoder.py`
  - Commands: BlockEmbedding → EPT → atom/block pooling adaptation
  - Tests: encoder forward and SE(3) tests
  - Result: Passed; source sequence preserved with explicit PVB metadata.
  - Commit: Pending phase commit

- [x] T305 Implement Anew-equivalent `H_atom → H_block` pooling.

  Evidence:
  - Source files: Anew `utils/gnn_utils.py`
  - Target files: `module/anew_block_encoder.py`
  - Commands: `std_conserve_scatter_mean` reuse
  - Tests: exact scatter-sum/sqrt-length comparison
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T306 Implement `X_atom → X_block` pooling.

  Evidence:
  - Source files: Anew `model_edge.py` coordinate pooling
  - Target files: `module/anew_block_encoder.py`
  - Commands: `scatter_mean` pooling
  - Tests: exact block coordinate mean comparison
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T307 Return a typed/documented output dictionary.

  Evidence:
  - Source files: requested architecture contract
  - Target files: `module/anew_block_encoder.py`
  - Commands: documented `Dict[str, Tensor]` forward
  - Tests: key and shape assertions
  - Result: Passed; all eight required output keys are returned.
  - Commit: Pending phase commit

- [x] T308 Keep PVB coordinate mean equal to `x0`.

  Evidence:
  - Source files: PVB `module/model.py` bridge contract
  - Target files: `module/model.py`, `tests/test_fusion.py`
  - Commands: fused `_encode_anew_block` keeps `x_mu = x.clone()`
  - Tests: zero-latent-noise source-mean test
  - Result: Passed; Anew coordinate output is not substituted.
  - Commit: Pending phase commit

- [x] T309 Ensure `X_atom`/`X_block` are diagnostic-only in milestone one.

  Evidence:
  - Source files: Anew encoder output contract
  - Target files: `module/model.py`, `module/anew_block_encoder.py`
  - Commands: no X output used in PVB bridge or decoder condition
  - Tests: fused source-mean and output-key tests
  - Result: Passed; X outputs are returned only for diagnostics.
  - Commit: Pending phase commit

- [x] T310 Remove unused PVB encoder-edge construction in `anew_block` mode.

  Evidence:
  - Source files: PVB `module/model.py`, `module/graph.py`
  - Target files: `module/model.py`
  - Commands: branch encoder graph construction on `fusion_mode`
  - Tests: fused one-batch forward/backward
  - Result: Passed; fused paths do not build PVB encoder edges.
  - Commit: Pending phase commit

- [x] T311 Add shape, pooling, batch-isolation, gradient, and SE(3) tests.

  Evidence:
  - Source files: encoder and metadata contracts
  - Target files: `tests/test_anew_block_encoder.py`, `tests/test_fusion.py`
  - Commands: `python -m unittest -v tests.test_anew_block_encoder tests.test_fusion`
  - Tests: 8 encoder/fusion tests
  - Result: Passed; shapes, pooling, isolation, gradients, SE(3), and source mean verified.
  - Commit: Pending phase commit

Gate P3: PASSED; block outputs and coordinate contract were verified before decoder conditioning.

## Phase 4 — PVB decoder conditioning

- [x] T400 Add fusion configuration with `off` and `anew_block`.

  Evidence:
  - Source files: PVB `train.py`, `config/train.yaml`
  - Target files: `train.py`, `config/train.yaml`, `module/model.py`
  - Commands: config/model construction audit
  - Tests: off and fused model instantiation
  - Result: Passed; `fusion.mode` supports both paths.
  - Commit: Pending phase commit

- [x] T401 Add block projection `LayerNorm(512) → Linear(512, hidden_dim)`.

  Evidence:
  - Source files: requested architecture contract
  - Target files: `module/model.py`
  - Commands: `nn.Sequential(LayerNorm, Linear)` construction
  - Tests: projection gradient test
  - Result: Passed; default Anew hidden size is 512 and output is PVB hidden size.
  - Commit: Pending phase commit

- [x] T402 Add scalar zero-initialized `block_gate`.

  Evidence:
  - Source files: requested zero-gating contract
  - Target files: `module/model.py`
  - Commands: `nn.Parameter(torch.zeros(1))`
  - Tests: gate-zero parity test
  - Result: Passed; initial gate is exactly zero.
  - Commit: Pending phase commit

- [x] T403 Broadcast block features with `H_block[atom_block_id]`.

  Evidence:
  - Source files: explicit block metadata contract
  - Target files: `module/model.py`
  - Commands: project block states then `index_select` by global atom block ID
  - Tests: metadata and fused gradient tests
  - Result: Passed; no coordinate-based membership lookup is used.
  - Commit: Pending phase commit

- [x] T404 Inject conditioning into the PVB decoder input.

  Evidence:
  - Source files: PVB `module/torchmd_et.py`
  - Target files: `module/model.py`, `module/torchmd_et.py`
  - Commands: add gated condition to decoder branch features
  - Tests: gate-zero and fused overfit tests
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T405 Apply identical conditioning to both cross-attention branches.

  Evidence:
  - Source files: PVB TorchMD cross-attention `x0`/`xt` branches
  - Target files: `module/torchmd_et.py`
  - Commands: condition added once to each branch before neighbor embedding
  - Tests: decoder parity test and fused forward
  - Result: Passed; source and time branches receive the same tensor.
  - Commit: Pending phase commit

- [x] T406 Preserve `fusion.mode=off` behavior.

  Evidence:
  - Source files: PVB `module/model.py`
  - Target files: `module/model.py`, `scripts/overfit_one_batch.py`
  - Commands: explicit off branch retains PVB encoder/bridge path
  - Tests: PVB off one-batch overfit and baseline profile
  - Result: Passed; off mode does not instantiate or run Anew fusion.
  - Commit: Pending phase commit

- [x] T407 Add gate-zero decoder parity test.

  Evidence:
  - Source files: PVB decoder implementation
  - Target files: `tests/test_fusion.py`
  - Commands: `python -m unittest -v tests.test_fusion`
  - Tests: zero-gated condition versus no condition
  - Result: Passed; decoder outputs match within `1e-6` tolerance.
  - Commit: Pending phase commit

- [x] T408 Verify nonzero projector and gate gradients.

  Evidence:
  - Source files: fused model projection/gate
  - Target files: `tests/test_fusion.py`
  - Commands: fused backward with gate set to `0.2` for gradient check
  - Tests: finite `block_gate` and projector gradients
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T409 Run one-batch overfit for `off` and `anew_block`.

  Evidence:
  - Source files: PVB bridge/decoder and Anew encoder
  - Target files: `scripts/overfit_one_batch.py`
  - Commands: `python -m scripts.overfit_one_batch --atoms 16 --samples 2 --steps 5` with both modes
  - Tests: deterministic finite-loss regression
  - Result: Passed; off `29.2191 → 3.1783`, fused `11.6374 → 0.8612`.
  - Commit: Pending phase commit

Gate P4: PASSED; parity, gradients, and both one-batch overfit checks succeed.

## Phase 5 — Checkpoints

- [x] T500 Replace whole-object checkpoint loading with state-dict loading.

  Evidence:
  - Source files: PVB `train.py`, `trainer/abs_trainer.py`; Anew `train.py`
  - Target files: `train.py`, `utils/checkpoint.py`, `trainer/abs_trainer.py`
  - Commands: source checkpoint audit; `python train.py --help`
  - Tests: checkpoint fixture suite
  - Result: Passed; target constructs `dyVAE` first and extracts state dictionaries explicitly.
  - Commit: Pending phase commit

- [x] T501 Implement PVB key migration.

  Evidence:
  - Source files: PVB serialized `dyVAE.state_dict()` layout
  - Target files: `utils/checkpoint.py`
  - Commands: scoped `decoder.*`, `vel_ffn.*`, and `drf_ffn.*` selection
  - Tests: serialized PVB module fixture
  - Result: Passed; PVB encoder keys are reported as unused rather than loaded into Anew.
  - Commit: Pending phase commit

- [x] T502 Implement Anew key migration.

  Evidence:
  - Source files: Anew `CondIterAutoEncoderEdge` attribute layout
  - Target files: `utils/checkpoint.py`, `module/anew_block_encoder.py`
  - Commands: explicit mappings for embedding, EPT, edge, projection, and `Wx_log_var`
  - Tests: synthetic standalone-Anew state dictionary
  - Result: Passed; 47/47 compatible Anew keys matched.
  - Commit: Pending phase commit

- [x] T503 Implement full fused resume.

  Evidence:
  - Source files: PVB trainer optimizer/scheduler lifecycle
  - Target files: `utils/checkpoint.py`, `trainer/abs_trainer.py`, `train.py`
  - Commands: save/load model, optimizer, scheduler, EMA, and progress fields
  - Tests: resume fixture restores model and optimizer state
  - Result: Passed; model and optimizer coverage was 136/136.
  - Commit: Pending phase commit

- [x] T504 Print key coverage and mismatch reports.

  Evidence:
  - Source files: checkpoint role contracts
  - Target files: `utils/checkpoint.py`
  - Commands: `CheckpointReport.summary()` during every role load
  - Tests: checkpoint test output
  - Result: Passed; matched, missing, unexpected, shape-mismatch, and percentage fields are printed.
  - Commit: Pending phase commit

- [x] T505 Add configurable minimum coverage thresholds.

  Evidence:
  - Source files: requested checkpoint acceptance contract
  - Target files: `config/train.yaml`, `train.py`, `utils/checkpoint.py`
  - Commands: `model.checkpoint_min_coverage` passed to each loader
  - Tests: low-coverage fixture raises `CheckpointCoverageError`
  - Result: Passed; insufficient coverage cannot load silently.
  - Commit: Pending phase commit

- [x] T506 Add checkpoint-loading tests with synthetic and real checkpoints.

  Evidence:
  - Source files: PVB module serialization and Anew state layout
  - Target files: `tests/test_checkpoints.py`
  - Commands: `python -m unittest -v tests.test_checkpoints`
  - Tests: PVB module, Anew translated state dict, resume, and low-coverage fixtures
  - Result: Passed; 4 tests.
  - Commit: Pending phase commit

## Phase 6 — Training and performance

- [x] T600 Add frozen-Anew training stage.

  Evidence:
  - Source files: staged training specification
  - Target files: `utils/fusion_training.py`, `trainer/abs_trainer.py`, `config/train.yaml`
  - Commands: Stage A configuration
  - Tests: `python -m unittest -v tests.test_training_stages`
  - Result: Passed; Anew parameters freeze while PVB decoder/heads, projector, and gate remain trainable.
  - Commit: Pending phase commit

- [x] T601 Add separate optimizer parameter groups.

  Evidence:
  - Source files: PVB `trainer/abs_trainer.py`
  - Target files: `utils/fusion_training.py`, `trainer/abs_trainer.py`, `train.py`
  - Commands: named `pvb`, `anew`, and `projector_gate` groups with separate rates
  - Tests: Stage A group test
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T602 Add selective EPT unfreezing.

  Evidence:
  - Source files: vendored Anew EPT layer layout
  - Target files: `utils/fusion_training.py`, `config/train.yaml`
  - Commands: Stage B last-N layer selection
  - Tests: three-layer fixture unfreezes only layers 1 and 2
  - Result: Passed.
  - Commit: Pending phase commit

- [x] T603 Log per-module gradient norms and gate value.

  Evidence:
  - Source files: PVB trainer logging lifecycle
  - Target files: `utils/fusion_training.py`, `trainer/abs_trainer.py`
  - Commands: post-backward `Fusion/*` logging
  - Tests: finite diagnostics test
  - Result: Passed; PVB/projector gradient norms and scalar gate are reported.
  - Commit: Pending phase commit

- [x] T604 Profile 256/512/1024/2000-atom batches.

  Evidence:
  - Source files: PVB/Anew model paths
  - Target files: `scripts/profile_components.py`, `HANDOFF.md`
  - Commands: `python -m scripts.profile_components --device cuda --atoms {256,512,1024,2000} --steps 2` for both modes
  - Tests: stable post-warmup forward/backward and peak-memory measurements
  - Result: Passed; all eight runs finite and 2000 atoms completed without OOM.
  - Commit: Pending phase commit

- [x] T605 Separate graph, encoder, decoder, and backward timings.

  Evidence:
  - Source files: PVB `module/graph.py`, `module/model.py`
  - Target files: `scripts/profile_components.py`
  - Commands: timed wrappers around graph, encoder, decoder, and backward
  - Tests: baseline/fused component profiles
  - Result: Passed; component timings are reported separately.
  - Commit: Pending phase commit

- [x] T606 Add BF16 autocast behind a config flag.

  Evidence:
  - Source files: PVB trainer forward/validation lifecycle
  - Target files: `trainer/abs_trainer.py`, `train.py`, `config/train.yaml`
  - Commands: `training.bf16_autocast` flag and guarded `torch.autocast`
  - Tests: CUDA fused BF16 forward/backward probe (`BF16_OK True True`)
  - Result: Passed; finite loss and gradients.
  - Commit: Pending phase commit

- [x] T607 Add attention-aware batch budgeting.

  Evidence:
  - Source files: PVB `data/dataset_wrapper.py`
  - Target files: `config/train.yaml`, `tests/test_training_stages.py`
  - Commands: set `data.complexity: "n*n"`
  - Tests: dynamic wrapper budget test
  - Result: Passed; each synthetic batch stays within the pairwise budget.
  - Commit: Pending phase commit

- [x] T608 Compare PVB, legacy fusion if retained, and block fusion.

  Evidence:
  - Source files: PVB baseline and Anew H-block paths
  - Target files: `scripts/profile_components.py`, `HANDOFF.md`
  - Commands: matched 256/512/1024/2000 profiles for `off` and `anew_block`
  - Tests: finite timing/memory comparison
  - Result: Passed; no legacy fusion path is retained, and the direct baseline/H-block comparison is recorded.
  - Commit: Pending phase commit

- [x] T609 Decide whether a separate block-sparse encoder is necessary.

  Evidence:
  - Source files: Anew EPT source default `sparse_k=3`; benchmark results
  - Target files: `DECISIONS.md`, `HANDOFF.md`
  - Commands: source sparse-default audit; 2000-atom profile
  - Tests: faithful path completed within 80-GB A100 memory
  - Result: Deferred; no separately named `block_sparse` approximation is needed at this milestone. xFormers remains unavailable.
  - Commit: Pending phase commit

## Phase 7 — Final verification

- [x] T700 Run full unit tests.

  Evidence:
  - Source files: target `tests/`
  - Target files: all fused-repository tests
  - Commands: `python -m unittest discover -s tests -v`
  - Tests: 22 tests
  - Result: Passed in 15.297 s. Root-level discovery additionally finds unrelated optional PVB packages requiring legacy imports/OpenMM; the target suite is green.
  - Commit: Pending final commit

- [x] T701 Run protein training smoke test.

  Evidence:
  - Source files: PVB model/trainer path; Anew block encoder path
  - Target files: `scripts/protein_smoke.py`
  - Commands: `python -m scripts.protein_smoke --device cuda --fusion-mode off`; same command with `--fusion-mode anew_block`
  - Tests: one synthetic protein training step per mode
  - Result: Passed; finite losses `13.7539` and `4.1669`, with finite gradients.
  - Commit: Pending final commit

- [x] T702 Run protein inference smoke test.

  Evidence:
  - Source files: PVB protein inference path and fused decoder conditioning
  - Target files: `scripts/protein_smoke.py`
  - Commands: the two T701 smoke commands, each including a short inference rollout
  - Tests: finite generated coordinates for `off` and `anew_block`
  - Result: Passed; both modes returned shape `[16, 3]` with finite values.
  - Commit: Pending final commit

- [x] T703 Confirm source repositories are unchanged.

  Evidence:
  - Source files: `/workspace/PVB`, `/workspace/AnewOmni`
  - Target files: none
  - Commands: `git -C /workspace/PVB status --porcelain=v1`; `git -C /workspace/AnewOmni status --porcelain=v1`
  - Tests: source worktree audit
  - Result: Passed; both source worktrees are clean at the recorded SHAs.
  - Commit: Pending final commit

- [x] T704 Confirm no runtime dependency on sibling repositories.

  Evidence:
  - Source files: target Python tree
  - Target files: `third_party/anewomni/`, vendored integration modules, parity test
  - Commands: `rg -n 'sys\\.path|/workspace/(PVB|AnewOmni)|import AnewOmni|from AnewOmni' . --glob '*.py' --glob '!tests/test_anew_vendor_parity.py'`
  - Tests: sibling path/import audit
  - Result: Passed; no PVB/Anew sibling path or import exists in runtime fusion code. Existing optional PVB local `sys.path.append('..')` compatibility lines are documented in `HANDOFF.md`; parity subprocesses are test-only.
  - Commit: Pending final commit

- [x] T705 Update all four planning/handoff files.

  Evidence:
  - Source files: user-provided planning contracts
  - Target files: `PLAN.md`, `TASKS.md`, `HANDOFF.md`, `DECISIONS.md`
  - Commands: final document review and status/evidence updates
  - Tests: live-state consistency review
  - Result: Passed; phase status, validation, limitations, decisions, and next action are recorded.
  - Commit: Pending final commit

- [ ] T706 Record final changed-file list and benchmark table. **IN_PROGRESS**

  Evidence:
  - Source files: target Git history and `scripts/profile_components.py`
  - Target files: `HANDOFF.md`, final target commit
  - Commands: `git status --short`; `git diff --check`; final commit
  - Tests: changed-file and benchmark review
  - Result: Pending final commit and SHA update.
  - Commit: Pending final commit
