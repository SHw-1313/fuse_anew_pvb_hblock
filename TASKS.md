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
  - Commit: 62ceeb4

- [x] T101 Copy Anew license.

  Evidence:
  - Source files: `AnewOmni/LICENSE`
  - Target files: `third_party/anewomni/LICENSE`
  - Commands: complete-file copy
  - Tests: source/target license inspection
  - Result: Passed; license retained.
  - Commit: 62ceeb4

- [x] T102 Copy `models/modules/EPT/ept.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/EPT/ept.py`
  - Target files: `third_party/anewomni/models/modules/EPT/ept.py`
  - Commands: complete-file copy with import-only namespace patch
  - Tests: EPT numerical parity test
  - Result: Passed; output matches the source implementation.
  - Commit: 62ceeb4

- [x] T103 Copy `models/modules/EPT/radial_basis.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/EPT/radial_basis.py`
  - Target files: `third_party/anewomni/models/modules/EPT/radial_basis.py`
  - Commands: complete-file copy
  - Tests: vendored EPT parity subprocess
  - Result: Passed; source implementation reused.
  - Commit: 62ceeb4

- [x] T104 Copy `models/modules/GET/tools.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/GET/tools.py`
  - Target files: `third_party/anewomni/models/modules/GET/tools.py`
  - Commands: complete-file copy
  - Tests: vendored encoder forward
  - Result: Passed; Anew block/unit edge utility is available locally.
  - Commit: 62ceeb4

- [x] T105 Copy `models/modules/nn.py`.

  Evidence:
  - Source files: `AnewOmni/models/modules/nn.py`
  - Target files: `third_party/anewomni/models/modules/nn.py`
  - Commands: complete-file copy
  - Tests: BlockEmbedding numerical parity test
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T106 Copy required `utils/nn_utils.py`.

  Evidence:
  - Source files: `AnewOmni/utils/nn_utils.py`
  - Target files: `third_party/anewomni/utils/nn_utils.py`
  - Commands: complete-file copy
  - Tests: vendored EPT import and encoder tests
  - Result: Passed; stable normalization/batching dependency is present.
  - Commit: 62ceeb4

- [x] T107 Copy required `utils/gnn_utils.py`.

  Evidence:
  - Source files: `AnewOmni/utils/gnn_utils.py`
  - Target files: `third_party/anewomni/utils/gnn_utils.py`
  - Commands: complete-file copy
  - Tests: Anew pooling formula test
  - Result: Passed; variance-preserving pooling matches reference semantics.
  - Commit: 62ceeb4

- [x] T108 Copy `utils/register.py` or remove registration with a documented minimal patch.

  Evidence:
  - Source files: `AnewOmni/utils/register.py`
  - Target files: `third_party/anewomni/utils/register.py`
  - Commands: complete-file copy
  - Tests: namespaced imports
  - Result: Passed; registration dependency is isolated under the vendored namespace.
  - Commit: 62ceeb4

- [x] T109 Convert copied imports to package-relative imports.

  Evidence:
  - Source files: vendored EPT imports
  - Target files: `third_party/anewomni/models/modules/EPT/ept.py`
  - Commands: changed only sibling utility imports to relative imports
  - Tests: source/target subprocess parity without `PYTHONPATH` or `sys.path` injection
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T110 Record every copied source and SHA in `DECISIONS.md`.

  Evidence:
  - Source files: pinned Anew source tree
  - Target files: `DECISIONS.md`
  - Commands: source path/SHA audit
  - Tests: source worktree status
  - Result: Passed; D017 records the complete vendor map.
  - Commit: 62ceeb4

- [x] T111 Add import tests for the vendored package.

  Evidence:
  - Source files: vendored package modules
  - Target files: `tests/test_anew_vendor_parity.py`
  - Commands: `python -m unittest -v tests.test_anew_vendor_parity`
  - Tests: import and numerical parity tests
  - Result: Passed; 2 tests.
  - Commit: 62ceeb4

- [x] T112 Add parity tests against the corresponding Anew source implementation.

  Evidence:
  - Source files: Anew `BlockEmbedding` and `XTransEncoderAct`
  - Target files: `tests/test_anew_vendor_parity.py`
  - Commands: `python -m unittest -v tests.test_anew_vendor_parity`
  - Tests: source-vendored output comparisons
  - Result: Passed; numerical parity confirmed.
  - Commit: 62ceeb4

Gate P1: PASSED; vendored EPT, embedding, edge, and pooling dependencies match the source implementation.

## Phase 2 — Block data contract

- [x] T200 Inspect every PVB dataset used by the requested training stage.

  Evidence:
  - Source files: `data/*_dataset.py`, `data/subgraph.py`, `data/collate.py`
  - Target files: same PVB paths
  - Commands: graph-crop and block-field call-site audit
  - Tests: all relevant graph crop paths reviewed
  - Result: Passed; protein and complex preprocessors were identified and updated where applicable.
  - Commit: 62ceeb4

- [x] T201 Define explicit block metadata schema.

  Evidence:
  - Source files: PVB item/collate contracts
  - Target files: `data/block_metadata.py`
  - Commands: schema implementation
  - Tests: metadata contract tests
  - Result: Passed; schema is `atom_block_id`, `block_type`, `block_batch`, `block_lengths`.
  - Commit: 62ceeb4

- [x] T202 Add preprocessing support for local block IDs.

  Evidence:
  - Source files: `data/mmap_dataset.py`, `data/data_init.py`
  - Target files: same paths; `data/block_metadata.py`
  - Commands: annotate records during preprocessing and batch creation
  - Tests: explicit preprocessing annotation test
  - Result: Passed; local IDs are materialized before model forward.
  - Commit: 62ceeb4

- [x] T203 Make graph cropping preserve complete residue blocks.

  Evidence:
  - Source files: PVB graph crop call sites
  - Target files: `data/subgraph.py`, protein/complex dataset preprocessors
  - Commands: pass explicit `block_id` to `graph_cut`
  - Tests: partial outer block expansion test
  - Result: Passed; selected blocks are complete.
  - Commit: 62ceeb4

- [x] T204 Offset block IDs correctly in `data/collate.py`.

  Evidence:
  - Source files: `data/collate.py`
  - Target files: `data/collate.py`, `data/block_metadata.py`
  - Commands: global offset implementation
  - Tests: two-sample offset test
  - Result: Passed; IDs remain contiguous and sample-isolated.
  - Commit: 62ceeb4

- [x] T205 Produce `atom_block_id`, `block_type`, `block_batch`, `block_lengths`.

  Evidence:
  - Source files: PVB collate/data-init paths
  - Target files: `data/collate.py`, `data/data_init.py`
  - Commands: explicit metadata update in returned batches
  - Tests: metadata contract test
  - Result: Passed; all four fields are emitted for explicit and legacy records.
  - Commit: 62ceeb4

- [x] T206 Add compatibility conversion for old datasets, on CPU only.

  Evidence:
  - Source files: legacy `b0`/`btype` records
  - Target files: `data/block_metadata.py`
  - Commands: CPU fallback derivation with warning
  - Tests: legacy warning/fallback test
  - Result: Passed; fallback is outside model GPU forward.
  - Commit: 62ceeb4

- [x] T207 Add block ordering, length, batch-isolation, and repeated-residue tests.

  Evidence:
  - Source files: metadata/collate implementation
  - Target files: `tests/test_block_metadata.py`
  - Commands: `python -m unittest -v tests.test_block_metadata`
  - Tests: ordering, lengths, offsets, repeated residue IDs, and crop behavior
  - Result: Passed; 4 tests.
  - Commit: 62ceeb4

- [x] T208 Reject unsupported ligand/molecular block mappings clearly.

  Evidence:
  - Source files: verified PVB/Anew vocabularies
  - Target files: `module/anew_block_encoder.py`
  - Commands: explicit protein vocabulary assertion
  - Tests: non-protein block rejection test
  - Result: Passed; unsupported element/ligand blocks raise a clear protein-only error.
  - Commit: 62ceeb4

Gate P2: PASSED; model code consumes explicit integer metadata and does not infer blocks from `b0` equality.

## Phase 3 — Anew block encoder

- [x] T300 Add `module/anew_block_encoder.py`.

  Evidence:
  - Source files: Anew `models/IterVAE/model_edge.py`
  - Target files: `module/anew_block_encoder.py`
  - Commands: implementation and compileall
  - Tests: `tests.test_anew_block_encoder`
  - Result: Passed; explicit-metadata encoder is available.
  - Commit: 62ceeb4

- [x] T301 Reuse Anew `BlockEmbedding`.

  Evidence:
  - Source files: Anew `models/modules/nn.py`
  - Target files: `module/anew_block_encoder.py`, `third_party/anewomni/models/modules/nn.py`
  - Commands: direct vendored import
  - Tests: vendor parity and encoder shape tests
  - Result: Passed; no replacement embedding was reimplemented.
  - Commit: 62ceeb4

- [x] T302 Reuse Anew EPT.

  Evidence:
  - Source files: Anew `models/modules/EPT/ept.py`, `radial_basis.py`
  - Target files: `module/anew_block_encoder.py`, `third_party/anewomni/models/modules/EPT/`
  - Commands: direct vendored import
  - Tests: EPT parity and finite-gradient tests
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T303 Reuse Anew edge utilities.

  Evidence:
  - Source files: Anew `models/modules/GET/tools.py`
  - Target files: `module/anew_block_encoder.py`, vendored GET package
  - Commands: `knn_edges` and vendored edge embedding path
  - Tests: encoder batch-isolation test
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T304 Adapt the encoding sequence from Anew `model_edge.py`.

  Evidence:
  - Source files: Anew `models/IterVAE/model_edge.py`
  - Target files: `module/anew_block_encoder.py`
  - Commands: BlockEmbedding → EPT → atom/block pooling adaptation
  - Tests: encoder forward and SE(3) tests
  - Result: Passed; source sequence preserved with explicit PVB metadata.
  - Commit: 62ceeb4

- [x] T305 Implement Anew-equivalent `H_atom → H_block` pooling.

  Evidence:
  - Source files: Anew `utils/gnn_utils.py`
  - Target files: `module/anew_block_encoder.py`
  - Commands: `std_conserve_scatter_mean` reuse
  - Tests: exact scatter-sum/sqrt-length comparison
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T306 Implement `X_atom → X_block` pooling.

  Evidence:
  - Source files: Anew `model_edge.py` coordinate pooling
  - Target files: `module/anew_block_encoder.py`
  - Commands: `scatter_mean` pooling
  - Tests: exact block coordinate mean comparison
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T307 Return a typed/documented output dictionary.

  Evidence:
  - Source files: requested architecture contract
  - Target files: `module/anew_block_encoder.py`
  - Commands: documented `Dict[str, Tensor]` forward
  - Tests: key and shape assertions
  - Result: Passed; all eight required output keys are returned.
  - Commit: 62ceeb4

- [x] T308 Keep PVB coordinate mean equal to `x0`.

  Evidence:
  - Source files: PVB `module/model.py` bridge contract
  - Target files: `module/model.py`, `tests/test_fusion.py`
  - Commands: fused `_encode_anew_block` keeps `x_mu = x.clone()`
  - Tests: zero-latent-noise source-mean test
  - Result: Passed; Anew coordinate output is not substituted.
  - Commit: 62ceeb4

- [x] T309 Ensure `X_atom`/`X_block` are diagnostic-only in milestone one.

  Evidence:
  - Source files: Anew encoder output contract
  - Target files: `module/model.py`, `module/anew_block_encoder.py`
  - Commands: no X output used in PVB bridge or decoder condition
  - Tests: fused source-mean and output-key tests
  - Result: Passed; X outputs are returned only for diagnostics.
  - Commit: 62ceeb4

- [x] T310 Remove unused PVB encoder-edge construction in `anew_block` mode.

  Evidence:
  - Source files: PVB `module/model.py`, `module/graph.py`
  - Target files: `module/model.py`
  - Commands: branch encoder graph construction on `fusion_mode`
  - Tests: fused one-batch forward/backward
  - Result: Passed; fused paths do not build PVB encoder edges.
  - Commit: 62ceeb4

- [x] T311 Add shape, pooling, batch-isolation, gradient, and SE(3) tests.

  Evidence:
  - Source files: encoder and metadata contracts
  - Target files: `tests/test_anew_block_encoder.py`, `tests/test_fusion.py`
  - Commands: `python -m unittest -v tests.test_anew_block_encoder tests.test_fusion`
  - Tests: 8 encoder/fusion tests
  - Result: Passed; shapes, pooling, isolation, gradients, SE(3), and source mean verified.
  - Commit: 62ceeb4

Gate P3: PASSED; block outputs and coordinate contract were verified before decoder conditioning.

## Phase 4 — PVB decoder conditioning

- [x] T400 Add fusion configuration with `off` and `anew_block`.

  Evidence:
  - Source files: PVB `train.py`, `config/train.yaml`
  - Target files: `train.py`, `config/train.yaml`, `module/model.py`
  - Commands: config/model construction audit
  - Tests: off and fused model instantiation
  - Result: Passed; `fusion.mode` supports both paths.
  - Commit: 62ceeb4

- [x] T401 Add block projection `LayerNorm(512) → Linear(512, hidden_dim)`.

  Evidence:
  - Source files: requested architecture contract
  - Target files: `module/model.py`
  - Commands: `nn.Sequential(LayerNorm, Linear)` construction
  - Tests: projection gradient test
  - Result: Passed; default Anew hidden size is 512 and output is PVB hidden size.
  - Commit: 62ceeb4

- [x] T402 Add scalar zero-initialized `block_gate`.

  Evidence:
  - Source files: requested zero-gating contract
  - Target files: `module/model.py`
  - Commands: `nn.Parameter(torch.zeros(1))`
  - Tests: gate-zero parity test
  - Result: Passed; initial gate is exactly zero.
  - Commit: 62ceeb4

- [x] T403 Broadcast block features with `H_block[atom_block_id]`.

  Evidence:
  - Source files: explicit block metadata contract
  - Target files: `module/model.py`
  - Commands: project block states then `index_select` by global atom block ID
  - Tests: metadata and fused gradient tests
  - Result: Passed; no coordinate-based membership lookup is used.
  - Commit: 62ceeb4

- [x] T404 Inject conditioning into the PVB decoder input.

  Evidence:
  - Source files: PVB `module/torchmd_et.py`
  - Target files: `module/model.py`, `module/torchmd_et.py`
  - Commands: add gated condition to decoder branch features
  - Tests: gate-zero and fused overfit tests
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T405 Apply identical conditioning to both cross-attention branches.

  Evidence:
  - Source files: PVB TorchMD cross-attention `x0`/`xt` branches
  - Target files: `module/torchmd_et.py`
  - Commands: condition added once to each branch before neighbor embedding
  - Tests: decoder parity test and fused forward
  - Result: Passed; source and time branches receive the same tensor.
  - Commit: 62ceeb4

- [x] T406 Preserve `fusion.mode=off` behavior.

  Evidence:
  - Source files: PVB `module/model.py`
  - Target files: `module/model.py`, `scripts/overfit_one_batch.py`
  - Commands: explicit off branch retains PVB encoder/bridge path
  - Tests: PVB off one-batch overfit and baseline profile
  - Result: Passed; off mode does not instantiate or run Anew fusion.
  - Commit: 62ceeb4

- [x] T407 Add gate-zero decoder parity test.

  Evidence:
  - Source files: PVB decoder implementation
  - Target files: `tests/test_fusion.py`
  - Commands: `python -m unittest -v tests.test_fusion`
  - Tests: zero-gated condition versus no condition
  - Result: Passed; decoder outputs match within `1e-6` tolerance.
  - Commit: 62ceeb4

- [x] T408 Verify nonzero projector and gate gradients.

  Evidence:
  - Source files: fused model projection/gate
  - Target files: `tests/test_fusion.py`
  - Commands: fused backward with gate set to `0.2` for gradient check
  - Tests: finite `block_gate` and projector gradients
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T409 Run one-batch overfit for `off` and `anew_block`.

  Evidence:
  - Source files: PVB bridge/decoder and Anew encoder
  - Target files: `scripts/overfit_one_batch.py`
  - Commands: `python -m scripts.overfit_one_batch --atoms 16 --samples 2 --steps 5` with both modes
  - Tests: deterministic finite-loss regression
  - Result: Passed; off `29.2191 → 3.1783`, fused `11.6374 → 0.8612`.
  - Commit: 62ceeb4

Gate P4: PASSED; parity, gradients, and both one-batch overfit checks succeed.

## Phase 5 — Checkpoints

- [x] T500 Replace whole-object checkpoint loading with state-dict loading.

  Evidence:
  - Source files: PVB `train.py`, `trainer/abs_trainer.py`; Anew `train.py`
  - Target files: `train.py`, `utils/checkpoint.py`, `trainer/abs_trainer.py`
  - Commands: source checkpoint audit; `python train.py --help`
  - Tests: checkpoint fixture suite
  - Result: Passed; target constructs `dyVAE` first and extracts state dictionaries explicitly.
  - Commit: 62ceeb4

- [x] T501 Implement PVB key migration.

  Evidence:
  - Source files: PVB serialized `dyVAE.state_dict()` layout
  - Target files: `utils/checkpoint.py`
  - Commands: scoped `decoder.*`, `vel_ffn.*`, and `drf_ffn.*` selection
  - Tests: serialized PVB module fixture
  - Result: Passed; PVB encoder keys are reported as unused rather than loaded into Anew.
  - Commit: 62ceeb4

- [x] T502 Implement Anew key migration.

  Evidence:
  - Source files: Anew `CondIterAutoEncoderEdge` attribute layout
  - Target files: `utils/checkpoint.py`, `module/anew_block_encoder.py`
  - Commands: explicit mappings for embedding, EPT, edge, projection, and `Wx_log_var`
  - Tests: synthetic standalone-Anew state dictionary
  - Result: Passed; 47/47 compatible Anew keys matched.
  - Commit: 62ceeb4

- [x] T503 Implement full fused resume.

  Evidence:
  - Source files: PVB trainer optimizer/scheduler lifecycle
  - Target files: `utils/checkpoint.py`, `trainer/abs_trainer.py`, `train.py`
  - Commands: save/load model, optimizer, scheduler, EMA, and progress fields
  - Tests: resume fixture restores model and optimizer state
  - Result: Passed; model and optimizer coverage was 136/136.
  - Commit: 62ceeb4

- [x] T504 Print key coverage and mismatch reports.

  Evidence:
  - Source files: checkpoint role contracts
  - Target files: `utils/checkpoint.py`
  - Commands: `CheckpointReport.summary()` during every role load
  - Tests: checkpoint test output
  - Result: Passed; matched, missing, unexpected, shape-mismatch, and percentage fields are printed.
  - Commit: 62ceeb4

- [x] T505 Add configurable minimum coverage thresholds.

  Evidence:
  - Source files: requested checkpoint acceptance contract
  - Target files: `config/train.yaml`, `train.py`, `utils/checkpoint.py`
  - Commands: `model.checkpoint_min_coverage` passed to each loader
  - Tests: low-coverage fixture raises `CheckpointCoverageError`
  - Result: Passed; insufficient coverage cannot load silently.
  - Commit: 62ceeb4

- [x] T506 Add checkpoint-loading tests with synthetic and real checkpoints.

  Evidence:
  - Source files: PVB module serialization and Anew state layout
  - Target files: `tests/test_checkpoints.py`
  - Commands: `python -m unittest -v tests.test_checkpoints`
  - Tests: PVB module, Anew translated state dict, resume, and low-coverage fixtures
  - Result: Passed; 4 tests.
  - Commit: 62ceeb4

## Phase 6 — Training and performance

- [x] T600 Add frozen-Anew training stage.

  Evidence:
  - Source files: staged training specification
  - Target files: `utils/fusion_training.py`, `trainer/abs_trainer.py`, `config/train.yaml`
  - Commands: Stage A configuration
  - Tests: `python -m unittest -v tests.test_training_stages`
  - Result: Passed; Anew parameters freeze while PVB decoder/heads, projector, and gate remain trainable.
  - Commit: 62ceeb4

- [x] T601 Add separate optimizer parameter groups.

  Evidence:
  - Source files: PVB `trainer/abs_trainer.py`
  - Target files: `utils/fusion_training.py`, `trainer/abs_trainer.py`, `train.py`
  - Commands: named `pvb`, `anew`, and `projector_gate` groups with separate rates
  - Tests: Stage A group test
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T602 Add selective EPT unfreezing.

  Evidence:
  - Source files: vendored Anew EPT layer layout
  - Target files: `utils/fusion_training.py`, `config/train.yaml`
  - Commands: Stage B last-N layer selection
  - Tests: three-layer fixture unfreezes only layers 1 and 2
  - Result: Passed.
  - Commit: 62ceeb4

- [x] T603 Log per-module gradient norms and gate value.

  Evidence:
  - Source files: PVB trainer logging lifecycle
  - Target files: `utils/fusion_training.py`, `trainer/abs_trainer.py`
  - Commands: post-backward `Fusion/*` logging
  - Tests: finite diagnostics test
  - Result: Passed; PVB/projector gradient norms and scalar gate are reported.
  - Commit: 62ceeb4

- [x] T604 Profile 256/512/1024/2000-atom batches.

  Evidence:
  - Source files: PVB/Anew model paths
  - Target files: `scripts/profile_components.py`, `HANDOFF.md`
  - Commands: `python -m scripts.profile_components --device cuda --atoms {256,512,1024,2000} --steps 2` for both modes
  - Tests: stable post-warmup forward/backward and peak-memory measurements
  - Result: Passed; all eight runs finite and 2000 atoms completed without OOM.
  - Commit: 62ceeb4

- [x] T605 Separate graph, encoder, decoder, and backward timings.

  Evidence:
  - Source files: PVB `module/graph.py`, `module/model.py`
  - Target files: `scripts/profile_components.py`
  - Commands: timed wrappers around graph, encoder, decoder, and backward
  - Tests: baseline/fused component profiles
  - Result: Passed; component timings are reported separately.
  - Commit: 62ceeb4

- [x] T606 Add BF16 autocast behind a config flag.

  Evidence:
  - Source files: PVB trainer forward/validation lifecycle
  - Target files: `trainer/abs_trainer.py`, `train.py`, `config/train.yaml`
  - Commands: `training.bf16_autocast` flag and guarded `torch.autocast`
  - Tests: CUDA fused BF16 forward/backward probe (`BF16_OK True True`)
  - Result: Passed; finite loss and gradients.
  - Commit: 62ceeb4

- [x] T607 Add attention-aware batch budgeting.

  Evidence:
  - Source files: PVB `data/dataset_wrapper.py`
  - Target files: `config/train.yaml`, `tests/test_training_stages.py`
  - Commands: set `data.complexity: "n*n"`
  - Tests: dynamic wrapper budget test
  - Result: Passed; each synthetic batch stays within the pairwise budget.
  - Commit: 62ceeb4

- [x] T608 Compare PVB, legacy fusion if retained, and block fusion.

  Evidence:
  - Source files: PVB baseline and Anew H-block paths
  - Target files: `scripts/profile_components.py`, `HANDOFF.md`
  - Commands: matched 256/512/1024/2000 profiles for `off` and `anew_block`
  - Tests: finite timing/memory comparison
  - Result: Passed; no legacy fusion path is retained, and the direct baseline/H-block comparison is recorded.
  - Commit: 62ceeb4

- [x] T609 Decide whether a separate block-sparse encoder is necessary.

  Evidence:
  - Source files: Anew EPT source default `sparse_k=3`; benchmark results
  - Target files: `DECISIONS.md`, `HANDOFF.md`
  - Commands: source sparse-default audit; 2000-atom profile
  - Tests: faithful path completed within 80-GB A100 memory
  - Result: Deferred; no separately named `block_sparse` approximation is needed at this milestone. xFormers remains unavailable.
  - Commit: 62ceeb4

## Phase 7 — Final verification

- [x] T700 Run full unit tests.

  Evidence:
  - Source files: target `tests/`
  - Target files: all fused-repository tests
  - Commands: `python -m unittest discover -s tests -v`
  - Tests: 22 tests
  - Result: Passed in 15.297 s. Root-level discovery additionally finds unrelated optional PVB packages requiring legacy imports/OpenMM; the target suite is green.
  - Commit: 62ceeb4

- [x] T701 Run protein training smoke test.

  Evidence:
  - Source files: PVB model/trainer path; Anew block encoder path
  - Target files: `scripts/protein_smoke.py`
  - Commands: `python -m scripts.protein_smoke --device cuda --fusion-mode off`; same command with `--fusion-mode anew_block`
  - Tests: one synthetic protein training step per mode
  - Result: Passed; finite losses `13.7539` and `4.1669`, with finite gradients.
  - Commit: 62ceeb4

- [x] T702 Run protein inference smoke test.

  Evidence:
  - Source files: PVB protein inference path and fused decoder conditioning
  - Target files: `scripts/protein_smoke.py`
  - Commands: the two T701 smoke commands, each including a short inference rollout
  - Tests: finite generated coordinates for `off` and `anew_block`
  - Result: Passed; both modes returned shape `[16, 3]` with finite values.
  - Commit: 62ceeb4

- [x] T703 Confirm source repositories are unchanged.

  Evidence:
  - Source files: `/workspace/PVB`, `/workspace/AnewOmni`
  - Target files: none
  - Commands: `git -C /workspace/PVB status --porcelain=v1`; `git -C /workspace/AnewOmni status --porcelain=v1`
  - Tests: source worktree audit
  - Result: Passed; both source worktrees are clean at the recorded SHAs.
  - Commit: 62ceeb4

- [x] T704 Confirm no runtime dependency on sibling repositories.

  Evidence:
  - Source files: target Python tree
  - Target files: `third_party/anewomni/`, vendored integration modules, parity test
  - Commands: `rg -n 'sys\\.path|/workspace/(PVB|AnewOmni)|import AnewOmni|from AnewOmni' . --glob '*.py' --glob '!tests/test_anew_vendor_parity.py'`
  - Tests: sibling path/import audit
  - Result: Passed; no PVB/Anew sibling path or import exists in runtime fusion code. Existing optional PVB local `sys.path.append('..')` compatibility lines are documented in `HANDOFF.md`; parity subprocesses are test-only.
  - Commit: 62ceeb4

- [x] T705 Update all four planning/handoff files.

  Evidence:
  - Source files: user-provided planning contracts
  - Target files: `PLAN.md`, `TASKS.md`, `HANDOFF.md`, `DECISIONS.md`
  - Commands: final document review and status/evidence updates
  - Tests: live-state consistency review
  - Result: Passed; phase status, validation, limitations, decisions, and next action are recorded.
  - Commit: 62ceeb4

- [x] T706 Record final changed-file list and benchmark table.

  Evidence:
  - Source files: target Git history and `scripts/profile_components.py`
  - Target files: `HANDOFF.md`, final target commit
  - Commands: `git status --short`; `git diff --check`; final commit
  - Tests: changed-file and benchmark review
  - Result: Passed; final changed-file list and multi-scale baseline/H-block benchmark table are recorded in `HANDOFF.md`.
  - Commit: 62ceeb4 (implementation; this task also includes the final documentation follow-up)

## Phase 8 — Performance diagnosis

- [x] T800 Update profiling with CUDA-event warmup, repeated measurements, and p50/p90 statistics.

  Evidence:
  - Source files: existing `scripts/profile_components.py` timing wrappers and PVB/Anew model paths.
  - Target files: `scripts/profile_components.py`.
  - Commands: `python -m py_compile scripts/profile_components.py`; CPU off/fused smoke; CUDA off/fused smoke with `--warmup-steps 1 --steps 2`.
  - Tests: finite CPU and CUDA losses; warmup excluded from measurements; CUDA event timing and allocated/reserved memory fields present.
  - Result: Passed; profiler now reports forward, backward, optimizer, graph, encoder, decoder, p50/p90/std/CV, and raw measurements. Two CUDA smoke modes completed with finite values.
  - Commit: pending.

- [x] T801 Re-run the synthetic scaling curve at 512/1024/1536/1800/2000/2048 atoms.

  Evidence:
  - Source files: PVB `module/model.py`, PVB `module/torchmd_et.py`, and vendored Anew EPT/graph paths.
  - Target files: `scripts/profile_components.py`; raw profiles under `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/profiles/`.
  - Commands: 12 CUDA runs on `cuda:5`, each with `--warmup-steps 10 --steps 20`, covering `off` and `anew_block` at 512/1024/1536/1800/2000/2048 atoms.
  - Tests: all 12 JSON outputs have 20 finite measured steps; focused target suite `python -m unittest discover -s tests -v` passed 22 tests.
  - Result: Passed; Anew H-block p50 step time is `0.259745 s` at 1800, `0.279082 s` at 2000, and `0.289102 s` at 2048 (`+7.44%`, then `+3.59%`). There is no 2000-atom discontinuity or OOM. H-block is `1.44–1.65×` the PVB `off` step time across the measured range, so real-batch and operator-level profiling remain necessary.
  - Commit: pending.

- [x] T802 Profile real PDBBind batches and record atom, padding, block, and edge distributions.

  Evidence:
  - Source files: `data/mmap_dataset.py`, `data/dataset_wrapper.py`, `data/block_metadata.py`, and `/data/pvb_cross_dataset_20260810/blocks/pdbbind/`.
  - Target files: `scripts/profile_real_batches.py`; `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/real_batches/pdbbind_profile.json`.
  - Commands: `python -m scripts.profile_real_batches --dataset-root /data/pvb_cross_dataset_20260810/blocks --dataset pdbbind --max-records 100 --budgets 2000 4000000 8000000 --max-groups 8`.
  - Tests: `python -m py_compile scripts/profile_real_batches.py`; JSON parse and summary validation; 100 records inspected per split.
  - Result: Passed; PDBBind raw records use legacy CPU metadata fallback (100/100 per split), with atom p50 `2228.5/2193/1738` for train/valid/test and unsupported element-block atoms p50 `26/25/29`. The default `n*n` budget `2000` forms zero groups and skips every PDBBind record; budgets `4e6` and `8e6` produce usable groups. Representative hypothetical padding ratios range from `0.58` to `1.00`.
  - Commit: pending.

- [x] T803 Compare all-trainable, source-frozen adapter, and forward-only execution.

  Evidence:
  - Source files: legacy fused/PVB checkpoint state dictionaries; `utils/checkpoint.py`; PVB `module/model.py`.
  - Target files: `utils/checkpoint.py`, `utils/fusion_training.py`, `train.py`, `trainer/abs_trainer.py`, `data/protein_view.py`, `scripts/profile_training_paths.py`, and `tests/test_checkpoints.py` / `tests/test_training_stages.py`.
  - Commands: checkpoint coverage audit; `python -m unittest tests.test_checkpoints tests.test_training_stages -v`; `CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_training_paths ... --record-index 0`.
  - Tests: PVB decoder/head coverage `150/150`; Anew H-block coverage `68/68`; zero shape mismatches; protein-only view `2147` atoms/`282` blocks/`4368` bonds; all three execution modes finite.
  - Result: Passed. All-trainable `169.3 ms`, `14.75 GiB`, `10,949,066` trainable parameters; strict adapter `114.1 ms`, `9.53 GiB`, `33,281` trainable parameters; forward-only `68.3 ms`, `1.81 GiB`. Adapter gradients are finite and include the projector/gate path.
  - Commit: pending.
- [x] T804 Run PyTorch profiler on representative 1024- and 2000-atom batches.

  Evidence:
  - Source files: PVB graph/decoder, vendored Anew EPT/GET tools, and real PDBBind block records.
  - Target files: `scripts/profile_operator.py`; `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/operator/*.json` and `*.json.gz`.
  - Commands: `python -m py_compile scripts/profile_operator.py`; `CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_operator ... --record-index 24 --row-limit 500`; same command with `--record-index 0`.
  - Tests: both reports parsed successfully, contain CUDA operator events, nested Anew KNN/EPT scopes, peak-memory measurements, and Chrome traces.
  - Result: Passed; real protein-only representatives were `911` atoms/`109` blocks and `2147` atoms/`282` blocks, bracketing the requested 1024/2000 sizes.
  - Commit: pending.

- [x] T805 Attribute the scaling to graph construction, dense EPT attention, memory, and backward.

  Evidence:
  - Source files: `third_party/anewomni/models/modules/EPT/ept.py`, `third_party/anewomni/models/modules/GET/tools.py`, PVB `module/graph.py`, and `module/torchmd_et.py`.
  - Target files: `scripts/profile_operator.py`; phase-8 operator reports; `PLAN.md`, `DECISIONS.md`, and `HANDOFF.md`.
  - Commands: JSON operator-event extraction; source inspection of `Transformer.forward`, `SelfAttnLayer.forward`, `knn_edges`, and `_unit_edges_from_block_edges`.
  - Tests: dense attention, block-KNN, PVB graph, decoder, backward, and peak-memory events were present in both traces.
  - Result: Passed. EPT attention uses padded dense tensors `[1,4,912,912]` and `[1,4,2152,2152]`; representative softmax/batched-attention allocations grow from about `26.6/53.2 MiB` to `148.2/296.4 MiB`. Anew block-KNN construction was about `51.9/53.9 ms`, Anew EPT about `35.5/65.9 ms`, and peak allocation was `6.31/14.75 GiB` for `911/2147` atoms. The earlier 2000-atom discontinuity is not established; the dominant mechanism is dense atom-level EPT attention plus repeated block-candidate construction, with PVB graph/decoder and backward also scaling.
  - Commit: pending.

- [x] T806 Record the root cause and decide whether a separate `block_sparse` mode is needed.

  Evidence:
  - Source files: `data/dataset_wrapper.py`, vendored `graph_to_batch_nx`, T801/T802 profiles, and T804/T805 operator traces.
  - Target files: `scripts/profile_real_batches.py`; `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/real_batches/pdbind_profile.json`; planning files.
  - Commands: `python -m py_compile scripts/profile_real_batches.py`; real PDBBind profile with `n*n` budgets `2000/4e6/8e6`; sampled `get_len` versus materialized atom-count audit.
  - Tests: 100 records per split plus representative dynamic groups; exact padded attention work and stale-length ratios were computed.
  - Result: Passed. `get_len` underestimates materialized raw atom counts for all sampled records, with median estimator/actual ratios `0.726/0.742/0.738` for train/valid/test. Multi-record groups can also pay `count * ceil(max_N/8)^2`; representative attention-work ratios were as low as `0.516` of the padded work. The faithful path remains unchanged; no `block_sparse` mode is approved yet. Phase 8 gate passed.
  - Commit: pending.

Gate P8: timing evidence must be stable and operator-level before performance changes are introduced.

## Phase 9 — Source-frozen adapter training and evaluation

- [x] T900 Download the official Anew checkpoint and record URL, size, and SHA256.

  Evidence:
  - Source files: official Anew release URL `https://github.com/bytedance/AnewOmni/releases/download/init/model.ckpt`.
  - Target files: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_model.ckpt`; derived encoder state dict and `/tmp/anew_official_audit.txt`.
  - Commands: `curl -L --fail --retry 3 ...`; `stat -c '%s %n'`; `sha256sum`; source-environment `torch.load(...).state_dict()` extraction.
  - Tests: official payload type/key count, encoder dimensions, derived artifact size/hash, and current-model role-loader audit.
  - Result: Passed provenance. Official file is `722534016` bytes with SHA256 `961599acb617b8baf742755b4416322979507c9814ebd1a1e8582cff9c0a74c1`; it is `models.confidence.model.Confidence` with `786` state keys and official Anew encoder dimensions `512/64/6/8`.
  - Commit: pending.

- [x] T901 Audit PVB/Anew architecture compatibility, key translation, and coverage.

  Evidence:
  - Source files: official Anew state dict; previous fused/PVB state dict artifacts; `utils/checkpoint.py`.
  - Target files: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_encoder_state_dict.pt`; role audit reports; current fusion loader.
  - Commands: `python -m scripts.profile_operator` model construction; official role-loader audit with `min_coverage=0`; legacy role audit from T803; shape/key report extraction.
  - Tests: official and legacy PVB/Anew role reports, translation fixture tests, and full target suite.
  - Result: Official Anew maps to only `1/68` current Anew keys (`1.47%`), with `67` shape mismatches and `259` unexpected keys because the release is `512/64/6/8` while the requested fused model is `128/16/2/4`. The previous shape-matched fused state dict remains valid at PVB `150/150` and Anew `68/68`, with zero shape mismatches. The official checkpoint is provenance-only; the requested run will use the prior fused Anew state dict and will not resize or partially load the official model.
  - Commit: pending.

- [x] T902 Generate a block-complete PDBBind protein-only train/valid/test view.

  Evidence:
  - Source files: `/data/pvb_cross_dataset_20260810/blocks/pdbbind/{train,valid,test}_block`, `data/protein_view.py`, `data/mmap_dataset.py`.
  - Target files: `scripts/materialize_protein_view.py`; materialized views and manifests under `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/`.
  - Commands: `PYTHONPATH=. python scripts/materialize_protein_view.py --manifest .../materialization_manifest.json`.
  - Tests: full write/read verification in the materializer and exact-length profile on all three splits.
  - Result: 6413/367/167 records materialized; source IDs preserved by SHA256; all output `get_len` values equal filtered atom counts; all selected blocks are complete protein residue blocks.
  - Commit: pending.

- [x] T903 Validate split identity, residue completeness, metadata, bonds, and batch isolation.

  Evidence:
  - Source files: materialized split views, `data/block_metadata.py`, `data/collate.py`, `tests/test_block_metadata.py`.
  - Target files: `materialization_manifest.json`, `materialized_profile.json`, and `materialized_validation.json`.
  - Commands: `PYTHONPATH=. python scripts/profile_real_batches.py --dataset-root .../phase9/data --dataset pdbind_protein_only --max-records 0 --budgets 4000000 8000000`; three-sample collate/isolation assertions.
  - Tests: 28-test target suite plus full materialized split validation.
  - Result: explicit metadata on every output record, zero unsupported atoms/mixed blocks, valid remapped bonds, preserved IDs, and batch-isolation checks passed for train/valid/test.
  - Commit: pending.

- [x] T904 Generate a fused checkpoint provenance manifest for every target key.

  Evidence:
  - Source files: `utils/checkpoint.py`, `scripts/audit_fused_provenance.py`, and the constructed `anew_block` model.
  - Target files: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/source_frozen_provenance.json` and its stdout copy.
  - Commands: `python -m scripts.audit_fused_provenance --pvb-checkpoint .../pvb_state_dict.pt --anew-checkpoint .../legacy_fused_state_dict.pt`.
  - Tests: per-target-key source-role/source-key/shape/dtype/parameter/checksum audit.
  - Result: PVB `150/150` and Anew `68/68`; union `218` state keys / `217` parameter tensors / `9,090,374` parameters; complement `50` tensors / `1,858,692` parameters; all audit assertions passed. Manifest SHA256 is `f0371b437196f219f37feaba0d144a9461b7f986710bbe6991a46fa3aa973972`.
  - Commit: pending.

- [x] T905 Add a `source_frozen` stage that freezes the union of matched source keys.

  Evidence:
  - Source files: `utils/fusion_training.py`, `train.py`, and `trainer/abs_trainer.py`.
  - Target files: same implementation paths and `tests/test_training_stages.py`.
  - Commands: source-key union from both `CheckpointReport.matched_keys`; `configure_fusion_parameters(..., stage="source_frozen")`.
  - Tests: source-frozen key-union and exact optimizer-complement tests.
  - Result: every source-loaded target parameter has `requires_grad=False`; non-source keys are exposed for the optimizer audit.
  - Commit: pending.

- [x] T906 Restrict the optimizer to non-source-loaded parameters and print the complete key lists.

  Evidence:
  - Source files: `utils/fusion_training.py`, `scripts/audit_fused_provenance.py`.
  - Target files: provenance manifest and `scripts/profile_training_paths.py`.
  - Commands: optimizer ID-set audit and source-frozen real-batch profiler.
  - Tests: exact optimizer membership assertion.
  - Result: optimizer contains exactly the `50`-tensor / `1,858,692`-parameter source complement. The current `anew_block` path gives gradients to only the five projector/gate tensors; the other 45 complement tensors are structurally unused legacy PVB encoder/prior parameters and are reported explicitly rather than silently dropped.
  - Commit: pending.

- [x] T907 Add frozen checksum, gradient, and optimizer-membership tests.

  Evidence:
  - Source files: `utils/checkpoint.py`, `utils/fusion_training.py`, `tests/test_checkpoints.py`, `tests/test_training_stages.py`, and `scripts/source_frozen_overfit.py`.
  - Target files: source-key tracking, exact-complement unit test, and checksum smoke script.
  - Commands: `timeout 120 python -m unittest discover -s tests -v`.
  - Tests: all 28 target tests passed; source-frozen smoke checks finite gradients, optimizer complement, and bitwise source checksums.
  - Result: passed; all source-loaded tensors remain unchanged in the smoke run.
  - Commit: pending.

- [x] T908 Run source-frozen one-batch overfit and training smoke tests.

  Evidence:
  - Source files: materialized `train_block[0]`, PVB/Anew shape-matched state dictionaries, and fused model path.
  - Target files: `scripts/source_frozen_overfit.py`; `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/source_frozen_overfit_lr1e3.json`.
  - Commands: `CUDA_VISIBLE_DEVICES=5 python -m scripts.source_frozen_overfit ... --steps 20 --pvb-lr 1e-3 --projector-lr 1e-3`.
  - Tests: fixed stochastic bridge draw, finite loss/gradients, source checksum equality, and exact optimizer complement.
  - Result: `2147` atoms / `282` blocks / `4368` bonds; loss `1.0472314 → 1.0452697`; all checks passed. Artifact SHA256 is `556ad74993423212752d409464fdbda00884c8045f69f59e79ceab115cd640b1`.
  - Commit: pending.

- [x] T909 Train the adapter on PDBBind protein-only train and select by valid loss.

  Evidence:
  - Source files: `scripts/phase9_train_eval.py`, `data/protein_view.py`, `utils/fusion_training.py`, and the audited PVB/Anew role state dictionaries.
  - Target files: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/source_frozen_epoch1_best.ckpt`, `last.ckpt`, and `profiles/source_frozen_train_epoch1.json`.
  - Commands: `CUDA_VISIBLE_DEVICES=5 timeout 86400 python -m scripts.phase9_train_eval --mode train_fused --epochs 1 --batch-budget 4000000 --max-atoms-per-batch 4096 --max-items-per-batch 8 --pvb-lr 1e-3 --projector-lr 1e-3 --anew-lr 1e-5 ...`.
  - Tests: exact-length traversal, finite forward/backward gradients, source checksum audit, and exact optimizer-complement audit.
  - Result: all `6413/6413` train records were used in `6203` exact batches (`14,666,461` atoms); `3929` oversized records were explicit singleton batches, not dropped. Full valid `367/367` records were used for selection (`354` batches, `239` oversized singletons); best batch-mean valid loss is `1.0267511`. Source union `218` keys / `9,090,374` parameters remained bitwise unchanged; optimizer complement is exact (`50` tensors / `1,858,692` parameters). Elapsed time was `2483.96 s`.
  - Commit: pending.

- [x] T910 Evaluate the existing PVB checkpoint on complete original valid/test splits only.

  Evidence:
  - Source files: `scripts/phase9_train_eval.py`, the original PVB valid/test mmap splits, and `/output/pvb_cross_dataset_20260810/performance_v1/pvb/checkpoints/version_0/checkpoint/epoch24_step74921.ckpt`.
  - Target files: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/pvb_epoch24_valid_test.json`.
  - Commands: `CUDA_VISIBLE_DEVICES=5 timeout 86400 python -m scripts.phase9_train_eval --mode eval_pvb --pvb-checkpoint .../epoch24_step74921.ckpt --eval-seeds 20260810 20260811 20260812`.
  - Tests: complete original `pcqm4mv2`, `ani1x`, and `pdbbind` valid/test traversal; no training and no `max_batches` truncation.
  - Result: all original evaluation items were processed: valid `168929/285072/367`, test `168930/161913/167`; all three seeds completed. Artifact SHA256: `01f5cb806be59c7901e8ef3281726d082a82a802f86e9773689c1596ba616ab`.
  - Commit: pending.

- [x] T911 Evaluate PVB `off` and fused H-block on paired protein-only valid/test splits.

  Evidence:
  - Source files: `scripts/phase9_train_eval.py`, the exact materialized protein-only PDBBind views, the original PVB checkpoint, and the selected source-frozen fused checkpoint.
  - Target files: `profiles/pvb_off_protein_valid_test.json`, `profiles/fused_epoch1_valid_test.json`, and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/source_frozen_epoch1_best.ckpt`.
  - Commands: `... --mode eval_pvb_protein ...` and `... --mode eval_fused --resume-checkpoint .../source_frozen_epoch1_best.ckpt --eval-seeds 20260810 20260811 20260812`.
  - Tests: complete paired valid/test traversal, exact atom/batch counts, and three fixed seeds for both models.
  - Result: PVB-off processed valid/test `367/167` records (`354/155` batches); fused processed the same `367/167` records (`354/155` batches). PVB-off artifact SHA256 `90ad2e50d4dd1352c1c533e0f620cb9aa898674040b8cb9883d7e1d558ffa02d`; fused artifact SHA256 `5bd5de29862e7d0e68a421ea7e56e85e704f4dba643b80c4598f464aa3ced02a`.
  - Commit: pending.

- [x] T912 Report batch-compatible and atom-weighted loss, KL, velocity, and drift metrics.

  Evidence:
  - Source files: `scripts/phase9_train_eval.py` metric aggregation.
  - Target files: the three Phase 9 evaluation JSON artifacts and the metric tables in `PLAN.md`/`HANDOFF.md`.
  - Commands: JSON aggregation over `aggregate.{loss,kl,rec_vel,rec_drf}.{batch_mean,atom_weighted_mean}`.
  - Tests: every report has three seeds, mean/std, atom counts, batch counts, padded cost, and explicit oversized-singleton counts.
  - Result: complete original-PVB and paired protein-only metrics are recorded; no evaluation was silently truncated.
  - Commit: pending.

- [x] T913 Verify every source-loaded parameter is bitwise unchanged after training.

  Evidence:
  - Source files: `scripts/phase9_train_eval.py`, `utils/fusion_training.py`, and source-key provenance manifest.
  - Target files: `profiles/source_frozen_train_epoch1.json` and `checkpoints/source_frozen_epoch1_best.ckpt`.
  - Commands: source checksum comparison before/after training and checkpoint payload audit.
  - Tests: `218` source checksums equal before/after; checkpoint stores `218` before/after checksums with equality; optimizer ID set equals the exact `50`-tensor complement.
  - Result: passed. No source checksum mismatch; source parameters are stop-gradient/frozen throughout the run.
  - Commit: pending.

- [x] T914 Update PLAN.md, TASKS.md, HANDOFF.md, and DECISIONS.md with results and benchmarks.

  Evidence:
  - Source files: Phase 9 runner, result artifacts, checkpoint hashes, and source worktree audits.
  - Target files: all four live planning files plus `module/graph.py`, `scripts/phase9_train_eval.py`, and `tests/test_graph_empty_bonds.py`.
  - Commands: `timeout 120 python -m unittest discover -s tests -v`; `git diff --check`; source `git status --porcelain=v1`; artifact SHA256 audit.
  - Tests: 28 target tests passed; PVB and Anew source repositories remain clean; target dependency/source audit passed.
  - Result: Phase 9 gate passed. The selected one-epoch fused checkpoint and all complete valid/test reports are recorded. Fused H-block is currently worse than PVB-off on the paired PDBBind view; this is reported as an experimental result, not hidden as a success claim.
  - Commit: pending.

Gate P9: PASSED — checkpoint provenance, source freeze, complete valid selection, and one-time complete test evaluation all passed.

## Phase 10 — PVB-posterior-preserving H-block conditioning

- [x] T1000 Initialize Phase 10 documentation and provenance.
  Status: DONE

  Evidence:
  - Source files: `/workspace/PVB`, `/workspace/AnewOmni`, and target Git history.
  - Target files: `PLAN.md`, `TASKS.md`, `HANDOFF.md`, `DECISIONS.md`.
  - Commands: `enter-container`; `conda activate torch-ito`; target/source status and SHA audit; complete four-document read.
  - Tests: document initialization consistency; Phase 9 artifact preservation.
  - Result: Passed; target commit/source SHAs and clean source worktrees were recorded, all four documents were read completely, only the four live documents changed, and Phase 9 hashes were unchanged. Phase 10 directory was absent before work.
  - Commit: pending.

- [x] T1001 Reproduce and audit the Phase 9 loss decomposition.
  Status: DONE

  Evidence:
  - Source files: `module/interpolant_matcher.py`, `module/model.py`, `utils/checkpoint.py`, `scripts/phase9_train_eval.py`, `scripts/audit_phase9_loss.py`, and Phase 9 reports/manifests.
  - Target files: `scripts/audit_phase9_loss.py`, `PLAN.md`, `DECISIONS.md`, and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase9_loss_audit.json`.
  - Commands: `python -m py_compile scripts/audit_phase9_loss.py`; `python -m scripts.audit_phase9_loss --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase9_loss_audit.json`; `sha256sum` on the artifact.
  - Tests: formula recomputation, posterior source assertions, legacy role coverage audit, source-frozen manifest audit, and source revision capture.
  - Result: Passed. Maximum aggregate formula error is `1.24e-8`; batch-mean legacy/PVB KL ratio is `157.18x`, atom-weighted ratio is `157.97x`; legacy `rec_total` is lower by `0.078212` batch-mean while total loss is higher by `0.791090`. Anew `Wx_log_var` is loaded/frozen in the legacy path; PVB encoder/posterior keys are not loaded by the legacy PVB role and remain in its inactive complement. Artifact SHA256: `6ef70eed26b5cb4c3144047f657c4d664a77f67b715b07f3bf9a407b984a6273`.
  - Commit: pending.

- [x] T1002 Add a backward-compatible full-PVB checkpoint role.
  Status: DONE

  Evidence:
  - Source files: PVB `module/model.py` state-dict layout; `utils/checkpoint.py`; current fused model construction; and existing checkpoint tests.
  - Target files: `utils/checkpoint.py`, `scripts/profile_training_paths.py`, `train.py`, `config/train.yaml`, `tests/test_checkpoints.py`, `scripts/audit_pvb_full_checkpoint.py`, and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json`.
  - Commands: `python -m py_compile utils/checkpoint.py tests/test_checkpoints.py scripts/profile_training_paths.py train.py`; `python -m unittest -v tests.test_checkpoints`; `python -m scripts.audit_pvb_full_checkpoint --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json`; `sha256sum` on the coverage artifact; real checkpoint load and `train.build_dyvae_from_config` role smoke.
  - Tests: 7 checkpoint tests passed; real Phase 9 PVB checkpoint loaded with `pvb_full` coverage `195/195` (encoder 42, posterior 3, decoder 138, velocity 6, drift 6); no missing, unexpected, or shape-mismatched keys; Anew role remained `68/68`; the builder recorded 263 source-loaded keys and all PVB posterior keys.
  - Result: Passed. The existing `pvb` role remains decoder/output-head scoped and backward-compatible. `pvb_full` always requires 100% expected-key coverage, reports matched/missing/unexpected/shape-mismatched keys, and loads the complete PVB encoder, posterior heads, decoder, and output heads after constructing the fused model. Coverage artifact SHA256: `1b5210793836f7ccc69f38de41a18cab855285673279992c51af85c90b0ff976`.
  - Commit: pending.

- [x] T1003 Add an explicit corrected fusion mode.
  Status: DONE

  Evidence:
  - Source files: existing `off` and `anew_block` paths.
  - Target files: `module/model.py`, `config/train.yaml`, training/inference entrypoints, and tests.
  - Commands: `python -m py_compile module/model.py train.py scripts/profile_training_paths.py tests/test_fusion.py`; mode construction and configuration smoke.
  - Tests: `python -m unittest -v tests.test_fusion tests.test_checkpoints` (11 tests passed), including corrected-mode construction and legacy namespace/checkpoint fixtures.
  - Result: Passed. `anew_block_pvb_posterior` is accepted and constructs the Anew encoder/projector with a zero gate; `off` and legacy `anew_block` defaults are unchanged, and existing legacy checkpoint tests remain green.
  - Commit: pending.

- [x] T1004 Implement PVB posterior plus Anew H-block conditioning.
  Status: DONE

  Evidence:
  - Source files: PVB encoder/bridge and Anew `AnewBlockEncoder`.
  - Target files: `module/model.py`, `utils/fusion_training.py`, and `tests/test_fusion.py`.
  - Commands: corrected `_train`, `inference`, and `realization` route smoke; `python -m py_compile module/model.py utils/fusion_training.py tests/test_fusion.py`.
  - Tests: `python -m unittest -v tests.test_fusion` (5 tests passed); corrected inference/realization finite; mutating Anew `Wx_log_var` leaves corrected loss unchanged and its gradients absent.
  - Result: Passed. Corrected training/inference use the original PVB encoder/posterior and attach only projected/broadcast Anew `H_block` to the decoder; Anew `X_atom`, `X_block`, and `log_var_block` are retained as diagnostics, and `Wx_log_var` is disconnected from the loss. Fusion stages recognize the corrected mode.
  - Commit: pending.

- [x] T1005 Add complete gate-zero parity tests.
  Status: DONE

  Evidence:
  - Source files: PVB `off` and corrected fusion paths.
  - Target files: `tests/test_fusion.py` and parity fixtures.
  - Commands: `python -m py_compile tests/test_fusion.py`; `python -m unittest -v tests.test_fusion`; identical weights/inputs and Torch/NumPy seeds `20260810`–`20260812`.
  - Tests: posterior `x_rep`, KL, velocity, drift, total loss, inference source sample, decoder output, zero condition, and legacy `anew_block` reload.
  - Result: Passed. Six fusion tests passed; corrected gate-zero matches `off` within `1e-6` for PVB posterior sample/KL, both reconstruction terms, total loss, inference trajectory, decoder inputs/outputs, and zero condition. Legacy load remains `150/150` PVB plus `68/68` Anew.
  - Commit: pending.

- [x] T1006 Audit source freezing and optimizer membership.
  Status: DONE

  Evidence:
  - Source files: `utils/checkpoint.py`, `utils/fusion_training.py`, and PVB/Anew role reports.
  - Target files: `scripts/audit_fused_provenance.py`, `tests/test_training_stages.py`, and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/source_frozen_provenance.json`.
  - Commands: `python -m py_compile scripts/audit_fused_provenance.py tests/test_training_stages.py`; `python -m unittest -v tests.test_training_stages`; `python -m scripts.audit_fused_provenance --fusion-mode anew_block_pvb_posterior --pvb-role pvb_full ...`; `sha256sum` on the Phase 10 provenance artifact.
  - Tests: corrected-mode unit test; real `pvb_full`/Anew coverage, source-to-target bitwise checksums, exact source-frozen complement, optimizer IDs, and trainable/effective-gradient tensor list.
  - Result: Passed. `pvb_full=195/195`, Anew `68/68`; 263 source state tensors/262 source parameter tensors match checkpoint bytes; source-frozen leaves exactly five trainable projector/gate tensors (`33,281` parameters), one optimizer group, and no PVB encoder/posterior in the complement. Artifact SHA256: `c99aa39be4c39aa8592e79630ed4b9c33e913e8b4c427b1797c321535badf6ce`.
  - Commit: pending.

- [x] T1007 Add posterior and conditioning diagnostics.
  Status: DONE

  Evidence:
  - Source files: PVB encoder/posterior path, Anew `Wx_log_var` output, and Phase 9 metric/checkpoint utilities.
  - Target files: `module/model.py`, `utils/phase10_diagnostics.py`, `scripts/phase10_diagnostics.py`, `tests/test_phase10_diagnostics.py`, and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_diagnostics_record0.json`.
  - Commands: `python -m unittest -v tests.test_phase10_diagnostics`; corrected real-batch diagnostic with `pvb_full`, Phase 9 Anew checkpoint, record 0, and `--no-update`.
  - Tests: two diagnostic unit tests passed; real corrected batch produced finite PVB/Anew variance quantiles, KL, reconstruction, gate/condition, and gradient reports; Anew `Wx_log_var` gradient was absent.
  - Result: Passed. PVB log-var mean/std `-0.790057/0.026386`, PVB KL `0.006921472`; Anew block log-var mean/std `-2.074082/0.270299`, diagnostic KL `1.018065`; corrected loss `0.314263`, `rec_total=0.308613`, gate/condition norms zero at initialization, projector/gate gradient norm `6.44585e-4`, and Anew variance was diagnostic-only. Artifact SHA256: `6b2032350ca03d3839bd31e6b1001e04b2b7741db96d75b432b537c6f83641ea`.
  - Commit: pending.

- [x] T1008 Run focused tests and smoke tests.
  Status: DONE

  Evidence:
  - Source files: target tests, `scripts/protein_smoke.py`, `train.py`, and `infer_prot.py`.
  - Target files: `scripts/protein_smoke.py`, T1008 logs under `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/`, and the four live documents.
  - Commands: `python -m unittest discover -s tests -v`; `python -m compileall -q .`; `python train.py --help`; `python infer_prot.py --help`; `CUDA_VISIBLE_DEVICES=5 python -m scripts.protein_smoke --device cuda --fusion-mode {off,anew_block,anew_block_pvb_posterior}`; source status audit.
  - Tests: 36 unit tests passed; compile and both CLI help commands passed; all three smoke paths returned finite train/inference results and inference shape `[16, 3]`; PVB and Anew source worktrees were clean.
  - Result: Passed after a minimal compatibility fix: the first corrected-mode smoke exposed that `protein_smoke.py` still allowed only the two legacy choices; its parser was extended without changing defaults. Smoke losses were `off=12.922840`, `anew_block=4.031592`, and `anew_block_pvb_posterior=29.173777`. Unit-test log SHA256 `a894bb52be44fd3c2e352db3d68158851709869bee1314035d45b86569c5d204`; train-help SHA256 `0ec56796a2d9ea77710ed55dce98e8359ae7c576a2c127d3568c51d8a74597a7`; infer-help SHA256 `f7832bbab3d41ba2c9759a2f1b45cc6760bbdaee9c2b412b448ba3e14dced837`.
  - Commit: pending.

- [x] T1009 Run a fixed-real-batch overfit/checksum gate.
  Status: DONE

  Evidence:
  - Source files: exact-length protein-only materialized data, `scripts/source_frozen_overfit.py`, Phase 9 runner utilities, and corrected source-freezing helpers.
  - Target files: `scripts/source_frozen_overfit.py` and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/source_frozen_overfit_corrected.json`.
  - Commands: corrected fixed batch with `pvb_full`, current shape-matched Anew role, 20 steps, projector/gate LR `1e-3`, and `CUDA_VISIBLE_DEVICES=5`.
  - Tests: finite loss/gradients; exact source checksum and optimizer complement; nonzero gate gradient; projector gradient after step 0; decreasing `rec_total`; PVB-like KL; no dropped/truncated atoms; Anew variance absent from the gradient graph.
  - Result: Passed. Record 0 has `2147` atoms, `282` blocks, `4368` bonds, and exact dataset/raw/view atom counts. `rec_total` decreased `0.308612682 → 0.307434760` over 20 steps; PVB KL stayed near `0.007063`. `pvb_full=195/195` and Anew `68/68`; five trainable adapter tensors (`33,281` parameters) form the exact optimizer complement; gate gradients occur at steps `0–19`, projector gradients after the initial update at steps `1–19`; source checksum mismatches `0`; no Anew `Wx_log_var` gradient. Artifact SHA256: `ed51fda34ea1ec81aa4ee8642bfe9051892cc41b6f31408727145aebb7acb4e2`.
  - Commit: pending.

- [x] T1010 Train the corrected adapter.
  Status: DONE

  Evidence:
  - Source files: `scripts/phase9_train_eval.py`, `scripts/phase10_train_eval.py`, `data/protein_view.py`, and the exact Phase 9 materialized protein-only views.
  - Target files: `scripts/phase10_train_eval.py`, `scripts/phase10_training_report.py`, `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/train_corrected.log`, and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_train_interrupted.json`.
  - Commands: `CUDA_VISIBLE_DEVICES=5 python -m scripts.phase10_train_eval ... --epochs 5 --min-epochs 3 --patience 2 --projector-lr 1e-3 --grad-clip 1.0`; `python -m scripts.phase10_training_report ...`.
  - Tests: complete train traversal `6413/6413` in each of four completed epochs (`6200/6200/6200/6200` steps; `14,660,149` atoms per epoch), valid traversal `367/367` (`354` batches, `847,978` atoms, `239` oversized singleton batches), source checksums, frozen-source optimizer membership, and no test use.
  - Result: Passed the minimum three-epoch protocol with four complete validation epochs. Valid batch-mean `rec_total` was `0.222021`, `0.221780`, `0.222065`, `0.221543`; best epoch `3`, global step `24797`, with PVB-like batch KL `0.006957666`. The long-running session stopped during the next training epoch before validation; this is recorded explicitly and does not invalidate the already-selected valid checkpoint. Source checksums remained unchanged. Training report SHA256: `7cd906217a6446d9e9bdc18917f2731323f2431f77538d81fc5b6bd1e10b9dc6`.
  - Commit: pending.

- [x] T1011 Lock the best checkpoint using valid only.
  Status: DONE

  Evidence:
  - Source files: `scripts/phase10_lock_checkpoint.py`, Phase 10 best checkpoint payload, and the complete validation reports.
  - Target files: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/anew_block_pvb_posterior_best.ckpt` and `phase10_best.lock.json`.
  - Commands: `python -m scripts.phase10_lock_checkpoint --checkpoint .../anew_block_pvb_posterior_best.ckpt --output .../phase10_best.lock.json ...`.
  - Tests: valid-only metric consistency, `pvb_full=195/195`, Anew `68/68`, source checksum before/after equality, no test payload, and immutable checkpoint SHA.
  - Result: Passed. Locked epoch `3`, global step `24797`, valid batch-mean `rec_total=0.2215431160951233`; checkpoint SHA256 `5ad3b769b602037da2dd47889592a53ed6ff4f66cf4fde162aeae3a180e4d204`; lock SHA256 `b9d574ad81c637e000610142f9d4a4150b4451157b8f00dcbf451e6b81444e09`. `test_evaluated=false` before T1012.
  - Commit: pending.

- [x] T1012 Run the one-time paired evaluation.
  Status: DONE

  Evidence:
  - Source files: scripts/phase9_train_eval.py, scripts/phase10_paired_eval.py, scripts/recover_phase10_paired_eval.py, PVB off, Phase 9 legacy fused checkpoint, and the locked Phase 10 checkpoint.
  - Target files: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_paired_valid_test.json, phase10_paired_eval.log, and phase10_paired_recovery.log.
  - Commands: CUDA_VISIBLE_DEVICES=5 python -m scripts.phase10_paired_eval --device cuda:0 ... --eval-seeds 20260810 20260811 20260812; then recovery from the completed evaluator log without rerunning any split.
  - Tests: all six traversals completed; valid 367/367 records, 354 batches, 847978 atoms, 239 oversized singleton batches; test 167/167 records, 155 batches, 334142 atoms, 70 oversized singleton batches; identical counts for all three models.
  - Result: Passed with an honest recovery. The evaluator completed PVB off, Phase 9 legacy, and Phase 10 corrected on valid then test, but its final JSON writer failed on tensor-valued resume metadata after all six aggregate lines were emitted. The recovery report did not rerun evaluation. Valid batch means (loss/KL/rec_vel/rec_drf/rec_total) were PVB off (0.273862/0.006958/0.102848/0.165448/0.268296), legacy (1.064952/1.093584/0.064430/0.125655/0.190084), corrected (0.264250/0.006958/0.099335/0.159349/0.258684). Test batch means were PVB off (0.255278/0.007027/0.098625/0.151031/0.249657), legacy (1.043928/1.089811/0.060861/0.111218/0.172079), corrected (0.246216/0.007027/0.095238/0.145356/0.240595). Atom-weighted metrics are preserved in the report. Report SHA256: 9653534757efc8323db1538fcdd1993c9c36f16c95a58559b912ee1252006c52; evaluation log SHA256: 2ee28bd2638631304c53ebb57caf6cc57a20e41d2d074f7f2a82aba1ad316af5. The corrected KL remains PVB-like; reconstruction and total loss improve over PVB off, while legacy remains KL-dominated. Per-seed detail was not reconstituted after the writer failure; aggregate mean/std and fixed seeds are recorded explicitly.
  - Commit: pending.

- [x] T1013 Run final audits.
  Status: DONE

  Evidence:
  - Source files: target/source Git state, Phase 9/10 manifests, locked checkpoint, paired evaluation report, and runtime dependency scan.
  - Target files: `scripts/audit_phase10_final.py` and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_final_audit.json`.
  - Commands: `python -m unittest discover -s tests -v`; `python -m compileall -q .`; `python train.py --help`; `python infer_prot.py --help`; `python -m scripts.audit_phase10_final --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_final_audit.json`; `git diff --check`; source status and artifact hash audits.
  - Tests: 36 target tests passed; compile and both CLI help commands passed; final audit passed; PVB/Anew source worktrees are clean; protected Phase 9 hashes match; Phase 10 checkpoint/report coverage is complete; no runtime sibling-repository dependency was found.
  - Result: Passed. Phase 9 artifacts are unchanged, including checkpoint SHA256 `c9df6928268c4a5f27779067b83703af1a15d92f187f570bb454baa2441d57` and valid/test report SHAs `90ad2e50d4dd1352c1c533e0f620cb9aa898674040b8cb9883d7e1d558ffa02d` and `5bd5de29862e7d0e68a421ea7e56e85e704f4dba643b80c4598f464aa3ced02a`. Final audit SHA256: `e48d25108041f0948d7cbb94acd5dada53694d3b0d3da46269aeb2541fdfd050`. The current shape-matched Anew 128/2-layer limitation and aggregate-only paired-report recovery limitation remain explicit.
  - Commit: pending.

- [x] T1014 Close Phase 10 documentation.
  Status: DONE

  Evidence:
  - Source files: Phase 10 audit/report/checkpoint manifests, paired valid/test report, current target changes, and the official Anew provenance audit.
  - Target files: PLAN.md, TASKS.md, HANDOFF.md, DECISIONS.md, and /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_final_audit.json.
  - Commands: final live-document review; git diff --check; python -m unittest discover -s tests -v; python -m scripts.audit_phase10_final --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_final_audit.json; post-close timeout 900 python -m unittest discover -s tests -v.
  - Tests: 36 target tests passed after the documentation close; post-close unit-log SHA256 is 7818b71442c27fd36433b1014dd94b94ed5fa8273c59378cefd1673fc43aa7af; final audit and git diff --check passed; all Phase 10 checkpoint/report coverage and paired-count checks passed; PVB/Anew source worktrees are clean; no task remains IN_PROGRESS; protected Phase 9 artifacts remain unchanged.
  - Result: Passed. Gate P10 is closed. The corrected mode preserves the PVB posterior, passes complete gate-zero parity and source-freezing gates, decreases fixed-batch reconstruction, and improves paired aggregate valid/test reconstruction and total loss over PVB off while keeping KL PVB-like. The aggregate-only per-seed reporting limitation is retained explicitly. Phase 11 official-Anew alignment is proposed only and is not implemented.
  - Commit: pending.

Gate P10: PASSED — pvb_full coverage, corrected posterior contract, complete stochastic parity, diagnostic-only Anew variance, exact source freeze/optimizer membership, fixed-batch overfit, valid-only lock, paired evaluation, final audit, legacy reproducibility, and Phase 9 immutability all have evidence.
