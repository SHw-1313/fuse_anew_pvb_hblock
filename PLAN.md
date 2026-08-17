# Plan

## 1. Objective

Luna should build a self-contained fused repository based on PVB that uses AnewOmni’s real block representation:

```text
Anew atom embedding/EPT
    → H_atom
    → Anew atom-to-block pooling
    → H_block
    → project and broadcast to atoms
    → zero-gated conditioning of PVB decoder
```

The first stable implementation must not replace PVB’s coordinate mean with Anew’s atom coordinate output.

Initial coordinate contract:

```python
x_mu = x0
```

Anew’s `H_block` conditions PVB’s velocity/drift decoder. Anew’s `X_atom` and `X_block` may be returned for diagnostics, but must not alter the bridge coordinates during the first milestone.

## 2. Repository roles

| Repository | Role |
| --- | --- |
| `/workspace/PVB` | Base repository and source of training, data, bridge, decoder, inference, evaluation, and trainer code |
| `/workspace/AnewOmni` | Source of BlockEmbedding, EPT, graph utilities, pooling, and checkpoint semantics |
| `/workspace/fuse_anew_pvb_hblock` | Self-contained output repository; the only writable project |

The target starts as a clean copy of PVB, excluding:

* `.git`
* checkpoints
* datasets
* logs
* generated results
* caches
* virtual environments

The target existed before work and was inspected before bootstrapping; it contained only an empty `agents/` directory and was not overwritten blindly.

## 3. Source-reuse rule

Before implementing any function:

1. Search both source repositories with `rg`.
2. Determine whether equivalent code already exists.
3. Copy or minimally adapt existing code whenever possible.
4. Record the original repository, commit SHA, source path, and target path.
5. Preserve copyright and license headers.
6. Keep semantic modifications separate from import/path modifications.
7. Do not use runtime `sys.path` injection to import the sibling Anew repository.

The output must eventually be self-contained.

Allowed new code is limited primarily to:

* PVB/Anew interface glue
* block metadata collation
* projection/gating layers
* checkpoint key translation
* tests and profiling scripts

## 4. Required source map

Luna must verify these paths against the actual source SHAs before copying.

### PVB sources

| Purpose | Source |
| --- | --- |
| Base model and bridge | `/workspace/PVB/module/model.py` |
| PVB equivariant decoder | `/workspace/PVB/module/torchmd_et.py` |
| Graph construction | `/workspace/PVB/module/graph.py` |
| Interpolant/bridge objective | `/workspace/PVB/module/interpolant_matcher.py` |
| Batch collation | `/workspace/PVB/data/collate.py` |
| Dataset preprocessing | `/workspace/PVB/data/*_dataset.py` |
| Training entrypoint | `/workspace/PVB/train.py` |
| Base trainer | `/workspace/PVB/trainer/abs_trainer.py` |
| Dynamic/MD trainer | `/workspace/PVB/trainer/dynamic_trainer.py` |
| Configuration | `/workspace/PVB/config/train.yaml` |
| Vocabulary mapping | `/workspace/PVB/utils/bio_utils.py` |

### AnewOmni sources

| Purpose | Source |
| --- | --- |
| Correct encoder and block-pooling reference | `/workspace/AnewOmni/models/IterVAE/model_edge.py` |
| BlockEmbedding | `/workspace/AnewOmni/models/modules/nn.py` |
| EPT implementation | `/workspace/AnewOmni/models/modules/EPT/ept.py` |
| EPT radial basis | `/workspace/AnewOmni/models/modules/EPT/radial_basis.py` |
| Block/unit edge construction | `/workspace/AnewOmni/models/modules/GET/tools.py` |
| Stable norm and EPT batching | `/workspace/AnewOmni/utils/nn_utils.py` |
| Variance-preserving block pooling | `/workspace/AnewOmni/utils/gnn_utils.py` |
| Registration dependency, if retained | `/workspace/AnewOmni/utils/register.py` |
| Official architecture settings | `/workspace/AnewOmni/configs/train_vae.yaml` |
| License | `/workspace/AnewOmni/LICENSE` |

Vendor only the minimal required Anew files under:

```text
third_party/anewomni/
```

Prefer copying complete source files and changing imports to package-relative imports. Avoid manually retyping EPT, radial-basis, graph, or pooling implementations.

## 5. Architecture contract

`AnewBlockEncoder.forward()` should consume explicit metadata and return:

```python
{
    "H_atom": Tensor[N_atom, anew_hidden],
    "X_atom": Tensor[N_atom, 3],
    "H_block": Tensor[N_block, anew_hidden],
    "X_block": Tensor[N_block, 3],
    "log_var_block": Tensor[N_block, 1],
    "atom_block_id": Tensor[N_atom],
    "block_batch": Tensor[N_block],
    "block_lengths": Tensor[N_block],
}
```

Use Anew’s original pooling semantics:

```python
H_block = scatter_sum(H_atom, atom_block_id, dim=0)
H_block = H_block / sqrt(block_lengths).unsqueeze(-1)

X_block = scatter_mean(X_atom, atom_block_id, dim=0)
```

Decoder conditioning:

```python
block_condition = block_projection(H_block)[atom_block_id]
x = x + tanh(block_gate) * block_condition
```

Requirements:

* `block_projection`: `LayerNorm → Linear`
* output dimension equals PVB decoder hidden dimension
* `block_gate` initialized to zero
* gate-zero mode must preserve PVB decoder behavior
* conditioning is applied consistently to both PVB cross-attention branches

## 6. Initial scope

The first milestone is protein-only.

PVB residue IDs and Anew amino-acid IDs can be mapped explicitly after verifying their orders in both repositories.

Do not reuse the previous arbitrary offset mapping for molecular blocks. Anew ligand blocks use a learned fragment vocabulary, so PVB element-level block IDs are not semantically interchangeable with Anew fragment IDs.

For unsupported non-protein blocks, fail with a clear error rather than silently mapping them incorrectly.

## 7. Implementation phases

### Phase 0: provenance and untouched PVB baseline

* Record source SHAs and worktree status.
* Bootstrap target from PVB, excluding runtime artifacts.
* Verify unchanged target reproduces PVB tests/imports.
* Create profiling and one-batch-overfit scripts.
* Record baseline timing, memory, and losses.

Stop if the copied baseline cannot reproduce PVB.

### Phase 1: vendor Anew code

* Copy minimal Anew implementation into `third_party/anewomni`.
* Preserve headers and license.
* Change only imports/registration needed for namespacing.
* Add import and numerical parity tests.

Do not implement a simplified EPT from memory.

### Phase 2: explicit block metadata

Add batch fields:

```text
atom_block_id
block_type
block_batch
block_lengths
```

Block membership must not be inferred from floating-point `b0` comparisons inside the GPU forward pass.

Prefer generating block IDs during preprocessing. A compatibility fallback may derive them in CPU collation for old datasets, with a warning.

Cropping must preserve complete residue blocks.

### Phase 3: block encoder

* Adapt Anew’s encoder path from `model_edge.py`.
* Return both atom and pooled block representations.
* Preserve PVB’s initial `x_mu = x0`.
* Do not use Anew `X_atom` as PVB’s source coordinate.
* Remove the PVB encoder graph calculation when it is unused in block-fusion mode.

### Phase 4: zero-gated decoder conditioning

* Add the projected/broadcast block condition to PVB.
* Implement `fusion.mode`:

  * `off`
  * `anew_block`

* Optional diagnostic-only compatibility:

  * `anew_atom_legacy`

* Prove gate-zero decoder parity.

### Phase 5: checkpoint migration

Use separate configuration fields:

```yaml
model:
  pvb_checkpoint: null
  anew_checkpoint: null
  resume_checkpoint: null
```

Always construct the fused model first.

* `pvb_checkpoint`: load compatible PVB decoder and heads
* `anew_checkpoint`: load compatible Anew embedding/EPT parameters
* `resume_checkpoint`: restore complete fused training state

Print:

* matched keys
* missing keys
* unexpected keys
* shape mismatches
* coverage percentage

Never silently load an entire serialized original PVB object as the fused model.

### Phase 6: staged training

Stage A:

* pretrained PVB decoder/head
* pretrained Anew encoder
* Anew frozen
* train block projector, gate, and PVB decoder/head

Stage B:

* unfreeze only the last one or two EPT layers
* separate learning rates
* record per-module gradient norms

Stage C, only after stable convergence:

* introduce a separately gated `X_block` coordinate residual
* keep coordinate gate initialized to zero
* add block-level KL, warmup, and free bits
* compare against invariant-block-conditioning-only ablation

### Phase 7: performance

Optimize only after correctness:

1. Remove redundant graph construction.
2. Precompute static topology/block metadata.
3. Add BF16 autocast where numerically safe.
4. Benchmark Anew efficient attention if xFormers is installed.
5. Use an `n*n`-aware dynamic batch budget.
6. If dense EPT remains too slow, add a separately named `block_sparse` encoder.

The sparse encoder is an approximation and must not replace the faithful Anew path silently.

## 8. Acceptance gates

Required before completion:

* untouched PVB baseline runs
* explicit block metadata is correct across multi-sample batches
* block pooling matches Anew reference behavior
* gate-zero decoder parity passes
* batch isolation passes
* SE(3) tests pass
* forward and backward gradients are finite
* block projector and gate receive gradients
* one fixed batch shows meaningful loss decrease
* checkpoint coverage is reported
* baseline and fused profiling results are recorded
* protein inference smoke test passes

## 9. Phase 8 — 2000-atom performance diagnosis

The existing 2000-atom table is diagnostic only. It used too few measured
steps and one run showed non-monotonic PVB timing, so it cannot establish a
performance discontinuity. The current working hypothesis is that faithful
Anew EPT still performs dense atom-level attention and that the apparent jump
comes from noisy sampling plus quadratic attention cost.

* Update profiling to use CUDA events, at least 10 warmup steps, and at least
  20 measured steps; report p50, p90, min/max, and coefficient of variation.
* Measure 512, 1024, 1536, 1800, 2000, and 2048 atoms on synthetic batches.
* Profile real PDBBind batches by atom count, maximum padded length, padding
  ratio, block count, and edge count.
* Separate data/H2D, graph, Anew encoder, projection, decoder, backward, and
  optimizer-step timings; record allocated and reserved peak memory.
* Compare PVB `off`, Anew H-block with all parameters trainable, source-frozen
  adapter training, and forward-only execution.
* Use PyTorch profiler on representative 1024- and 2000-atom batches to
  attribute time and memory to `graph_to_batch_nx`, dense `einsum`, softmax,
  and `[B,N,N]` tensors.
* Treat `sparse_k=3` as block-edge expansion sparsity only; it does not make
  EPT self-attention sparse.
* Do not change faithful EPT before the root cause is measured. If needed,
  introduce `block_sparse` as a separately named approximation.

Current execution evidence:

* T800 completed the profiler protocol update with CUDA-event timing, warmup,
  repeated measurements, and memory statistics.
* T801 completed 12 synthetic CUDA profiles on an A100 at 512/1024/1536/1800/
  2000/2048 atoms. The H-block step curve is smooth through 2000 atoms; the
  measured 1800→2000 and 2000→2048 p50 increases are 7.44% and 3.59%.
* This does not identify the cause of the earlier apparent slowdown. Continue
  with real PDBBind batch-shape distributions, source-frozen/forward-only
  comparisons, and operator-level profiling before changing the faithful path.

* T802 inspected 100 records per PDBBind split. Atom-count p50 is
  `2228.5/2193/1738` for train/valid/test; unsupported element-block atom
  p50 is `26/25/29`, and every inspected record uses the CPU legacy metadata
  fallback.
* At the existing `n*n` budget `2000`, dynamic batching forms zero groups
  and skips every PDBBind record. Budgets `4e6` and `8e6` form usable groups,
  but must be validated against GPU memory before training.
* Raw PDBBind records contain protein residue blocks plus unsupported ligand
  element blocks. The fused path must use a separately validated protein-only
  view; raw mixed records must not be silently passed to Anew.
* T803 completed a real protein-only comparison on train record 0 (`2147` atoms,
  `282` blocks, `4368` bonds). The checkpoint audit matched PVB `150/150` and
  Anew `68/68` keys with no shape mismatches. All-trainable, strict adapter,
  and forward-only modes were finite; adapter used `33,281` trainable params,
  `9.53 GiB` peak allocation, and `114.1 ms` per measured train step.

* T804 completed operator-level profiling on real protein-only representatives near the requested scales: `911` atoms/`109` blocks and `2147` atoms/`282` blocks. The traces are stored under `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/operator/`.
  The latter used padded EPT length `2152` and showed peak allocation `14.75 GiB`.
* T805 identified the dominant scaling mechanism. Anew's `knn_edges` first evaluates block-level fully-connected candidates, while EPT's `SelfAttnLayer` constructs dense `[B, heads, N, N]` attention. The measured attention shapes were `[1,4,912,912]` and `[1,4,2152,2152]`; representative softmax/batched-attention allocations grew from about `26.6/53.2 MiB` to `148.2/296.4 MiB`.
  Anew block-KNN was about `51.9/53.9 ms`, Anew EPT about `35.5/65.9 ms`, and peak allocation was `6.31/14.75 GiB` at `911/2147` atoms. PVB graph/decoder and backward also scale with the graph size.
* Therefore the earlier apparent 2000-atom jump is not yet a hardware or algorithmic threshold: T801's repeated synthetic curve is smooth. The likely contributors are quadratic dense attention, repeated block-candidate construction, and shape/padding or dynamic-batch transitions. T806 must audit these transitions before deciding on a separately named `block_sparse` approximation.
* T806 completed that audit. The mmap `get_len` property underestimates materialized atom counts in all 100 sampled records per split; median estimator/actual ratios are `0.726/0.742/0.738` for train/valid/test. `DynamicBatchWrapper` therefore can admit larger real graphs than its `n*n` budget implies, and `graph_to_batch_nx` pads every sample in a batch to the largest atom count rounded to a multiple of eight.
* Phase 8 gate passed. The first correction is to materialize the protein-only view with exact atom-count properties and an explicit max-padded-N/budget audit. No faithful EPT rewrite or `block_sparse` approximation is introduced before that correction is benchmarked.

## 10. Phase 9 — source-frozen adapter training and evaluation

The requested evaluation uses an existing PVB checkpoint and a fused model
initialized from PVB and Anew checkpoints. Parameters loaded from either
source checkpoint are frozen with stop-gradient. Only parameters absent from
both loaded-key sets may be optimized.

Fixed inputs and artifacts:

```text
PVB checkpoint:
/output/pvb_cross_dataset_20260810/performance_v1/pvb/checkpoints/version_0/checkpoint/epoch24_step74921.ckpt

Anew checkpoint:
https://github.com/bytedance/AnewOmni/releases/download/init/model.ckpt

Container dataset:
/data/pvb_cross_dataset_20260810

Results:
/output/pvb_cross_dataset_20260810/hblock_adapter_v1
```

* Download the official Anew checkpoint only after recording its URL, file
  size, and SHA256; stop if its architecture or key coverage is insufficient.
* T900/T901 found that the official release is `Confidence` with Anew encoder
  dimensions `512/64/6/8`; against the current `128/16/2/4` fused model it
  matches only `1/68` Anew keys with `67` shape mismatches. The requested run
  therefore uses the prior shape-matched fused state dict, audited at PVB 150/150 and Anew 68/68.
* Reconstruct the fused model first, then load PVB and Anew state dictionaries
  separately. Record the source provenance of every target key.
* The target now records the union of matched source keys on the constructed
  model and supports `fusion.stage=source_frozen`; the strict `adapter` stage
  remains available for the requested projector/gate-only run. The durable
  profiler and protein-only CPU view are in `scripts/` and `data/`.
* Freeze the union of the two loaders' actual `matched_keys`; do not infer
  the freeze boundary from module names alone.
* Restrict the optimizer to the complement of the loaded-key union. Report
  frozen, trainable, unmatched, missing, unexpected, and shape-mismatched
  keys explicitly.
* Use a block-complete PDBBind protein-only derived view for fused training.
  ANI1x and PCQM4Mv2 are molecular inputs and cannot be silently mapped to
  Anew protein/fragment IDs in this milestone.
* Preserve the original train/valid/test split IDs. If the existing crop does
  not prove complete residue blocks, regenerate the view from its raw PDBBind
  manifest with block-aware cropping rather than guessing from partial atoms.
* Rewrite the materialized split index properties to the exact protein-only
  atom count used by `DynamicBatchWrapper`; stale upstream `get_len` values
  must not determine the `n*n` budget. Record actual versus padded attention
  work in the materialization and profile manifests.
* T902/T903 passed the Phase 9 data gate:
  - all `6413/367/167` train/valid/test records preserve source IDs;
  100% of output records have explicit metadata and complete protein blocks;
  filtered atom-count p50/max is `2190/6855`, `2217/6443`, and `1711/6083`;
  all remapped bond indices and three-sample batch-isolation checks pass.
  The exact-length profile reports candidate `n*n` budgets of `4e6/8e6`;
  these remain profiling choices until source-frozen training smoke determines
  a stable non-skipping budget. Representative padded-attention ratios range
  from 0.46 to 1.0, so max-padded-N remains a tracked cost.
* T904–T908 passed the source-frozen pre-training gate. The matched source
  union is `218` state keys / `217` parameter tensors / `9,090,374` parameters;
  its exact complement is `50` tensors / `1,858,692` parameters. Only the
  projector/gate's five tensors receive gradients in the current `anew_block`
  forward path; 45 legacy PVB encoder/prior tensors are structurally unused
  and remain explicitly reported in the complement audit.
* The source-frozen real-batch smoke preserved every source checksum and had
  finite gradients; fixed-batch loss decreased from `1.0472314` to `1.0452697`
  in 20 steps. Formal training may now start, but must record the inactive
  complement separately from the effective gradient-bearing set.
* T909 completed one full source-frozen epoch with an exact padded-cost runner:
  all `6413` train records were used in `6203` batches, including `3929`
  oversized singleton batches. Full valid selected the best checkpoint at
  batch-mean loss `1.0267511`; source checksums stayed unchanged.
* The formal runner is `scripts/phase9_train_eval.py`. It does not use the
  legacy `DynamicBatchWrapper` skip behavior: exact materialized lengths and
  max-padded-N cost form batches, while every over-budget record is emitted as
  an explicit singleton and counted. A target-side empty-bond compatibility
  fix in `module/graph.py` returns a zero bond-type mask for bondless records;
  the PVB source repository remains unchanged.
* Train only on the protein-only PDBBind train view. Use valid for model
  selection and run test only after the checkpoint is locked.
* Evaluate the existing PVB checkpoint without retraining: valid/test only on
  the complete original evaluation splits. Do not use PVB training to tune
  the fused adapter.
* Also evaluate PVB `off` and the fused model on the same protein-only
  PDBBind valid/test view for a paired comparison.
* Use fixed evaluation seeds `20260810`, `20260811`, and `20260812`, and
  report mean/std for loss, KL, velocity, and drift metrics. Do not truncate
  formal valid/test evaluation with `max_batches`.

### Phase 9 results and gate

* T909 completed one source-frozen epoch on all `6413` protein-only train
  records: `6203` exact batches, `14,666,461` atoms, and `3929` explicit
  oversized singleton batches. Valid selection used all `367` records in
  `354` batches; the selected checkpoint has batch-mean valid loss
  `1.0267511`, and all `218` source checksums remained unchanged.
* T910 completed evaluation of the original PVB checkpoint on the complete
  original valid/test splits only: valid `168929/285072/367` records for
  `pcqm4mv2/ani1x/pdbbind`, and test `168930/161913/167`. No training or
  evaluation truncation was used.
* T911 completed the paired protein-only comparison on the same derived
  PDBBind views: PVB `off` and fused H-block each processed valid/test
  `367/167` records, with `354/155` exact batches.
* T912 recorded three-seed mean/std for batch-mean and atom-weighted loss,
  KL, velocity, and drift metrics. The complete JSON reports are the source
  of truth; the paired summary is included below.

| Model / split / aggregation | Loss | KL | Rec. velocity | Rec. drift |
| --- | ---: | ---: | ---: | ---: |
| PVB `off` / valid / batch | `0.273862 ± 0.044841` | `0.006958 ± 1.8e-11` | `0.102848 ± 0.008595` | `0.165448 ± 0.036252` |
| PVB `off` / valid / atom | `0.265062 ± 0.027623` | `0.006927 ± 1.0e-11` | `0.100691 ± 0.004793` | `0.158829 ± 0.022842` |
| Fused H-block / valid / batch | `1.064952 ± 0.043088` | `1.093584 ± 3.8e-7` | `0.064429 ± 0.007411` | `0.125655 ± 0.035680` |
| Fused H-block / valid / atom | `1.057333 ± 0.026522` | `1.094271 ± 2.5e-7` | `0.062389 ± 0.003938` | `0.119527 ± 0.022584` |
| PVB `off` / test / batch | `0.255278 ± 0.015616` | `0.007027 ± 7.3e-11` | `0.098625 ± 0.003249` | `0.151031 ± 0.012400` |
| PVB `off` / test / atom | `0.253030 ± 0.012324` | `0.006980 ± 6.8e-11` | `0.098650 ± 0.003609` | `0.148796 ± 0.008782` |
| Fused H-block / test / batch | `1.043927 ± 0.014116` | `1.089811 ± 1.4e-6` | `0.060861 ± 0.002426` | `0.111217 ± 0.011725` |
| Fused H-block / test / atom | `1.042369 ± 0.012838` | `1.091011 ± 1.2e-6` | `0.060681 ± 0.003851` | `0.108879 ± 0.009128` |

The fused H-block loss is substantially worse than the paired PVB `off`
loss after this one-epoch adapter run. This is an honest diagnostic result,
not evidence of an improvement claim; the source-frozen protocol itself
passed and further optimization or architecture work remains a separate task.

The complete original-PVB loss summary is:

| Original split/source | Records | Batch loss | Atom-weighted loss |
| --- | ---: | ---: | ---: |
| valid / pcqm4mv2 | `168929` | `0.327371 ± 0.000987` | `0.325666 ± 0.000621` |
| valid / ani1x | `285072` | `0.545341 ± 0.002223` | `0.519560 ± 0.001362` |
| valid / pdbbind | `367` | `0.498044 ± 0.092286` | `0.476941 ± 0.058767` |
| test / pcqm4mv2 | `168930` | `0.327538 ± 0.000820` | `0.325933 ± 0.000857` |
| test / ani1x | `161913` | `0.535800 ± 0.000697` | `0.509813 ± 0.002976` |
| test / pdbbind | `167` | `0.461913 ± 0.025711` | `0.455625 ± 0.016793` |

Phase 9 artifacts:

```text
Best fused checkpoint:
/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/source_frozen_epoch1_best.ckpt
SHA256: c9df6928268c4c8a5f27779067b83703af1a15d92f187f570bb454baa2441d57

Training: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/source_frozen_train_epoch1.json
Original PVB valid/test: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/pvb_epoch24_valid_test.json
Paired PVB off: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/pvb_off_protein_valid_test.json
Paired fused: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/fused_epoch1_valid_test.json
```

T909–T914 are complete and the Phase 9 gate is **PASSED**: provenance,
source freezing, exact optimizer membership, complete valid selection, and
one-time complete test evaluation all passed.

Initial adapter-training defaults:

```yaml
model:
  pvb_checkpoint: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt
  anew_checkpoint: /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt
  resume_checkpoint: null
  checkpoint_min_coverage: 0.95
  fusion:
    mode: anew_block
    stage: source_frozen
    freeze_loaded_keys: true

training:
  lr: 1.0e-4
  warmup: 500
  max_epoch: 100
  patience: 10
  save_topk: 3
  bf16_autocast: false
```

## 11. Additional acceptance gates

Phase 8/9 acceptance gate status: **PASSED**. The completed evidence is retained below:

* 2000-atom timing uses at least 20 stable measured steps and includes raw
  step-level results.
* Dense EPT performance attribution is recorded before any approximation is
  introduced.
* Anew/PVB checkpoint coverage and source provenance are reported.
* The optimizer parameter set is exactly the non-source-loaded parameter set.
* Frozen parameters have no gradients and are bitwise unchanged after
  training; trainable parameters have finite gradients.
* Projector and gate receive gradients after the initial gate-zero step.
* Protein-only metadata has complete residues, no ligand blocks, no cross-
  sample block mixing, and valid bonds.
* The fixed-batch source-frozen smoke test decreases loss while the source
  checkpoint tensors remain bitwise unchanged.
* PVB original valid/test and paired protein-only valid/test reports are
  complete, reproducible, and not batch-truncated.

## 12. Phase 10 — PVB-posterior-preserving H-block conditioning

Phase 9 showed improved reconstruction but KL-dominated total-loss degradation.
On paired protein-only valid, PVB `off` was
`loss=0.273862, KL=0.006958, rec_vel=0.102848, rec_drf=0.165448`;
legacy fused H-block was
`loss=1.064952, KL=1.093584, rec_vel=0.064429, rec_drf=0.125655`.
The likely cause is that the legacy path uses Anew block `Wx_log_var` as
an atom-level posterior variance under PVB's different coordinate prior. Phase
9 source-frozen training did not update the Anew encoder or variance head.

### Phase 10 T1001 audit result

The read-only T1001 audit reproduced the Phase 9 loss contract
`loss = 0.8 * KL + rec_vel + rec_drf` with maximum aggregate formula error
`1.24e-8`. On the paired valid batch mean, legacy H-block KL was
`157.18x` the PVB-off KL while `rec_total` decreased by `0.078212`;
the resulting total-loss difference was `+0.791090`. The atom-weighted KL
ratio was `157.97x`.

The posterior provenance audit confirmed that PVB `off` obtains
`x_rep`/KL from `W_vec_log_var(h)`, while legacy `anew_block` obtains them
from Anew block `log_var_block` through `Wx_log_var`, broadcast to atoms.
The Phase 9 source-frozen manifest shows both Anew variance tensors loaded
and frozen, while all PVB encoder/posterior keys were absent from the
legacy PVB role and remained in the inactive complement. This agrees with
D045 and authorizes the corrected posterior-preserving architecture work.

### Phase 10 T1002 result

The new `pvb_full` role was implemented without changing the existing
`pvb` role. It expects the explicit union of
`encoder.*`, `W_vec_mu.*`, `W_vec_log_var.*`,
`decoder.*`, `vel_ffn.*`, and `drf_ffn.*` keys and
requires 100% coverage independently of the configured minimum threshold.
The real Phase 9 PVB state dict loaded `195/195` keys with zero missing,
unexpected, or shape-mismatched keys; posterior coverage was
`W_vec_mu=1` plus `W_vec_log_var=2`, and encoder coverage was
`42`. The builder smoke constructed the fused model first and recorded
the complete PVB/Anew source union.

The durable report is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json`,
SHA256
`1b5210793836f7ccc69f38de41a18cab855285673279992c51af85c90b0ff976`.

### Phase 10 T1003 result

The explicit `anew_block_pvb_posterior` mode is now accepted by model
construction and configuration/training glue. The existing `off` and legacy
`anew_block` paths retain their defaults and checkpoint semantics. Focused
model/fusion and checkpoint tests passed (`11` tests total), including the
legacy namespace/checkpoint fixtures.

### Phase 10 T1004 result

The corrected path now runs the original PVB encoder/posterior for `x_rep` and
`kl_loss`, then runs Anew separately for `H_block` decoder conditioning. The
projected condition is broadcast by `atom_block_id` and passed to both decoder
cross-attention branches through the existing interface. Anew `Wx_log_var`,
`X_atom`, and `X_block` remain diagnostic-only. Corrected training, inference,
and realization smoke tests are finite; mutating `Wx_log_var` does not change
the corrected loss and its gradients remain absent. Fusion-stage helpers now
recognize the corrected mode. T1005 is the sole active task.

### Phase 10 T1005 result

Complete gate-zero tests use identical PVB weights, inputs, and Torch/NumPy
seeds. Corrected `anew_block_pvb_posterior` matches `off` within `1e-6` for
PVB `x_rep`, KL, velocity/drift/total training loss, inference trajectory,
decoder input/output, and the zero condition. The legacy `anew_block` loader
continues to load `150/150` PVB and `68/68` Anew keys. The parity gate passed.

### Phase 10 T1006 result

The corrected source-freezing audit loads `pvb_full=195/195` and Anew
`68/68`, verifies bitwise equality for all `263` source state tensors, and
freezes exactly the `262` source parameter tensors. The exact complement is
only five projector/gate tensors (`33,281` parameters), and the optimizer
contains exactly that set. No PVB encoder/posterior tensor is random or in the
trainable complement. Artifact:
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/`
`source_frozen_provenance.json`, SHA256
`c99aa39be4c39aa8592e79630ed4b9c33e913e8b4c427b1797c321535badf6ce`.
T1007 is the sole active task.

### Phase 10 architecture contract

Preserve the existing modes and add an explicit corrected mode:

```text
off                         # original PVB behavior
anew_block                  # Phase 9 legacy semantics; unchanged
anew_block_pvb_posterior    # corrected Phase 10 semantics
```

For `anew_block_pvb_posterior`, the original PVB encoder/posterior heads
produce `x_rep` and `kl_loss` exactly as `off`. Anew runs
separately to produce `H_block`, which is projected, broadcast using
`atom_block_id`, and applied through the zero-initialized decoder gate.
Anew `X_atom`, `X_block`, and `log_var_block` are
diagnostic-only. Anew `Wx_log_var` must not enter PVB KL or the Phase 10
loss graph. The coordinate contract remains `x_mu=x0`.

Add a separately named `pvb_full` role loading compatible PVB
`encoder.*`, posterior heads, `decoder.*`, `vel_ffn.*`,
and `drf_ffn.*` keys. Keep the existing `pvb` role and legacy
`anew_block` checkpoint semantics unchanged. Gate-zero parity covers the
complete stochastic training objective and inference path, not only the decoder.

### Phase 10 protocol and gates

* First audit the Phase 9 loss decomposition, posterior provenance, and frozen
  variance tensors; do not edit architecture until the KL diagnosis agrees.
* Use the current shape-matched Anew `128`-hidden, two-layer state to
  isolate the posterior correction. Official `512`/six-layer alignment is
  deferred to Phase 11.
* Freeze the exact union of matched source keys and optimize its exact
  complement. Report every trainable/effective gradient-bearing tensor.
* Reuse exact materialized protein-only PDBBind views and padded-cost batching.
  Train at most five epochs, at least three before early stop, patience two,
  projector/gate LR `1e-3`, gradient clip `1.0`, and select by
  valid `rec_total=rec_vel+rec_drf`. Test runs once after locking.
* Log PVB/Anew log-variance distributions, KL, reconstructions, gate/condition
  norms, and module gradient norms. Never overwrite Phase 9 artifacts.

Phase 10 closes only if `pvb_full` coverage is complete; corrected
gate-zero parity matches `off` for posterior, KL, both reconstruction
terms, total loss, source sample, and decoder output; corrected KL is PVB-like;
Anew variance is absent from the loss graph; source tensors are bitwise
unchanged; optimizer membership is exact; fixed-batch reconstruction decreases;
legacy Phase 9 mode/checkpoint remains reproducible; and paired valid/test
traversal is complete, fixed-seed, and non-destructive. Any performance claim
must be supported by paired metrics. Official-checkpoint resizing, EPT
unfreezing, coordinate residuals, new block KL, block-sparse attention, test
tuning, and source-repo changes are out of scope.

Phase 10 tasks are T1000–T1014. Initially T1000 is the only active task; after
document initialization is verified, T1000 is DONE, T1001's audit is passed,
T1002's coverage gate, T1003 mode-plumbing gate, T1004 routing gate,
T1005 complete parity gate, and T1006 source-freezing gate are passed.

### Phase 10 T1007 result

T1007 diagnostics are complete. The reusable helper reports PVB and Anew
log-variance quantiles, PVB KL, diagnostic Anew KL, gate and conditioning
norms, reconstruction decomposition, and per-module gradients. On a fixed
corrected real batch, PVB log variance has mean `-0.790057` and PVB KL is
`0.006921472`; diagnostic-only Anew block log variance has mean `-2.074082`
and diagnostic KL `1.018065`. The corrected loss is `0.314263` with
`rec_total=0.308613`; Anew `Wx_log_var` receives no gradient. Artifact SHA256:
`6b2032350ca03d3839bd31e6b1001e04b2b7741db96d75b432b537c6f83641ea`.

### Phase 10 T1008 result

T1008 passed after extending the existing protein smoke CLI with the explicit
corrected mode while preserving its legacy default. The target suite ran `36`
tests in `44.304 s` with `OK`; compileall and both entrypoint help commands
passed. Protein-only smoke returned finite results for all modes, with
inference shape `[16, 3]` and train losses `off=12.922840`,
`anew_block=4.031592`, and `anew_block_pvb_posterior=29.173777`. PVB and
AnewOmni worktrees were clean. The first corrected smoke invocation exposed
the stale two-choice CLI and was fixed in `scripts/protein_smoke.py` without
changing defaults. ### Phase 10 T1009 result

The corrected fixed-real-batch gate passed on exact materialized train record 0:
`2147` atoms, `282` blocks, `4368` bonds, and identical dataset/raw/view atom
counts. Over 20 deterministic updates, `rec_total` decreased from
`0.308612682` to `0.307434760` and the PVB KL stayed near `0.007063`.
`pvb_full=195/195` and Anew `68/68`; the exact source-frozen complement is
five adapter tensors (`33,281` parameters), with gate gradients from step 0
and projector gradients after the first update. All source checksums remained
bitwise unchanged, Anew `Wx_log_var` had no gradient, and no item was dropped
or truncated. Artifact SHA256:
`ed51fda34ea1ec81aa4ee8642bfe9051892cc41b6f31408727145aebb7acb4e2`.

### Phase 10 T1010 result

The thin Phase 10 runner reused the exact Phase 9 materialized
protein-only train/valid views and padded-cost batching. Four complete
epochs traversed all `6413/6413` train records and all `367/367` valid
records; each valid pass had `354` batches, `847978` atoms, and `239`
explicit oversized singleton batches. Valid batch-mean reconstruction totals
were `0.222021`, `0.221780`, `0.222065`, and `0.221543`, with the best
at epoch `3` and global step `24797`. The PVB posterior KL stayed at
`0.006957666` batch mean. The long-running session stopped during the next
training epoch before its validation pass; the interruption is retained in
`phase10_train_interrupted.json`, and no test result was used.

The report is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_train_interrupted.json`,
SHA256
`7cd906217a6446d9e9bdc18917f2731323f2431f77538d81fc5b6bd1e10b9dc6`.

### Phase 10 T1011 result

The best corrected checkpoint was locked using valid batch-mean
`rec_total=rec_vel+rec_drf` only. The locked epoch is `3`, global step
`24797`, and its valid `rec_total` is `0.2215431160951233`. The
checkpoint SHA256 is
`5ad3b769b602037da2dd47889592a53ed6ff4f66cf4fde162aeae3a180e4d204`;
the lock manifest SHA256 is
`b9d574ad81c637e000610142f9d4a4150b4451157b8f00dcbf451e6b81444e09`.
The lock records complete `pvb_full` and Anew coverage, equal source
checksums before/after, and `test_evaluated=false`.

### Phase 10 T1012 result

The one-time paired evaluator completed all three models on the same exact
materialized protein-only PDBBind views with seeds 20260810, 20260811, and
20260812. Valid traversed 367/367 records in 354 batches and 847978 atoms,
with 239 explicit oversized singleton batches. Test traversed 167/167
records in 155 batches and 334142 atoms, with 70 oversized singleton batches.
All three models had identical counts on each split.

The first evaluator emitted all six aggregate JSON lines and then failed while
serializing tensor-valued resume metadata. The report was recovered directly
from that log without rerunning valid or test. The serializer was fixed to
retain only JSON-safe checkpoint metadata for future runs. The recovered report
is /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_paired_valid_test.json,
SHA256 9653534757efc8323db1538fcdd1993c9c36f16c95a58559b912ee1252006c52.
The completed evaluation log SHA256 is
2ee28bd2638631304c53ebb57caf6cc57a20e41d2d074f7f2a82aba1ad316af5.

Paired batch-mean metrics are listed as loss / KL / rec_vel / rec_drf /
rec_total:

| Split | PVB off | Phase 9 legacy | Phase 10 corrected |
| --- | --- | --- | --- |
| valid | 0.273862 / 0.006958 / 0.102848 / 0.165448 / 0.268296 | 1.064952 / 1.093584 / 0.064430 / 0.125655 / 0.190084 | 0.264250 / 0.006958 / 0.099335 / 0.159349 / 0.258684 |
| test | 0.255278 / 0.007027 / 0.098625 / 0.151031 / 0.249657 | 1.043928 / 1.089811 / 0.060861 / 0.111218 / 0.172079 | 0.246216 / 0.007027 / 0.095238 / 0.145356 / 0.240595 |

Atom-weighted values are retained in the JSON report. Corrected KL matches
PVB off, while corrected reconstruction and total loss improve over PVB off
on both paired splits. Legacy reconstruction is lower but its Anew-derived
KL remains about two orders of magnitude larger, so legacy total loss remains
worse. The recovered report contains aggregate mean/std and fixed-seed
provenance; per-seed detail was not reconstructed after the writer failure.

### Phase 10 T1013 result

The final audit passed. The target suite completed 36 tests with OK;
compileall and both training/inference CLI help commands passed. PVB and
AnewOmni worktrees are clean, the protected Phase 9 checkpoint and report
hashes are unchanged, all Phase 10 checkpoint/report coverage is complete,
and the runtime scan found no sibling-repository dependency in core runtime
paths. The durable audit is
/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_final_audit.json,
SHA256
e48d25108041f0948d7cbb94acd5dada53694d3b0d3da46269aeb2541fdfd050.
The Phase 9 legacy checkpoint remains reproducible. The current
shape-matched Anew 128-hidden, two-layer limitation and the aggregate-only
paired-report recovery limitation remain explicit.

### Phase 10 T1014 result

Phase 10 is closed. The four live documents now record all task evidence,
exact artifact hashes, the paired valid/test tables, source cleanliness, and
the aggregate-only recovery limitation. Gate P10 PASSED: the corrected
PVB-posterior contract is complete, gate-zero parity covers the full
stochastic objective, Anew variance is absent from the loss graph, source
checksums and optimizer membership are exact, the fixed-batch gate passes,
the valid-selected checkpoint is locked, the paired evaluation traversed all
three models on identical views, the legacy Phase 9 mode remains loadable,
and Phase 9 artifacts were not overwritten.

The paired aggregate result supports a bounded performance claim: the
corrected mode improves rec_total and both reconstruction terms over PVB off
on valid and test while retaining PVB-like KL. The Phase 9 legacy mode remains
KL-dominated. Individual per-seed metric rows are unavailable because the
first evaluator failed only during final JSON serialization after all six
traversals; the report was recovered without rerunning test.

### Proposed Phase 11 — exact official Anew representation alignment

This is a proposal only. Do not implement it as part of Phase 10.

1. Construct a separately named official-Anew mode using the exact official
   512-hidden, 64-head, six-layer, eight-radial-setting architecture and
   checkpoint semantics.
2. Require full shape/key coverage for the official encoder; reject partial
   or shape-mismatched loads rather than silently adapting them.
3. Preserve the Phase 10 PVB posterior and zero-gated H-block conditioning
   contract first; keep Anew coordinates and variance diagnostic-only until
   a separate experiment is specified.
4. Re-run vendor/source parity, SE(3), batch-isolation, complete gate-zero
   objective parity, source-freezing, fixed-batch overfit, valid-only
   checkpoint selection, and one-time paired valid/test evaluation.
5. Compare exact-official alignment against the locked Phase 10 result using
   the same views, batching, and seeds. Unfreeze Anew only after the frozen
   adapter baseline is reproducible.

Phase 10 tasks T1000–T1014 are DONE; no task is IN_PROGRESS. Post-close `git diff --check` and 36-test target suite passed; the post-close unit log SHA256 is `7818b71442c27fd36433b1014dd94b94ed5fa8273c59378cefd1673fc43aa7af`.

## 12. Phase 11 — lightweight shared-PVB H-block fusion

Phase 10 is a completed, preserved experiment. Phase 11 is a new architectural
ablation and must not overwrite Phase 9 or Phase 10 code semantics,
checkpoints, reports, or artifact hashes.

### 12.1 Objective

Replace the second Anew EPT encoder in the active fusion path with a lightweight
H-block branch that reuses the scalar atom representation already produced by
the pretrained PVB encoder:

```text
PVB atom/block embeddings
    -> pretrained PVB TorchMD encoder
    -> h_atom
       |-> unchanged PVB posterior, x_rep, and KL
       `-> block pooling -> lightweight adapter -> zero-gated decoder condition
```

For small molecules and ligands, enrich the block branch with Anew's principal-
subgraph vocabulary and, only if exact checkpoint provenance is demonstrated,
the official pretrained Anew block-embedding table. Do not run Anew EPT in
this mode.

This model must be described as `pvb_shared_hblock`. It is not the original
Anew EPT H-block representation.

### 12.2 Repository, branch, and artifact policy

| Item | Required location |
| --- | --- |
| Read-only PVB source | `/workspace/PVB` |
| Read-only AnewOmni source | `/workspace/AnewOmni` |
| Writable repository | `/workspace/fuse_anew_pvb_hblock` |
| Phase 11 branch | `phase11/pvb-shared-hblock` |
| Phase 11 artifacts | `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/` |

Phase 11 belongs in the same repository and the same four live documents as
Phases 0-10 because it is a directly comparable fusion ablation using the same
PVB model, datasets, checkpoint loaders, tests, and evaluation protocol. It
must use a separate branch and output directory.

Phase 11 should begin from the final Phase 10 commit. Prefer merging Phase 10
into `main` first; otherwise create Phase 11 as a stacked branch from the
Phase 10 branch and record the parent SHA.

### 12.3 Backward-compatible modes

Preserve every existing mode and its checkpoint semantics. Add exactly one new
mode:

```text
off                         # original PVB
anew_block                  # Phase 9 legacy dual-encoder semantics
anew_block_pvb_posterior    # Phase 10 posterior-preserving dual encoder
pvb_shared_hblock           # Phase 11 lightweight shared encoder
```

Do not silently redirect an older mode to the Phase 11 implementation.

### 12.4 Posterior contract

The original pretrained PVB posterior is authoritative:

```python
h_atom, vec = pvb_encoder(...)
x_log_var = -abs(pvb_W_vec_log_var(h_atom))
x_rep = x0 + exp(x_log_var / 2) * noise
kl_loss = pvb_kl(x_log_var)
```

The H-block path branches from `h_atom` after the PVB encoder. It must not
replace or perturb `h_atom`, `x_log_var`, `x_rep`, or `kl_loss`. During the
source-frozen Phase 11 experiment, the shared feature should be explicitly
detached before the new adapter unless a later separately named fine-tuning
stage is approved.

### 12.5 Lightweight H-block branch

Default pooling reuses Anew's variance-preserving convention:

```python
h_block = scatter_sum(h_atom.detach(), atom_block_id, dim=0)
h_block = h_block / sqrt(block_lengths).unsqueeze(-1)
```

Retain ordinary `scatter_mean` only as a named ablation.

Use a bottleneck adapter rather than a full encoder:

```text
LayerNorm(pvb_hidden)
-> Linear(pvb_hidden, adapter_rank)
-> SiLU
-> Linear(adapter_rank, pvb_hidden)
```

Start with `adapter_rank=32`. Initialize the scalar `shared_hblock_gate` to
zero so that the complete branch contributes exactly zero initially. Keep the
final adapter projection at ordinary initialization; zeroing both the final
projection and gate would also zero the gate's first-step gradient.

Broadcast the block condition to atoms with explicit integer metadata. Inject
it once, after the PVB decoder's `x0`/`xt` cross-branch merge and before its
equivariant attention stack:

```python
decoder_hidden = decoder_hidden + tanh(shared_hblock_gate) * condition_atom
```

Do not add the same condition independently to both pre-merge branches.

### 12.6 Separate structural and semantic block identities

Do not overload legacy PVB `btype`. Preserve it for checkpoint-compatible PVB
atom/block input embeddings.

The Phase 11 data contract is:

```text
btype                 [N_atom]  legacy PVB block type
atom_block_id         [N_atom]  atom -> semantic/pooling block
semantic_block_type   [N_block] Anew residue/fragment vocabulary ID
block_lengths         [N_block]
block_batch           [N_block]
```

Block definition:

| Input | Phase 11 pooling block |
| --- | --- |
| Standard protein | complete residue |
| Small molecule | Anew principal-subgraph fragment |
| Protein-ligand complex | protein residue plus ligand fragment |

The current protein block metadata may be reused. Molecular fragment
membership must be produced offline from the Anew tokenizer and raw chemical
identity/bond-order information. Do not infer Anew fragments from coordinates
or from a binary PVB `bond_index` that has lost bond order.

### 12.7 Anew semantic embedding

The semantic branch is optional in Phase 11A and required in Phase 11B.

Preferred Phase 11B implementation:

```text
exact Anew vocabulary ID
-> official frozen Anew block_embedding.weight, if exact key/shape provenance passes
-> low-rank projection to PVB hidden dimension
-> semantic_gate initialized to zero
-> combine with pooled PVB h_block
```

Do not instantiate Anew's redundant atom embedding because PVB already has a
pretrained atom embedding. Do not call randomly initialized fragment vectors
"pretrained Anew representation". If the official table cannot be extracted
and matched exactly, retain a separately labeled `random_vocab_embedding`
ablation or stop the pretrained-semantic experiment.

### 12.8 Staged execution

#### Phase 11A — shared residue H-block

Use the existing exact-length protein-only PDBBind view. Train only the new
pooling adapter and gates. Freeze the complete PVB source checkpoint and do not
run Anew EPT.

Required comparisons:

```text
PVB off
Phase 10 posterior-preserving dual encoder
Phase 11A shared PVB H-block
```

Phase 11B is intentionally deferred. The current Anew fragment tokenizer does
not return complete semantic block metadata for the affected population, so no
fragment branch, training, or evaluation is allowed until a separate repair
decision is made. Phase 11A remains the only accepted Phase 11 result.

#### Phase 11B — Anew fragment-vocabulary semantics

Vendor only the minimal tokenizer/vocabulary files from the pinned Anew source,
including the exact vocabulary asset and license. Run tokenization during CPU
preprocessing, never inside GPU forward.

Create a separately materialized complex/molecular view with source IDs,
fragment memberships, semantic IDs, bond orders, fallback reasons, and exact
atom counts. Stop if raw chemical identity cannot be reconstructed faithfully.

### 12.9 Training and selection

For both stages:

* preserve Phase 10's PVB posterior and checkpoint coverage;
* freeze all source-loaded PVB/Anew tensors;
* train only new adapter/projection/gate parameters;
* use exact optimizer-membership and checksum audits;
* use fixed seeds `20260810`, `20260811`, and `20260812` for formal evaluation;
* tune on train/valid only and run test once after checkpoint lock;
* select by valid `rec_total = rec_vel + rec_drf`, while reporting KL and total
  loss separately;
* use early stopping rather than assuming more epochs solve an architectural
  mismatch.

Record parameter counts and step-time/memory deltas against PVB `off`, Phase 9,
and Phase 10. The intended success criterion is a small linear-cost branch,
not merely fewer trainable parameters while retaining a second frozen EPT.

### 12.10 Acceptance gates

Phase 11 correctness requires:

* gate-zero full-objective parity with PVB `off` under identical RNG state;
* identical PVB `x_rep` and KL between `off` and `pvb_shared_hblock` at gate zero;
* no Anew EPT execution in `pvb_shared_hblock`;
* explicit complete block metadata with no cross-sample mixing;
* exact source-checkpoint coverage and bitwise preservation;
* optimizer membership equal to the new-parameter set;
* finite gradients on adapter and gates;
* fixed-batch reconstruction decrease;
* protein inference smoke success;
* measured parameter, time, and memory overhead;
* legacy Phase 9 and Phase 10 modes remain loadable and reproducible;
* no Phase 9 or Phase 10 artifact is overwritten.

Phase 11B additionally requires:

* tokenizer parity with the pinned Anew source on representative molecules;
* exact vocabulary asset checksum;
* every atom assigned to exactly one block;
* explicit unknown/fallback coverage reports;
* exact provenance for any official Anew embedding tensor;
* no fragment tokenization in the model forward pass.

### 12.11 Out of scope

Do not in Phase 11:

* change the PVB posterior variance or coordinate mean;
* run or fine-tune Anew EPT in the shared mode;
* unfreeze the PVB encoder/decoder before the adapter-only result is locked;
* introduce coordinate residuals or a new block KL;
* infer fragments from incomplete bond information;
* overwrite or reinterpret older fusion modes;
* implement block-sparse attention;
* tune on test;
* modify `/workspace/PVB` or `/workspace/AnewOmni`.

### 12.12 T1101 execution record

T1101 is complete. The audit confirms that PVB `TorchMD_VQ_ET` already
produces the scalar atom state needed by Phase 11A, but `dyVAE.encode()`
currently consumes it only through `W_vec_log_var` and does not expose it.
PVB `btype` is a legacy per-atom vocabulary: IDs 0–117 are element blocks and
118–137 are the 20 protein residue blocks. New pooling metadata remains
separate, and molecular/element IDs are not mapped into Anew fragment IDs.

The Anew EPT path constructs dense atom-level attention tensors with an
approximately quadratic `max_atom_count` cost. On the exact Phase 9
protein-only train record 0 (2147 atoms, 282 blocks, 4368 bonds), the one-step
forward-only diagnostic was 149.38 ms for PVB `off`, 102.72 ms for Phase 9
legacy Anew, and 282.05 ms for Phase 10 corrected PVB-posterior plus Anew EPT.
The corrected path therefore measures the cost of two encoders, not the
Phase 11A target. The Phase 11A cost boundary is: reuse PVB `h_atom`, perform
sum/sqrt(block-length) pooling and the rank-32 adapter, and make no Anew EPT
call. The full machine-readable audit is recorded at
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/profiles/t1101_audit.json`.


### 12.13 T1102 execution record

T1102 is complete. The target now accepts the distinct `pvb_shared_hblock`
mode without constructing Anew's encoder. PVB `encode()` retains its historical
two-value return by default and exposes an optional state dictionary containing
`h_atom`, `vec_atom`, and `log_var_pvb` for the next integration step. The new
mode follows the original PVB posterior path; it does not yet pool or inject a
condition. PVB full checkpoint loading covers 195/195 keys with no mismatch.
Focused shared-mode, legacy-fusion, and stage tests passed 17/17.

### 12.14 T1103 execution record

T1103 is complete. The new shared branch reuses vendored Anew's
`std_conserve_scatter_mean` for `scatter_sum(h_atom) / sqrt(block_length)` and
does not infer membership from coordinates. `SharedHBlockAdapter` implements
the rank-32 `LayerNorm -> Linear -> SiLU -> Linear` bottleneck; its final
projection uses ordinary initialization while the model-level
`shared_hblock_gate` is zero initialized, preserving a first-step gate
gradient while keeping the complete branch a no-op at initialization.
The adapter receives detached PVB `h_atom`, broadcasts the block condition with
explicit `atom_block_id`, and is classified as the projector/gate optimizer
group without changing legacy mode names. At hidden size 256 the branch has
7 parameter tensors and 17,185 parameters. The focused shared/stage suite
passed 14/14 and compilation passed. Decoder injection remains T1104.

### 12.15 T1104 execution record

T1104 is complete. The decoder now accepts a separately named
`post_cross_condition` and adds it exactly once after the `x0`/`xt` cross-branch
merge and before the equivariant attention stack. The existing
`block_condition` path used by Phase 9/10 modes is unchanged. Training,
`inference()`, and `realization()` pass the new condition only for
`pvb_shared_hblock`; Equiformer rejects either conditioning path explicitly.

The focused Phase 11 suite passed 10/10, including full gate-zero objective and
inference parity, one-time post-merge injection, finite gradients after the
gate moves off zero, and SE(3) behavior. Existing fusion and staged-training
regression tests passed 14/14; compilation and `git diff --check` passed.

### 12.16 T1105 execution record

T1105 is complete. The existing `pvb_full` role loads the complete PVB
encoder, posterior heads, decoder, and output heads into a freshly constructed
`pvb_shared_hblock` model with `195/195` coverage and no missing, unexpected,
or shape-mismatched keys. The new mode constructs no Anew encoder.

The real provenance audit is recorded at
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/`
`t1105_shared_provenance.json` (SHA256
`a7209f4e51176e2daa90e55ebeb3404ebf0868b587209a3ba95b9044b0a6269a`). It
checks 195 source parameter tensors / 9,965,317 parameters bitwise, freezes
the exact source union, and exposes only the 7 shared adapter/gate tensors
(`17,185` parameters) in one `projector_gate` optimizer group. The focused
checkpoint/stage suite passed 17/17 and compilation passed.

### 12.17 T1106 execution record

T1106 is complete. The focused Phase 11 suite passed 12/12. With identical
weights, evaluation mode, inputs, and Torch/NumPy seeds, gate-zero
`pvb_shared_hblock` matched PVB `off` for `h_atom`, `vec_atom`,
`log_var_pvb`, `x_rep`, KL, both reconstruction terms, total loss, inference
source sampling, and decoder output. The test also loaded the protected Phase
9 legacy `source_frozen_epoch1_best.ckpt`/legacy Anew state and the protected
Phase 10 corrected checkpoint with complete required coverage. No parity or
compatibility failure was observed, so the fixed-batch gate may proceed.
### 12.18 T1107 execution record

T1107 is complete. The exact materialized train record 0 contains 2147 atoms,
282 blocks, and 4368 bonds with identical dataset/raw/view counts. The shared
mode constructs no Anew encoder and uses 17,185 trainable adapter/gate
parameters. Its forward-only profile was 79.029 ms / 1.813 GiB and its
source-frozen step was 117.069 ms / 9.527 GiB; the Phase 10 corrected path was
85.729 ms / 1.809 GiB and 138.208 ms / 9.531 GiB on the same A100/GPU slot.
The 20-step fixed-batch gate passed with finite gradients, exact optimizer and
source checksums, gate gradient from step 0, projector gradient after step 1,
and `rec_total` decreasing from 0.308612704 to 0.308233753. The profile and
overfit artifact hashes are recorded in `TASKS.md` and `HANDOFF.md`.
### 12.19 T1108 implementation preflight

T1108 is in progress. A thin Phase 11A training runner now reuses the Phase 9
exact materialized protein-only batching and metric accumulation and the Phase
10 valid-only `rec_total` selection protocol. It constructs
`pvb_shared_hblock`, requires complete `pvb_full` coverage, freezes the exact
loaded PVB union, and optimizes only the seven shared adapter/gate tensors. A
separate paired evaluator requires the locked manifest and refuses to run after
`test_evaluated` is set, so test remains post-selection only. A one-item/one-step
smoke reached validation and wrote a debug checkpoint/lock with PVB-like KL,
unchanged source checksums, and no Anew encoder. A `rec_total` aggregation bug
was found and fixed before formal training; no formal artifact or test result
has been recorded yet.
### 12.20 T1108 storage incident and restart condition

The first formal T1108 epoch completed the entire train view (6,200 steps,
14,660,149 atoms) and valid view (354 batches, 847,978 atoms) before saving the
selected checkpoint. Valid `rec_total` was `0.218641275` and KL was
`0.006957666`, but `/output` was at 100% with zero available bytes, so the
checkpoint write failed and no formal report or lock was produced. Only the
three task-owned transient/partial Phase 11 checkpoint files were removed;
Phase 9/10 artifacts were untouched. The runner now permits best/last aliasing
to keep one approximately 40 MiB checkpoint, and T1108 must restart from the
source checkpoint before any test evaluation.
### 12.21 T1108 single-checkpoint replacement repair

The formal T1108 run completed epochs 0–2 end to end, with validation
`rec_total` values `0.218641275`, `0.218670571`, and `0.218561299`; KL stayed
at approximately `0.006957666`. The epoch-2 improvement required replacing an
existing approximately 40 MiB checkpoint, but PyTorch's direct overwrite
needed old and new files simultaneously and failed at zero free bytes. The
runner now unlinks only its own previous Phase 11 checkpoint before writing the
replacement, and the run will restart from the immutable PVB source checkpoint.
### 12.22 T1108 checkpoint-size repair

The immutable PVB source state is approximately 39.94 MB, while the available
space after removing the failed Phase 11 archive is 40,161,280 bytes. The Phase
11 model-only checkpoint fits this boundary; adapter optimizer moments do not
need to be present for the locked evaluation artifact because exact optimizer
membership is already checked and recorded during training. The runner now
records `optimizer_state_saved=false`, deletes only its own prior archive before
replacement, and retains the optimizer protocol in the JSON report.
### 12.23 T1108 adapter-only checkpoint format

Two formal runs reached complete train/valid epochs 0–2 but could not write a
full fused model archive on the zero-slack `/output` volume. The Phase 11A
checkpoint contract is therefore explicit: save only the seven adapter/gate
state tensors, record the immutable external `pvb_full` path/checksums in the
report and lock, and reload `pvb_full` before applying the adapter state for
paired evaluation. A format smoke produced a 109 KiB checkpoint and verified
`pvb_full` coverage `195/195` plus adapter coverage `7/7`; no optimizer moments
are needed for the evaluation lock.

### 12.24 T1108 formal train/valid lock

The formal Phase 11A adapter-only run completed all five configured epochs and
30,984 steps over the exact protein-only views: 6,413 train items / 14,666,461
atoms and 367 valid items / 847,978 atoms. Validation batch-mean
`rec_total = rec_vel + rec_drf` was `0.2186412669`, `0.2186705688`,
`0.2185613004`, `0.2184535972`, and `0.2183195718`; epoch 4 was selected solely
by the last value. Final valid metrics were loss `0.2238857052`, KL
`0.0069576662`, `rec_vel` `0.0911244328`, and `rec_drf` `0.1271951389`.
The locked adapter-only checkpoint SHA256 is
`fecb7371033bb2dc5f82d865890f182fb41991c43104b3b302533d0f8dcab08f`; its
external immutable PVB source checkpoint SHA256 is
`4f0ad88356c7159cd5d0b9641b6c1e5c5f97a87ed95e7748c8189e1a110d1a77`, and the
lock SHA256 is `131138fa13701543d17c71ba2153ffd062ce798c8149f9ced7dbc61c95821178`.
The source checksum audit remained bitwise clean, with 195/195 PVB keys loaded,
exactly seven adapter/gate tensors trainable, no Anew encoder constructed, and
`test_evaluated=false`. The next action is the one-time three-seed paired
valid/test traversal; test must not change this locked checkpoint.

### 12.25 T1108 locked paired valid/test result and Gate P11A

The one-time paired evaluator completed all four models on identical exact
protein-only views with seeds `20260810`, `20260811`, and `20260812`: valid had
367 items / 847,978 atoms / 354 batches and test had 167 items / 334,142 atoms /
155 batches. The Phase 11 shared adapter-only result was:

| Split | Weighting | Loss | KL | rec_vel | rec_drf | rec_total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| valid | batch mean | 0.260572391 | 0.006957666 | 0.097893420 | 0.157112839 | 0.255006258 |
| valid | atom weighted | 0.251991546 | 0.006926987 | 0.095785901 | 0.150664055 | 0.246449956 |
| test | batch mean | 0.242662904 | 0.007027225 | 0.093872097 | 0.143169028 | 0.237041126 |
| test | atom weighted | 0.240526811 | 0.006980063 | 0.093844615 | 0.141098147 | 0.234942761 |

PVB `off` batch-mean `rec_total` was `0.268296109` valid / `0.249656701`
test, and Phase 10 corrected was `0.258683866` valid / `0.240594518` test.
Thus shared improved reconstruction and total loss while retaining PVB-scale KL;
legacy reconstruction is lower but its KL remains approximately `1.09` and its
total loss is not comparable as an improvement. The paired report is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/shared_hblock_protein/t1108_paired_valid_test.json`
(SHA256 `ae078b126a3919936bc3a5b99c7f5cbaa7fa85d16b29012b9d6ca84f79064d1f`).
Gate P11A passed. T1109 is now the sole active task; no semantic pretrained
branch may be enabled until its official Anew table provenance is proven.

### 12.26 T1109 official block-embedding provenance

T1109 passed without modifying the fused runtime model. A source-native probe at
Anew commit `926e99818ea18cf9d9b2064ce0319fe691b7a1f1` found exactly one official
full-model key, `base_model.autoencoder.embedding.block_embedding.weight`, and
matched it bitwise to the derived key `embedding.block_embedding.weight` and the
new extracted artifact. The table is `437 x 512`, `torch.float32`, with tensor
SHA256 `2ba7c22abf1ca550d354d282e7c4ed2278ab972789ce223bf50db31abb69ddf4`.
The captured Anew block order has `UNK` at index 0, 20 amino acids at indices
1–20, followed by the remaining pinned vocabulary; its order SHA256 is
`b7e157f2f6cb62e673301430333fa4b6988573f68237dfb8b31fe806b4f133a1`.
The extracted artifact is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/anew_official_block_embedding.pt`
(SHA256 `ad31c380a2c6ab8cca24457948bdd8979829569150044c4022de3a938246f26c`);
the provenance JSON SHA256 is
`2e2525fd578389072219d51fdf2698ab1de1d9f49e81bb14dcbd1502d55810ee`.
The `--verify-only` rerun and focused test passed. T1110 may now vendor the
minimal tokenizer/vocabulary implementation and assets, while semantic model
code remains disabled.

### 12.27 T1110 vendor parity and T1111 start

T1110 passed. Thirteen pinned Anew tokenizer/vocabulary/helper files were
vendored under the namespaced package; all files are byte-identical after the
three documented import-only adaptations. Source/target vocabulary and
tokenization parity passed on four representative SMILES, with 437 block types,
119 atom types, and the exact size-300 kekulized cycle-priority tokenizer.
The provenance report SHA256 is
`ad1ec26af7eb4282b4c31d060d3adeba5873de8ab67d51e6df29093e6e40ecc4`.
T1111 is now blocked; semantic metadata work must stop if complete bond-order
and aromaticity information is unavailable.

### 12.28 T1111 semantic-data gate is blocked

T1111 completed the required CPU-only audit without modifying the original PVB
mmap dataset or raw ligand SDF files. All 6,947 train/valid/test records were
joined to the half-PVB manifest; all 219,314 ligand atoms preserved exact PVB
versus SDF heavy-atom order. The Anew tokenizer assigned atoms exactly once
when it succeeded, but upstream mapping assertions required explicit atom-level
fallback for 508 records (15,772 atoms; 7.31% of records and 7.19% of atoms).
More importantly, five train SDF bonds have `UNSPECIFIED` bond order. Mapping
those bonds to Anew's four bond types would require guessing, so the semantic
fragment gate is blocked and T1112+ model integration must not start.

The blocked audit is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/t1111_semantic_data_audit.json`
Latest prerequisite probe (2026-08-14) confirms the blocker rather than
clearing it. The Phase 12 chemistry audit correctly used valid SDF bond
metadata directly, restricted damaged-record inference to ligand-only
coordinates, and found no usable graph for the five affected records. The
508 fallback records also remain unresolved. Therefore T1112–T1117 must not
be started or marked complete.

The completed Phase 11A shared-PVB result is documented separately in
`reports/phase11/p9_p10_p11a_architecture.md`. It compares P9 legacy, P10
PVB-posterior H-block, and P11A shared-H-block architecture and paired loss
metrics. This is a provisional Phase 11A report: it does not claim semantic
Anew fragment integration or close Gate P11 while T1111 is blocked.
The report SHA256 is `7d8ab3cefcdb3820ce0b70c192ed64063ed19cb4cc97c1a3207b4b9e324fabd8`.


## 13. Phase 12 — source chemistry recovery and PVB dataset materialization

Phase 12 contains two independent data workstreams and must not modify the
Phase 11 protein-only test inputs, Phase 9/10/11A artifacts, or the read-only
PVB/AnewOmni source repositories. The workstreams may run concurrently under
the single documented T1200 coordinator.

### 13.1 T1200 — Coordinate the parallel data workstreams

T1200 owns two independently launched workers:

- Worker A / T1201: inspect the original chemistry datasets, determine whether
  tokenizer fallback is caused by missing SDF/bond metadata, and benchmark
  `xyz2mol` before any full reconstruction.
- Worker B / T1202: acquire PDB files using the official wwPDB rsync script,
  inspect the existing `ept_release` split/materialization scripts, and produce
  full then half PVB datasets.

T1202 remains a background acquisition lane while Phase 13 is active; it must
not be used by Phase 13 and must not consume the single-`IN_PROGRESS` slot.

### 13.2 T1201 — Audit original chemistry and xyz2mol feasibility

- Inspect `/data4/PVB/pcqm4mv2`, `/data4/PVB/ani1x`, and
  `/data4/PVB/pdbbind` for coordinates, atom identity, bond order, aromaticity,
  SDF/mol/graph files, and exact atom-order contracts.
- Distinguish missing source chemistry from Anew tokenizer mapping assertions;
  the existing PDBBind audit already found SDF inputs, so `fallback` must not
  be assumed to mean absent SDF.
- Verify/install the `xyz2mol` pip package only in the active environment and
  benchmark representative records before full processing.
- Treat a complete, readable SDF as authoritative: use its bond orders and
  aromaticity directly and never run coordinate inference over it. Restrict
  reconstruction to damaged/unspecified SDF records.
- For PDBBind repair, extract only the ligand atom table and ligand conformer
  coordinates. Never concatenate or pass receptor/protein coordinates to a
  bond-assignment routine.
- Process ANI-1x separately because it has coordinates but no explicit bond
  order/aromaticity arrays. Benchmark by unique molecular group and reuse a
  validated topology across conformers rather than inferring once per frame.
- Compare RDKit rdDetermineBonds and Open Babel bond perception with
  xyz2mol. RDKit is a C++ xyz2mol implementation and is a performance
  alternative, not an independent chemistry oracle; accept no result without
  valence, charge, aromaticity, atom-order, and known-edge validation.
- Estimate full processing wall time from measured throughput and record the
  sample size, hardware, package version, I/O cost, and projected total.
- If projected processing exceeds four hours, stop without launching the full
  conversion and record the blocker. If it is within four hours, write outputs
  to a new Phase 12 directory and never overwrite raw data.
- Accept a generated record only when atom alignment, bond order, and aromatic
  bond information are explicit; never guess an Anew bond type.

### 13.3 T1202 — Acquire PDBs and materialize full/half PVB data

- Inspect the supplied `rsyncPDB.sh` and the existing scripts under
  `/data4/PVB/pdb/ept_release` before downloading or overwriting anything.
- Download PDB files into `/data4/PVB/pdb/` using the official wwPDB script;
  preserve the downloader URL, revision/hash, command, and downloaded counts.
- Use the existing `ept_release` selection/split scripts to create the full
  PVB dataset under `/data4/PVB/pdb/ept_release`.
- Keep the EPT `processed/PDB` mmap as the selection/index intermediate; its
  `X/B/A/atom_positions/block_lengths/segment_ids` schema is not directly the
  PVB `atype/btype/x0/b0/bond_index` schema.
- Use the EPT chain index to construct structure-level train/valid/test manifests,
  then invoke the original PVB `data/pdb_dataset.py` path with an explicit
  NumPy compatibility shim, producing fresh PVB `*_block` directories.
- Create the requested half dataset under `/data4/users/sihao/data` (or a
  clearly named `pvb_cross_dataset_20260810` subdirectory if that is the
  existing convention), preserving split identity and provenance.
- Validate item counts, PDB identifiers, atom/block/edge metadata, split
  disjointness, and hashes. Do not use the processed PDB data in Phase 11 tests.
- Do not delete or silently replace existing PDB, full-dataset, or half-dataset
  material; inspect first and use new output names when necessary.

### 13.4 Measured T1201 result

The source audit is recorded in `reports/phase12/t1201_chemistry_xyz2mol_audit.md`.
PCQM4Mv2 contains SDF chemistry; ANI-1x contains 3,114 molecular groups and
4,956,005 conformers but no explicit bond-order/aromaticity arrays; PDBBind
contains 15,487 SDF records, including the five records with unspecified
bonds from the Phase 11 audit.

The original PDBBind timing is retained in the audit as superseded evidence
because its input boundary did not prove ligand-only coordinates. The corrected
probe passed only SDF ligand conformer coordinates: the five damaged records
(14–22 atoms) took 0.061–1.594 seconds and all returned zero usable molecules;
a separate 20-record ligand-only diagnostic had 5 returns and 15 three-second
timeouts. Slow behavior therefore can occur without protein coordinates, and
the result cannot justify processing valid SDF records. Complete readable SDF
chemistry is used directly. ANI remains separate: 3,114 unique groups and
4,956,005 conformers, with topology reuse; RDKit and Open Babel are comparison
candidates, not authoritative labels.

Follow-up probe result: all 508 tokenizer fallbacks are PDBBind records
(train/valid/test `480/15/13`), with complete SDF bond types and exact
PVB/SDF atom order. Their failure is the Anew `get_submol_atom_map()`
assertion, not missing chemistry. RDKit produced 68 tokenizer passes but zero
raw-graph-preserving candidates; Open Babel produced 50 passes, only one exact
graph (`5c28_holo`) whose tokenizer still failed. No geometry rewrite is
accepted for this population. The five damaged records have isolated
`0 -> SINGLE` candidates only; all fail normal RDKit boron-valence validation.

### 13.5 T1202 live acquisition measurement

The official dry run identified 242,490 regular files, 1,266 directories, and
53,593,441,967 bytes. The initial single connection was too slow; the current
36-prefix resumable pool omits transport compression for already-compressed
`.ent.gz` files. A one-minute sample grew from 476,201,351 to 647,679,590
bytes and from 2,777 to 3,724 files, an early residual projection of about
5.2 hours. The mirror remains asynchronous; materialization starts only after
the source tree is complete.

Live refresh (2026-08-14): the resumable 36-prefix pool was restarted after
an interruption and has reached 24,019 regular files and approximately
5.02 GB under /data4/PVB/pdb; materialization remains blocked until completion.
### Phase 12 acceptance gates

- Complete SDF records bypass reconstruction. Damaged-SDF repair is either
  stopped above four hours or passes exact ligand-only atom/bond/aromaticity
  validation; ANI is evaluated separately by unique molecular group.
- Full and half PVB datasets have reproducible split manifests, counts, source
  provenance, and no overlap or silent truncation.
- Phase 11 continues to use its existing protein-only materialized views.
- Original raw datasets, `/workspace/PVB`, `/workspace/AnewOmni`, and protected
  Phase 9/10/11A artifacts remain unchanged.

## Phase 13 — H-block information, capacity, and injection ablations

### Objective

Determine why the deterministic shared-PVB H-block path in Phase 11A gives a
small but consistent reconstruction improvement over Phase 10, while avoiding
the posterior/KL confound present in Phase 9. Phase 13 must distinguish five
explanations: record-specific H-block information, trainable adapter capacity,
atom-to-block pooling, feature-source alignment, and conditioning injection
location. It does not introduce a Gaussian block latent.

The registered observations are:

- Phase 9 improves reconstruction but replaces the PVB posterior variance and
  therefore cannot establish that a Gaussian H-block is beneficial.
- Phase 11A uses 17,185 trainable parameters versus Phase 10's 33,281; its
  improvement cannot be attributed to a larger raw parameter count.
- Phase 11A improves batch-mean `rec_total` over Phase 10 by 0.003677608 on
  valid and 0.003553392 on test. This is consistent but small and requires
  paired controls before an architectural conclusion.

### Registered hypotheses and matched variants

All corrected variants preserve the complete PVB posterior for `x_rep` and KL.
PVB and Anew source-loaded parameters remain frozen and checksummed; only the
declared projector/adapter and gate may be optimized.

1. `off`: unchanged PVB reference.
2. `pvb_shared_real`: current Phase 11A shared-PVB H-block path.
3. `pvb_shared_shuffled`: shuffle pooled H-block values only within each
   sample, deterministically by record and seed. Preserve block counts, tensor
   shapes, feature norms, adapter/gate parameter count, and batch isolation.
4. `pvb_shared_constant`: feed a zero or fixed non-record-specific block input
   through the same trainable adapter/gate. This is the matched capacity and
   injection control.
5. `pvb_atom_no_pool`: condition with the corresponding detached PVB atom
   features through the same adapter family, without atom-to-block pooling.
6. `anew_rank32_postmerge`: feed the frozen shape-matched Anew H-block through
   the same rank-32 adapter and post-merge injection used by Phase 11A.
7. A matched injection-location pair: hold feature source, pooling, adapter
   rank, initialization, and optimizer protocol fixed while comparing the
   Phase 11A post-merge injection with the Phase 10 cross-attention injection.

Where exact parameter equality is structurally impossible, report the exact
difference and add a frozen/dummy-parameter control; do not describe an
unmatched comparison as a capacity ablation.

### Execution stages

Stage A is a cheap dependency diagnostic using the locked Phase 11A checkpoint:
evaluate real, sample-local shuffled, and constant/zero inputs without
retraining. This tests whether the locked adapter currently depends on
record-specific H-block content, but it is an out-of-distribution perturbation
and is not by itself a performance comparison.

Stage B retrains the matched controls using identical protein-only train/valid
views, exact padded-cost batching, initialization seeds, learning rate,
maximum epochs, early-stopping rule, and gradient clipping. Use valid
`rec_total = rec_vel + rec_drf` to select and lock each checkpoint. Test is
evaluated only once after all checkpoints and hypotheses are locked.

Stage C performs paired complete valid/test evaluation with seeds `20260810`,
`20260811`, and `20260812`. Report batch-mean and atom-weighted loss, KL,
`rec_vel`, `rec_drf`, and `rec_total`; also report per-record paired deltas
and bootstrap confidence intervals because the observed Phase 10-to-11A effect
is small. Record gate value, condition norm, gradient norms, time, and peak
memory.

### Interpretation contract

- `real > shuffled/constant` supports record-specific H-block information.
- `real ~= shuffled/constant > off` supports adapter/injection capacity rather
  than usable block content.
- Matched PVB features outperforming matched Anew features supports feature
  alignment with the frozen PVB decoder.
- Pooled block features outperforming the atom no-pool control supports the
  block aggregation itself.
- An injection-location difference under otherwise matched conditions supports
  placement, not feature-source or capacity, as the cause.

Do not infer that H-block should be Gaussian from Phase 9. A stochastic block
latent is eligible only for a separately planned later phase if deterministic
record-specific content is first established. It must then use its own latent,
zero-gated decoder contribution, KL warmup/free bits, and must never replace
the PVB posterior or its log variance.

### Artifacts and acceptance gate

Write new artifacts only under:

```text
/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13
```

Gate P13 requires complete PVB-posterior parity at gate zero, exact source
checksum preservation, exact optimizer membership, batch-isolated deterministic
shuffle controls, declared parameter-count matching, finite fixed-batch
gradients, valid-only checkpoint selection, and paired complete valid/test
traversal. Phase 9/10/11/12 artifacts must not be overwritten. Phase 13 is now
active on the existing protein-only views; the PDB acquisition remains
asynchronous and is not an input to Phase 13.

### Current execution state — 2026-08-14

- T1300-T1310 are DONE for the four-control PVB-matched tranche: the protocol,
  registry, matched controls, parity, source-freezing, optimizer, capacity,
  fixed-real-batch, Stage-A diagnostic, valid-only training, one-time paired
  evaluation, and final audit all passed.
- No Phase 13 task is currently `IN_PROGRESS`. T1304/T1305 feature-source and
  injection-location extensions remain explicit TODOs for a separately
  registered paired experiment; Phase 11B remains deferred.

### Phase 13 Stage-B training result

All four controls used the same 6,413-record train view, 367-record valid view,
14,666,461/847,978 atom totals, optimizer membership, source checksums, seed,
learning rate, clipping, batch budget, and valid `rec_total` selection. The
selected valid `rec_total` values are:

- `pvb_shared_real`: `0.218319582`, epoch 4 of 5.
- `pvb_shared_shuffled`: `0.218891819`, epoch 0; stopped after patience.
- `pvb_shared_constant`: `0.218855551`, epoch 0; stopped after patience.
- `pvb_atom_no_pool`: `0.216537594`, epoch 4 of 5.

Each run has exactly 17,185 trainable adapter/gate parameters and no test
access. These were valid-only selection results; the one-time paired evaluator
has now completed on identical valid/test views. It produced aggregate
batch-mean and atom-weighted metrics, but no per-record vectors, so no
post-hoc bootstrap claim is made.

### Phase 13 paired result

The one-time paired evaluation found valid/test batch-mean `rec_total` of
`0.268296111/0.249656701` for PVB off, `0.255006258/0.237041124` for real,
`0.255652213/0.237688745` for shuffled, `0.255703018/0.237712422` for
constant, and `0.252977388/0.235095688` for atom-no-pool. All controls retained
PVB KL within numerical noise. Real's approximately `0.00065–0.00070`
advantage over shuffled/constant is a small record-specific signal; the much
larger common gain over off is compatible with adapter/injection capacity.
Atom-no-pool being best suggests the current block pooling path may discard
useful atom-level information. These results do not imply that H-block should
be Gaussian.
