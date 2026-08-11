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
