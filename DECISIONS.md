# Decisions

## D001 — Base the target on PVB

Status: Accepted

The output starts from the local PVB repository because PVB owns the bridge,
decoder, losses, trainer, inference, and evaluation behavior.

Do not rebuild a new repository around AnewOmni.

## D002 — Treat both source repositories as read-only

Status: Accepted

All modifications go to `/workspace/fuse_anew_pvb_hblock`.

## D003 — Vendor minimal Anew code

Status: Accepted

Copy the required Anew files into `third_party/anewomni` and preserve their
license headers. The final repository must not depend on sibling-repository
imports or `sys.path` manipulation.

## D004 — Reuse implementations before writing new ones

Status: Accepted

EPT, radial bases, BlockEmbedding, graph utilities, stable normalization, and
variance-preserving pooling must come from AnewOmni. PVB bridge, decoder,
trainers, and graph behavior must come from PVB.

New implementation is reserved for integration glue and tests.

## D005 — Use Anew’s actual block representation

Status: Accepted

The useful representation is produced after atom-to-block pooling. Returning
only `H_atom` or `X_atom` is not considered block fusion.

## D006 — Preserve PVB coordinate mean initially

Status: Accepted

Milestone one uses `x_mu = x0`. Anew’s coordinate output must not replace
PVB’s bridge source at initialization.

Reason: the original PVB contract fixes the coordinate mean at the input
structure, while a randomly initialized Anew coordinate mean changes the
source distribution immediately.

## D007 — Condition PVB through a zero gate

Status: Accepted

`H_block` is projected, broadcast to atoms, and injected through a
zero-initialized scalar gate.

Gate zero must preserve PVB decoder behavior.

## D008 — Protein-only first milestone

Status: Accepted

PVB amino-acid ordering can be mapped to Anew amino-acid ordering after an
explicit assertion.

PVB element blocks must not be mapped to Anew fragment IDs by an arbitrary
numeric offset. Ligand support requires a separate vocabulary/tokenization
design.

## D009 — Explicit block IDs

Status: Accepted

Block membership is generated in preprocessing or CPU collation. It must not
be recovered from floating-point equality of repeated block-center
coordinates inside the model.

## D010 — Preserve complete blocks during cropping

Status: Accepted

Atom-radius cropping may create partial residues. The block-aware model
requires complete blocks, so cropping must expand or reject partial blocks.

## D011 — Separate checkpoint roles

Status: Accepted

Use separate PVB initialization, Anew initialization, and fused-resume
checkpoints. Do not load whole serialized upstream models as if they were
fused checkpoints.

## D012 — Correctness before optimization

Status: Accepted

First implement the faithful Anew H-block path. Only after parity, overfit,
and SE(3) tests pass may performance changes be introduced.

## D013 — A sparse block-first encoder is a separate model

Status: Accepted

If dense EPT remains too slow, implement `block_sparse` behind a separate
configuration value. Do not describe it as identical to upstream Anew EPT.

## D014 — No coordinate-block latent in milestone one

Status: Accepted

`X_block` is returned and logged but does not alter `x_rep`. Coordinate
conditioning is a later, separately gated ablation with its own KL design.

## D015 — Stop gates are mandatory

Status: Accepted

Luna must stop and document the blocker if:

- the copied PVB baseline cannot be reproduced;
- source files differ materially from expected architecture;
- block pooling cannot match Anew;
- gate-zero parity fails;
- one-batch overfit fails;
- gradients become NaN or Inf;
- checkpoint coverage is insufficient.

## D016 — Bootstrap provenance

Status: Accepted

The target baseline was copied from PVB commit
`c08e5e3cd49d45c6d748387e78224843bd356f50`. AnewOmni source reuse will be
based on commit `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`. Both source
worktrees were clean before work began.

## D017 — Vendored Anew source map

Status: Accepted

All entries below come from AnewOmni commit
`926e99818ea18cf9d9b2064ce0319fe691b7a1f1` and are copied under the target
namespace:

| Source path | Target path | Modification |
| --- | --- | --- |
| `LICENSE` | `third_party/anewomni/LICENSE` | None |
| `models/modules/EPT/ept.py` | `third_party/anewomni/models/modules/EPT/ept.py` | Relative imports only |
| `models/modules/EPT/radial_basis.py` | `third_party/anewomni/models/modules/EPT/radial_basis.py` | None |
| `models/modules/GET/tools.py` | `third_party/anewomni/models/modules/GET/tools.py` | None |
| `models/modules/nn.py` | `third_party/anewomni/models/modules/nn.py` | None |
| `utils/nn_utils.py` | `third_party/anewomni/utils/nn_utils.py` | None |
| `utils/gnn_utils.py` | `third_party/anewomni/utils/gnn_utils.py` | None |
| `utils/register.py` | `third_party/anewomni/utils/register.py` | None |

The target uses package-relative imports and does not inject a sibling path at
runtime.

## D018 — Verified protein vocabulary mapping

Status: Accepted

PVB's 118-entry periodic-table order matches Anew's periodic table. PVB atom
IDs therefore map to Anew atom IDs with a `+1` dummy-token offset. PVB's
20-residue order matches Anew's amino-acid order, so residue block IDs map from
`PVB block_id - 118 + 1`. Any other PVB block ID is rejected in the protein-only
milestone; no arbitrary ligand offset is allowed.

## D019 — NumPy compatibility is target-local

Status: Accepted

The target replaces deprecated `np.compat.long` references in copied PVB
runtime files with `np.int64`, because the pinned `torch-ito` NumPy runtime has
no `np.compat`. This compatibility edit is confined to the target; source
repositories remain untouched.

## D020 — Decoder condition is shared across both branches

Status: Accepted

`H_block` is projected, indexed by `atom_block_id`, multiplied by
`tanh(block_gate)`, and added to both PVB TorchMD cross-attention branch inputs
(`x0` and `xt`) before neighbor embedding. At gate zero, the added tensor is
exactly zero and the decoder path matches the unconditioned path.

## D021 — Use an attention-aware dynamic batch budget

Status: Accepted

The training config uses the existing `DynamicBatchWrapper` with
`complexity: "n*n"`, matching the dominant pairwise attention cost. The
faithful Anew EPT path remains the default encoder; no separate `block_sparse`
approximation is introduced because the 2000-atom benchmark completed within
the available 80-GB A100 memory and xFormers is not installed.

## D022 — Treat the existing 2000-atom timing as diagnostic only

Status: Accepted

The previous table used too few measured steps and contained non-monotonic
PVB timing. It cannot establish a threshold or sudden performance failure.
The next profile must use warmup, repeated CUDA-synchronized measurements,
and raw step-level results.

## D023 — Faithful Anew EPT remains dense at atom level

Status: Accepted

Anew's `sparse_k` reduces block-edge expansion but does not sparsify the
dense atom-level EPT attention matrices. A separate `block_sparse` mode, if
needed, must be named and evaluated as an approximation.

## D024 — Use the official Anew checkpoint with provenance

Status: Accepted

Use the AnewOmni release checkpoint referenced by the upstream README. Record
the download URL, file size, SHA256, architecture audit, and key coverage.
Stop if the checkpoint cannot satisfy the current H-block contract.

## D025 — Freeze by actual checkpoint provenance

Status: Accepted

The frozen set is the union of the PVB and Anew loaders' actual matched
target keys. The optimizer receives only the complement. Module names or
shape similarity alone are not sufficient evidence for freezing.

## D026 — Train the fused milestone on a protein-only PDBBind view

Status: Accepted

ANI1x and PCQM4Mv2 are molecular records, while current Anew H-block
mapping is protein-only. Fused training and paired comparison therefore use
a block-complete protein-only PDBBind view. Unsupported ligand records are
reported rather than silently remapped.

## D027 — PVB is evaluation-only in this experiment

Status: Accepted

The existing PVB checkpoint is not retrained. It is evaluated on the full
original valid/test splits and on the paired protein-only PDBBind view.

## D028 — Valid selects the adapter; test runs once

Status: Accepted

Adapter training uses PDBBind protein-only train. Valid determines the best
checkpoint. Test is held out until the checkpoint is locked and is not used
for tuning. Fixed evaluation seeds report mean and standard deviation.

## D029 — Correctness and provenance precede optimization

Status: Accepted

No EPT approximation or performance rewrite is introduced until the timing
root cause, checkpoint coverage, frozen-key checks, and evaluation protocol
are all evidenced in the live handoff.

## D030 — Repeated synthetic scaling shows no 2000-atom threshold

Status: Accepted

With 10 warmup and 20 measured CUDA steps on an A100, the faithful Anew
H-block path remains finite and smooth from 512 through 2048 atoms. Its p50
step time is `0.259745 s` at 1800 atoms, `0.279082 s` at 2000, and
`0.289102 s` at 2048. The earlier apparent jump is therefore not established
as a size threshold; real-data shape and operator-level measurements must
precede any performance rewrite.

## D031 — Real PDBBind needs an explicit large-graph budget and protein-only view

Status: Accepted

The sampled legacy PDBBind records have roughly 2k–5k atoms and contain
protein residue blocks plus unsupported ligand element blocks. With the
current `n*n` budget of `2000`, DynamicBatchWrapper forms zero groups and
skips every PDBBind record. Adapter experiments must choose and validate a
larger budget after the protein-only view is materialized; `4e6` and `8e6`
are profiling candidates, not yet approved training settings.

## D032 — Migrate legacy fused checkpoints through state dictionaries

Status: Accepted

The prior fused checkpoint is a serialized object from the old fused namespace,
not a self-contained current fused model. It is converted once to a state-dict
artifact and loaded by explicit PVB/Anew role migration. The old
`anew_encoder.ctx_embedding` and unrelated decoder keys remain reported as
unexpected; they are never silently attached to the current model.

## D033 — Separate strict adapter from source-frozen complement

Status: Accepted

The strict `adapter` stage freezes all original parameters and trains only the
new block projector and gate. The `source_frozen` stage freezes the union of
actual matched checkpoint keys and exposes the full non-source complement for
optimizer membership audits.

## D034 — The observed slowdown is composite, not a proven 2000-atom threshold

Status: Accepted

T804/T805 operator traces show dense Anew atom-level attention with padded
shapes `[1,4,912,912]` and `[1,4,2152,2152]`, plus repeated block-level
fully-connected candidate construction in `knn_edges`. Attention memory and
PVB graph/decoder/backward costs all grow with the real batch shape. T801's
repeated synthetic curve remains smooth, so no faithful EPT change is allowed
until T806 audits padding and dynamic-batch transitions. Any approximation
would be a separately named `block_sparse` model and is not introduced by this phase.

## D035 — Dynamic batch cost must use exact materialized atom lengths

Status: Accepted

The PDBBind mmap property used by `UniDataset.get_len` is stale for the
requested protein-only view. In a 100-record sample per split, its ratio to
materialized atom count was below one for every record, with median ratios
`0.726/0.742/0.738` for train/valid/test. `DynamicBatchWrapper` can therefore
admit a batch whose real `n*n` cost is larger than the budget. In addition,
`graph_to_batch_nx` allocates `batch_size * ceil(max_N/8)^2` dense attention
work. Materialized split properties must be rewritten to exact protein-only
atom counts before training.

## D036 — Use the prior shape-matched fused Anew state for this run

Status: Accepted

The official Anew release is a full `Confidence` checkpoint with encoder
dimensions `hidden=512`, `edge=64`, `layers=6`, and `heads=8`. The current
fused model assembled in the previous work has `hidden=128`, `edge=16`,
`layers=2`, and `heads=4`; the official role audit therefore matches only
`1/68` keys and has `67` shape mismatches. The prior fused state dict has
already been converted and audits at PVB `150/150` plus Anew `68/68`.
For the requested adapter experiment, use that shape-matched state dict as
the Anew initialization source; do not resize or partially load the official release.

## D037 — Keep a separate exact-length protein-only materialized view

Status: Accepted

The source PDBBind mmap records mix protein residue blocks with unsupported
ligand element blocks and carry stale atom-count properties. The fused run uses
a derived train/valid/test view that retains complete protein residue blocks,
remaps atom and bond indices, preserves source IDs, and rewrites `get_len` to
the exact filtered atom count. The original source splits remain untouched and
continue to serve the PVB evaluation-only protocol.

## D038 — Treat 4e6 and 8e6 as profiled batch-budget candidates

Status: Accepted

After exact materialization, the current dynamic wrapper forms `1710/85/63`
groups at `4e6` and `3398/207/61` groups at `8e6` for train/valid/test,
while skipping records whose exact single-graph cost exceeds the budget.
These are measurement candidates only; source-frozen training smoke must
select a stable budget while tracking max padded EPT length and attention work.
The stale `n*n=2000` setting is not a valid PDBBind training budget.


## D039 — Distinguish source complement from effective gradient-bearing parameters

Status: Accepted

The source-frozen boundary is the exact complement of the PVB/Anew loaders'
matched target keys. In the current `anew_block` path this is `50` parameter
tensors, but `45` are legacy PVB encoder/prior tensors bypassed when the Anew
block encoder supplies the conditioning path. They remain in the explicit
complement and optimizer-membership audit, while their zero-gradient status
is reported separately from the five effective projector/gate tensors.

## D040 — Require a real source-frozen smoke before formal training

Status: Accepted

Formal training is allowed only after a fixed materialized protein batch has
finite loss and gradients, decreases loss, leaves every source-loaded tensor
bitwise unchanged, and uses exactly the non-source-loaded optimizer set. The
current smoke satisfies these checks: loss `1.0472314 → 1.0452697` in 20
fixed-draw steps, with no source checksum mismatch.


## D041 — Formal evaluation must not silently drop oversized records

Status: Accepted

The legacy `DynamicBatchWrapper` skips a record when its estimated cost is
over budget. Phase 9 instead uses exact materialized atom counts and
max-padded-N cost. Records that exceed the budget are emitted as explicit
singleton batches and counted in the artifact; they are never silently
removed from train, valid, or test.

## D042 — Preserve PVB behavior while supporting bondless molecular records

Status: Accepted

The original PVB graph helper assumes a non-empty explicit bond index. A
valid ANI record exposed this edge case during the evaluation-only baseline
run. The target adds a minimal guard returning an all-zero bond-type mask for
empty bonds; `/workspace/PVB` remains read-only and unchanged.

## D043 — Separate original PVB evaluation from paired protein-only comparison

Status: Accepted

The existing PVB checkpoint is evaluated, without retraining, on the complete
original valid/test splits, including `pcqm4mv2`, `ani1x`, and `pdbbind`. The
fused H-block and PVB `off` comparison uses the derived protein-only PDBBind
valid/test view so that both models receive the same supported protein input
and explicit block metadata.

## D044 — Treat the source-frozen fused result as diagnostic, not an improvement claim

Status: Accepted

The one-epoch source-frozen fused checkpoint passed provenance, optimizer,
checksum, and complete traversal gates, but its paired loss is substantially
worse than PVB `off`: valid batch loss `1.064952` versus `0.273862`, and test
batch loss `1.043927` versus `0.255278`. These reports are retained as the
baseline experiment result; further improvement requires a new experiment and
must not be described as achieved by this run.
