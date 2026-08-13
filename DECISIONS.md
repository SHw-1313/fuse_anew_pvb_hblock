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

## D045 — Phase 9 degradation is KL-dominated; reconstruction improved

Status: Accepted

On paired protein-only valid, legacy fused H-block reduced `rec_vel`
from `0.102848` to `0.064429` and `rec_drf` from
`0.165448` to `0.125655`, but increased KL from
`0.006958` to `1.093584`. Total-loss degradation is therefore
dominated by KL rather than reconstruction.

## D046 — Preserve the complete PVB posterior for x_rep and KL

Status: Accepted

The corrected Phase 10 mode runs the original PVB encoder and posterior heads
to produce `x_rep` and `kl_loss`. Anew H-block output is
decoder conditioning only and cannot replace the PVB posterior contract.
T1004 implements this as a separate PVB path followed by Anew `H_block`
projection/broadcast; corrected inference follows the same posterior contract.

The T1001 read-only audit reproduced the loss formula with maximum aggregate
error `1.24e-8`, measured a `157.18x` batch-mean KL ratio, and confirmed that
the legacy Anew `Wx_log_var` tensors were loaded/frozen while PVB posterior
keys were not loaded by the legacy role. The recorded KL-dominated diagnosis
therefore agrees with the implementation and Phase 9 reports.

## D047 — Preserve legacy Phase 9 behavior through an explicit mode

Status: Accepted

Existing `off` and `anew_block` semantics, checkpoints, and loaders
remain reproducible. Phase 10 uses separately named
`anew_block_pvb_posterior` semantics and never silently changes legacy
behavior. T1003 added the explicit mode to construction and training/config
glue; focused fusion and checkpoint tests passed while retaining the legacy
mode default. T1006 additionally confirms that corrected `pvb_full` plus Anew
coverage matches all `263` source state tensors bitwise and leaves only the
five projector/gate tensors in the optimizer complement.

## D048 — Add a full-PVB checkpoint role for the corrected path

Status: Accepted

The new `pvb_full` role loads all expected compatible PVB
`encoder.*`, `W_vec_mu.*`, `W_vec_log_var.*`,
`decoder.*`, `vel_ffn.*`, and `drf_ffn.*` keys with
complete coverage reporting. The existing `pvb` role remains the
Phase 9 scoped decoder/head load.

T1002 evidence: the real Phase 9 PVB state dict loaded `195/195`
expected keys with zero missing, unexpected, or shape-mismatched keys. The
coverage report is stored under
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json`.

## D049 — Gate-zero parity covers the complete stochastic objective

Status: Accepted

Corrected-mode gate-zero parity compares PVB posterior state, KL,
velocity/drift/total losses, stochastic source samples, and decoder outputs,
not only conditioned decoder features. T1005 passes these comparisons within
`1e-6`, including the complete inference trajectory and the legacy loader
coverage check.

T1006 additionally confirms that the source-loaded checkpoint tensors match
the constructed corrected model bitwise before freezing.

## D050 — Anew Wx_log_var is diagnostic-only in Phase 10

Status: Accepted

Anew `Wx_log_var` and block-level variance statistics are logged for
diagnosis but do not construct `x_rep`, contribute to PVB KL, or enter
the Phase 10 loss graph. T1004 routing tests confirm that changing the Anew
variance head leaves corrected loss unchanged and its gradients absent. T1007 also records Anew variance quantiles and diagnostic KL separately from the PVB
posterior: Anew diagnostic KL is `1.018065`, while the PVB KL used by the
corrected loss is `0.006921472` on the fixed real batch.

## D051 — Isolate posterior mismatch with the current shape-matched Anew state

Status: Accepted

Phase 10 uses the existing shape-matched Anew `128`-hidden, two-layer
state to isolate posterior correction from width migration. Rebuilding/loading
the official `512`-hidden, six-layer encoder is deferred to Phase 11.

## D052 — Valid reconstruction selects the checkpoint; test runs once

Status: Accepted

Phase 10 selects exactly one checkpoint using valid
`rec_total=rec_vel+rec_drf` after complete epochs. Test remains held
out until the checkpoint is locked and is evaluated once with fixed seeds
`20260810`, `20260811`, and `20260812`.

## D053 — Smoke entrypoints enumerate every explicit fusion mode

Status: Accepted

Phase 10 keeps `off` and `anew_block` defaults unchanged but requires smoke and
CLI paths to name `anew_block_pvb_posterior` explicitly. A stale two-choice
smoke parser is a compatibility defect, so it was fixed before the T1008 gate
was accepted.

## D054 — Lock the best corrected adapter before any test evaluation

Status: Accepted

Phase 10 completed four valid epochs and selected the epoch-3 checkpoint by
valid rec_total only. The training session ended during the following epoch
before validation; that interruption is recorded in the training report and
does not authorize test-based selection. The checkpoint and lock manifest are
immutable for the paired evaluation.

## D055 — Use one exact paired evaluator for all three models

Status: Accepted

PVB off, the Phase 9 legacy mode, and the corrected Phase 10 mode must use the
same materialized protein-only valid/test views, padded-cost batching, and
seeds 20260810, 20260811, and 20260812. The paired evaluator loads each model
after constructing its target architecture, asserts identical traversal
counts, and evaluates test only after the valid-only lock.

## D056 — Recover completed paired evaluation without rerunning test

Status: Accepted

The first Phase 10 paired evaluator completed all six model/split traversals
and emitted their aggregate lines, then failed only while serializing
tensor-valued resume metadata. The report is recovered from that immutable log
with test rerun disabled. The evaluator is fixed to retain only JSON-safe
metadata for future runs. The recovered report records aggregate mean/std,
fixed seeds, exact paired counts, and the missing per-seed-detail limitation
explicitly.

## D057 — Corrected paired result supports a performance claim only under the paired protocol

Status: Accepted

On the locked valid-selected checkpoint and the one-time paired evaluation,
anew_block_pvb_posterior preserves PVB KL and improves both reconstruction
terms and total loss relative to off on the paired valid and test views.
The Phase 9 legacy mode still has lower reconstruction but remains total-loss
dominated by its Anew-derived KL. This is a paired aggregate result, not a
claim about per-seed superiority; the recovered report lacks individual
per-seed metric records because the original writer failed after all six
traversals.

## D058 — Official Anew alignment is a separate Phase 11 model

Status: Accepted

Phase 10 closes with the shape-matched Anew 128-hidden, two-layer encoder used
to isolate the posterior-contract issue. The official Anew 512-hidden,
64-head, six-layer, eight-radial-setting representation must be introduced
under a separate Phase 11 mode with full shape/key coverage and its own parity
and checkpoint gates. Partial official-weight loading, silent shape adaptation,
Anew unfreezing, coordinate residuals, and new KL terms remain out of scope
until that separate baseline is proven.
