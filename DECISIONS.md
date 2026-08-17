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

## D059 — Keep Phase 10 and Phase 11 in the same repository but on separate branches

Status: Accepted

Phase 10 and the lightweight shared-H-block architecture are directly
comparable stages of the same PVB/Anew fusion project. They share loaders,
datasets, evaluation, and provenance infrastructure, so a new repository would
fragment history. Preserve Phase 10 on `phase10/pvb-posterior-hblock` (and merge
or tag it), then develop Phase 11 on `phase11/pvb-shared-hblock` from the final
Phase 10 commit.

## D060 — Continue the same four live documents and numbering

Status: Accepted

Phase 11 is appended to `PLAN.md`, `TASKS.md`, `DECISIONS.md`, and
`HANDOFF.md`. Phase 10 remains immutable historical evidence. Phase 11 tasks
start at T1100 and decisions continue at D059.

## D061 — Reuse the PVB encoder instead of running a second Anew EPT

Status: Accepted

`pvb_shared_hblock` pools the scalar `h_atom` already produced by the
pretrained PVB TorchMD encoder. It does not instantiate or execute Anew EPT.
The goal is to isolate whether block-level conditioning is useful without the
large time, memory, and parameter cost of a second equivariant encoder.

## D062 — Preserve the complete PVB posterior and stop gradients at the shared feature branch

Status: Accepted

PVB `h_atom`, posterior log variance, `x_rep`, and KL retain their original
semantics. During the source-frozen experiment, the H-block branch consumes a
detached view of `h_atom`; its gradients cannot update or perturb the PVB
encoder. A later fine-tuning stage would require a separate decision and mode.

## D063 — Keep legacy PVB block input separate from semantic pooling blocks

Status: Accepted

Legacy atom-level `btype` remains unchanged for PVB checkpoint compatibility.
`atom_block_id`, `semantic_block_type`, `block_lengths`, and `block_batch`
describe the new pooling/semantic blocks. Standard protein blocks are residues;
small-molecule and ligand blocks are Anew principal-subgraph fragments.

## D064 — Use variance-preserving pooling by default

Status: Accepted

The default block representation is `scatter_sum(h_atom) / sqrt(block_length)`,
matching Anew's variance-preserving pooling convention. Plain mean pooling is
retained only as an explicit ablation because it can shrink representations of
larger residues/fragments.

## D065 — Inject the shared condition once after decoder branch merging

Status: Accepted

The Phase 11 condition is added after the PVB decoder combines its `x0` and
`xt` cross-attention inputs and before the equivariant attention stack. This
avoids double injection and gives a single zero-gated location with strict
baseline parity. Older fusion modes retain their existing injection semantics.

## D066 — Separate Phase 11A structural pooling from Phase 11B fragment semantics

Status: Accepted

Phase 11A first tests shared residue H-block pooling on the already validated
protein-only PDBBind view. Phase 11B adds Anew fragment vocabulary and semantic
embedding only after Phase 11A correctness, source-freeze, and performance
gates pass. This prevents tokenizer or vocabulary effects from obscuring the
shared-encoder architectural test.

## D067 — Perform Anew fragment tokenization offline from complete chemistry

Status: Accepted

Principal-subgraph tokenization runs during CPU preprocessing using raw
chemical identity and bond orders. It is never run in the GPU model forward.
Existing PVB binary `bond_index` data must not be treated as sufficient when
bond order/aromaticity has been lost. Missing chemistry is a recorded blocker,
not an invitation to guess fragment membership.

## D068 — Reuse only the Anew block embedding with exact provenance

Status: Accepted

Phase 11B may extract and freeze the official Anew
`BlockEmbedding.block_embedding` tensor if its exact checkpoint key, shape,
dtype, vocabulary order, and SHA256 are proven. It does not reuse Anew's atom
embedding because PVB already has a pretrained atom embedding. Random semantic
embeddings must be labeled as a separate ablation and never called official
Anew representations.

## D069 — Use a low-rank adapter and zero-initialized gates

Status: Accepted

The structural branch uses a default bottleneck rank of 32. The structural and
semantic contributions use independent zero-initialized scalar gates. At zero
gate, the full stochastic objective and inference source sample must match PVB
`off` under identical RNG state.

## D070 — Preserve all earlier modes, checkpoints, and artifacts

Status: Accepted

Phase 11 adds `pvb_shared_hblock`; it does not change `off`, `anew_block`, or
`anew_block_pvb_posterior`. All outputs go below a new `phase11` directory.
Phase 9 and Phase 10 artifact hashes must be checked before and after Phase 11.

## D071 — Select checkpoints by reconstruction on valid, not by test or KL-dominated totals

Status: Accepted

The primary valid-selection metric is `rec_total = rec_vel + rec_drf`. KL and
total loss remain mandatory reported diagnostics. Test is evaluated once only
after a checkpoint is locked, using the same fixed seeds and paired views as
the relevant prior phase.

## D072 — Do not describe Phase 11 as the original Anew H-block encoder

Status: Accepted

Phase 11 uses PVB contextual atom states, Anew-style pooling, and optionally
Anew fragment vocabulary/block embeddings. Because it omits Anew EPT, reports
must call it a lightweight shared-PVB H-block model rather than a faithful
Anew H-block representation.

## D073 — Phase 11A must share PVB h_atom to avoid the dual-encoder cost

Status: Accepted

T1101 confirms that the Phase 10 corrected path performs both the PVB encoder
and Anew EPT. Vendored Anew EPT builds dense atom-level attention tensors whose
work and memory grow approximately with the square of the largest atom count.
Phase 11A therefore reuses the scalar `h_atom` already produced by PVB, pools it
into blocks, and must not construct or execute `AnewBlockEncoder`/Anew EPT in
`pvb_shared_hblock`.

## D074 — Use exact materialized traversal for Phase 11 cost and quality claims

Status: Accepted

The generic dynamic batch profiler can report oversized-record skips under
small `n*n` budgets. Phase 11 profiling, training, validation, and test claims
must use the exact materialized protein-only traversal that emits oversized
records as explicit singleton batches. A one-warmup/one-step shared-GPU probe
is diagnostic evidence only; final performance claims require the later
multi-size benchmark gate.


## D075 — Keep `pvb_shared_hblock` PVB-only during Phase 11A plumbing

Status: Accepted

The new mode is a separate compatibility surface. It follows the PVB encoder
and posterior path and exposes optional scalar atom state without constructing
Anew's atom embedding or EPT. Existing `off`, `anew_block`, and
`anew_block_pvb_posterior` modes retain their prior model construction and
checkpoint semantics. Pooling and decoder conditioning are added only in the
subsequent Phase 11A tasks.

## D076 — Reuse vendored variance-preserving pooling in Phase 11A

Status: Accepted

Phase 11A calls the vendored Anew std_conserve_scatter_mean helper for
sum(h_atom) divided by sqrt(block length). It receives explicit integer block
metadata, detaches PVB h_atom, and does not infer membership from coordinates.
The rank-32 adapter and scalar gate remain separate from PVB source keys.

## D077 — Inject the shared condition once after decoder branch merging

Status: Accepted

`pvb_shared_hblock` uses a dedicated `post_cross_condition` added after the
decoder's `x0`/`xt` cross-branch merge and before equivariant attention. The
legacy `block_condition` path remains unchanged, and the new tensor is passed
through training, inference, and realization only for the new mode.

## D078 — Keep the gate zero while preserving its first-step gradient

Status: Accepted

The scalar `shared_hblock_gate` is initialized to zero for exact baseline
parity. The adapter's final linear layer uses ordinary initialization rather
than zero initialization; zeroing both would make the initial gate gradient
zero and prevent the adapter-only stage from starting.

## D079 — Require `pvb_full` for Phase 11A source-frozen runs

Status: Accepted

The shared mode has no Anew encoder source role. Its checkpoint contract is a
freshly constructed `pvb_shared_hblock` model plus the complete `pvb_full`
role, including PVB encoder, posterior heads, decoder, and output heads.
Legacy `pvb` role semantics remain unchanged.

## D080 — Freeze the exact loaded union and optimize its exact complement

Status: Accepted

Phase 11A source-frozen training uses the loader's actual matched keys, not
module-prefix guesses. Every loaded PVB parameter must be frozen bitwise, and
the optimizer must contain exactly the seven new shared adapter/gate tensors.

## D081 — Full-objective parity is the Phase 11A gate

Status: Accepted

Gate-zero validation must compare the complete stochastic training objective,
not only the decoder tensor. It covers the PVB posterior state, `x_rep`, KL,
both reconstruction terms, total loss, inference source sample, and decoder
output under controlled Torch and NumPy RNG states.

## D082 — Preserve historical checkpoint compatibility as a separate contract

Status: Accepted

Phase 9 legacy and Phase 10 corrected checkpoint paths are compatibility
fixtures, not migration inputs. The new shared mode may add its own `pvb_full`
role, but it must not reinterpret or silently rewrite the old `anew_block` or
`anew_block_pvb_posterior` checkpoint semantics.

## D083 — Shared-mode profile is diagnostic, not a full-split performance claim

Status: Accepted

The Phase 11A shared path reuses PVB `h_atom` and does not construct Anew EPT.
The exact-record timing/memory comparison is a correctness and cost-boundary
check; full-split performance claims require a separately specified benchmark
with identical traversal, batching, warmup, and device controls.
## D084 — Phase 11 checkpoint selection is valid-only and test is post-lock

Status: Accepted

The Phase 11A runner may traverse train and valid during tuning and selects
exactly one checkpoint by validation `rec_vel + rec_drf` batch mean. It writes a
lock manifest with `test_evaluated=false`; the paired evaluator refuses an
unlocked or already-tested manifest and only then runs the three-seed valid/test
comparison. Test metrics cannot replace the selected checkpoint.
## D085 — Treat output-volume exhaustion as an explicit Phase 11 artifact constraint

Status: Accepted

The shared `/output` volume reached zero free bytes while writing the first
formal Phase 11 checkpoint after train and valid had completed. Phase 11 must
never delete Phase 9/10 artifacts to recover space. The runner may alias its
rolling last-checkpoint path to the selected best path, retaining one complete
checkpoint plus its lock/report; any capacity failure before a complete lock is
written invalidates the run and requires a restart from the immutable PVB source
checkpoint.
## D086 — Phase 11 checkpoint replacement must be single-file under zero-slack storage

Status: Accepted

When the output volume has room for one complete Phase 11 checkpoint but not two,
PyTorch direct overwrite is unsafe because the old archive remains allocated
while the new archive is written. The Phase 11 runner may unlink only its own
previous best/last target immediately before replacement, then write one
complete checkpoint. A failed replacement invalidates the formal run and must
not be repaired by deleting Phase 9 or Phase 10 artifacts.
## D087 — Phase 11 evaluation lock may omit optimizer moments under capacity pressure

Status: Accepted

The Phase 11A checkpoint is selected for inference and paired evaluation, not
for optimizer-state continuation. Because the immutable PVB model state nearly
fills the remaining output capacity, the locked Phase 11 checkpoint may omit
optimizer moments, provided the run report records `optimizer_state_saved=false`
and proves exact optimizer membership during training. This does not relax
source coverage, model-state completeness, or checkpoint hash requirements.
## D088 — Phase 11 adapter-only lock keeps PVB source external and explicit

Status: Accepted

The Phase 11A evaluation checkpoint stores only the seven newly trained
adapter/gate state tensors. The immutable PVB source checkpoint remains external
and must be loaded first with complete `pvb_full` coverage before applying the
adapter state. The lock must identify `checkpoint_kind=phase11_adapter_only`,
record source path/checksums, and report adapter coverage separately; this is a
storage-preserving representation, not a relaxation of model-state provenance.

## D089 — Select the Phase 11A adapter by valid reconstruction and lock before test

Status: Accepted

The complete Phase 11A train/valid run selected epoch 4 solely by minimum valid
batch-mean `rec_vel + rec_drf = 0.2183195718`. Its valid KL remained
`0.0069576662`, consistent with the PVB posterior, while all 195 source tensors
remained bitwise unchanged and the exact seven-tensor adapter/gate complement
was optimized. The adapter-only checkpoint and its external PVB source are
locked; test is now evaluated once and cannot alter checkpoint selection.

## D090 — Phase 11A shared H-block passes the locked paired gate

Status: Accepted

The one-time three-seed paired evaluation used identical protein-only valid/test
views for PVB off, Phase 9 legacy, Phase 10 corrected, and Phase 11 shared. The
shared adapter reduced batch-mean `rec_total` from `0.268296109` to
`0.255006258` on valid and from `0.249656701` to `0.237041126` on test, while
KL stayed at `0.006957666` / `0.007027225`, matching the PVB posterior. Phase 10
corrected was also improved. The legacy mode's lower reconstruction remains
KL-dominated and is not claimed as a total-loss improvement. Gate P11A passed;
Phase 11B must first prove exact official Anew embedding provenance.

## D091 — Official Anew block embedding provenance is approved

Status: Accepted

At Anew commit `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`, the official full
model key `base_model.autoencoder.embedding.block_embedding.weight` matches the
derived and extracted `embedding.block_embedding.weight` tensor exactly:
`437 x 512`, `float32`, tensor SHA256
`2ba7c22abf1ca550d354d282e7c4ed2278ab972789ce223bf50db31abb69ddf`. The complete
437-entry block order and tokenizer asset checksum are captured in the T1109
provenance report. This permits exact frozen table reuse, but does not by itself
approve tokenizer integration, semantic data generation, or a new model mode.

## D092 — Vendor tokenizer behavior only after exact source parity

Status: Accepted

T1110 vendors the pinned Anew tokenizer, vocabulary, helper files, and asset
under a namespaced package. The 13 mapped files are byte-identical after three
documented import-only adaptations, and source/target vocabulary plus
tokenization probes match on representative molecules. The fused runtime has no
sibling-repository import. T1111 must validate complete chemical metadata before
any semantic fragment IDs are used by model code.

## D093 — Block semantic integration on incomplete ligand chemistry

Status: Blocked

The full T1111 audit found five train SDF bonds with `UNSPECIFIED` order and
upstream tokenizer mapping assertions on 508 records. Anew accepts only
single, double, triple, and aromatic bond types; assigning an unspecified bond
to one of them would be a guessed chemical label. Until those source records
are repaired from an authoritative chemical source and the tokenizer fallback
is either resolved or explicitly accepted by a new decision, T1112 and all
semantic model integration remain blocked. The existing PVB protein-only shared

## D094 — Diagnose tokenizer fallback from original chemistry sources

Status: Accepted

Tokenizer fallback is not assumed to mean that SDF files are absent. Phase 12
must inspect the original `pcqm4mv2`, `ani1x`, and `pdbbind` trees and distinguish
missing bond/aromaticity metadata from Anew tokenizer mapping assertions or atom
ordering incompatibilities before choosing a reconstruction method.

## D095 — Apply a measured four-hour stop gate to xyz2mol

Status: Accepted

`xyz2mol` may be used only after a representative speed benchmark and projected
full-runtime calculation. If the full conversion is estimated to exceed four
hours, stop without launching it and document the measured basis. No guessed
bond order is acceptable merely to unblock semantic integration.

## D096 — Preserve raw chemistry and isolate Phase 12 outputs

Status: Accepted

Original chemistry files, PVB/Anew source repositories, protected Phase 9/10/11A
artifacts, and Phase 11 protein-only test views remain unchanged. Any successful
chemistry reconstruction is written to a new, provenance-recorded Phase 12
output directory.

## D097 — Use the official PDB acquisition and ept_release materialization path

Status: Accepted

PDB acquisition uses the supplied official wwPDB `rsyncPDB.sh` URL. Existing
`/data4/PVB/pdb/ept_release` selection and split scripts are the authority for
the full PVB dataset; the requested half dataset is derived afterward with an
explicit manifest. Processed PDB data is not introduced into Phase 11 tests.

## D098 — Run the two Phase 12 data lanes concurrently

Status: Accepted

Chemistry/xyz2mol feasibility and PDB/full-half materialization are independent
and should run in separate workers. `TASKS.md` retains one `IN_PROGRESS`
coordinator (T1200); T1201 and T1202 are parallel tracked lanes under it.

## D099 — Stop full xyz2mol reconstruction under the measured gate

Status: Accepted

The Phase 12 audit found that SDF absence is not the general fallback cause.
PCQM has SDF chemistry, ANI lacks explicit bond labels, and PDBBind has
present-but-unspecified bonds. The bounded PDBBind xyz2mol rate projects
about 5.4 hours for all 15,487 records, so full conversion is abandoned.
No guessed bond order or raw-source modification is permitted; the Phase 11
semantic blocker remains until authoritative chemistry is supplied.

## D100 — Keep EPT selection intermediate separate from PVB materialization

Status: Accepted

The existing ept_release PDB processor emits EPT-specific `X/B/A/`
`atom_positions/block_lengths/segment_ids` data, not PVB's
`atype/btype/x0/b0/bond_index` data. Use the EPT processor and index for
structure-level selection, then reuse PVB `data/pdb_dataset.py` to generate
fresh PVB `train_block`, `valid_block`, and `test_block` outputs. The same
manifests define the half dataset, and neither output is used by Phase 11.

## D101 — Use resumable parallel rsync for precompressed PDB files

Status: Accepted

The official wwPDB template and module remain the source of truth. Because the
remote divided files are already `.ent.gz`, the Phase 12 mirror may omit rsync
transport compression after a measured check, use non-overlapping prefix lanes,
and retain `--partial` without `--delete`. This changes scheduling only; it
does not change the source set or the raw PDB contents.

## D102 — Use valid SDF chemistry directly

Status: Accepted

A readable SDF with explicit bond orders and aromaticity is the authoritative
input. It must bypass xyz2mol and any other coordinate-based bond inference.
Tokenizer fallback is not evidence that the SDF is absent; damaged or
unspecified records are a separate repair population.

## D103 — PDBBind repair input is ligand-only

Status: Accepted

PDBBind bond-assignment probes must pass only the ligand atom table and ligand
conformer coordinates from the SDF. Receptor/protein PDB coordinates must never
be concatenated into the input. The previous aggregate PDBBind timing is
superseded because its input boundary was not recorded sufficiently to prove
this contract.

## D104 — Process ANI-1x separately by unique molecular group

Status: Accepted

ANI-1x provides coordinates and atom identities but no explicit bond-order or
aromaticity arrays. Bond assignment is therefore a separate coordinate-only
workstream. Infer and validate at most once per unique molecular group, then
reuse the accepted topology for its conformers; do not run xyz2mol independently
for all 4,956,005 frames.

## D105 — Compare RDKit and Open Babel without treating either as ground truth

Status: Accepted

RDKit rdDetermineBonds is documented as a C++ implementation of the xyz2mol
algorithm (https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html),
so it is a useful speed/API alternative but not an independent chemistry oracle.
Open Babel's ConnectTheDots() and PerceiveBondOrders()
(https://openbabel.org/docs/WritePlugins/AddFileFormat.html) is a second
heuristic candidate. Any inferred result requires valence, charge, aromaticity,
atom-order, and known-edge validation before it can be written.

## D106 — Do not accept geometry-only labels for the damaged Phase 12 records

Status: Accepted

The corrected ligand-only probes for the five damaged PDBBind SDFs returned no
usable xyz2mol graph. No inferred bond labels will be written from those
outputs. The Phase 11 chemistry blocker remains until an authoritative source,
validated repair, or separately approved fallback policy is available.

## D107 — Correct the scope of the historical PDBBind timing

Status: Accepted

D099 remains a historical stop decision, but its all-record timing is not a
valid ligand-only benchmark. The corrected Phase 12 policy is to use complete
SDF chemistry directly and reserve bounded inference for damaged ligand-only
SDF records, with no protein/receptor coordinates in the inference input.

## D108 — T1111 prerequisites remain incomplete after the Phase 12 audit

Status: Blocked

The existing T1111 CPU audit remains the authoritative Phase 11 gate. It
traversed all 6,947 records and proved exact PVB/SDF heavy-atom order, but five
train SDF bonds still have `UNSPECIFIED` order and 508 records still require
explicit Anew tokenizer fallback. The corrected Phase 12 ligand-only probes
did not produce usable graphs for the five damaged records, so no semantic
fragment metadata may be materialized and T1112–T1117 must not start.

## D109 — Report P11A separately from blocked semantic Phase 11B

Status: Accepted

The P9/P10/P11A architecture and loss schematic records the completed
`pvb_shared_hblock` adapter as the current Phase 11 result. It must not be
described as completed semantic Anew fragment fusion: T1111 remains blocked.
The schematic is therefore a provisional Phase 11A report and does not close
the Phase 11 gate or authorize claims about the unrun semantic branch.

## D110 — Treat the five `UNSPECIFIED` SDF repairs as provisional only

Status: Accepted

The five ligand-only probes support `SINGLE` as a structural candidate, but
`xyz2mol` and RDKit produced no validated graph and Open Babel did not validate
the chemistry. The isolated `0 -> SINGLE` files pass Anew only with
`sanitize=False`; they must not replace raw SDFs or unblock T1111.

## D111 — The 508 fallback records are tokenizer failures, not missing SDF bonds

Status: Accepted

All 508 records are in the half PDBBind view and have readable SDFs, complete
bond types, and exact PVB/SDF atom order. Their failure is Anew
`get_submol_atom_map()` assertion. Geometry tools must not rewrite this
authoritative chemistry; resolution requires tokenizer policy or source data.

## D112 — Require graph preservation and normal validation for any repair

Status: Accepted

A repair is accepted only if it preserves the known SDF graph, passes normal
RDKit/Open Babel validation, retains atom order and charges, and passes Anew
tokenization. The current five candidates fail this gate; the 508 require no
bond repair and remain a tokenizer/block-semantics decision.

## D113 — Do not infer Gaussian H-block benefit from Phase 9

Status: Accepted

Phase 9 reconstruction and KL changed together because Anew block log variance
replaced the PVB atom posterior variance. Its lower reconstruction and much
higher KL are therefore posterior-confounded evidence and do not justify making
H-block stochastic.

## D114 — Raw parameter count does not explain the Phase 11A gain

Status: Accepted

Phase 11A trains 17,185 adapter/gate parameters, whereas Phase 10 trains 33,281.
The small Phase 11A reconstruction improvement cannot be attributed to simply
adding more trainable parameters, though capacity and topology still require
matched controls.

## D115 — Pre-register five independent Phase 13 hypotheses

Status: Accepted

Phase 13 separately tests record-specific H-block information, trainable
capacity, block pooling, feature-source alignment, and conditioning injection
location. Conclusions must follow matched comparisons and may not transfer
across confounded variants.

## D116 — Preserve the complete PVB posterior throughout Phase 13

Status: Accepted

Every corrected ablation uses the original PVB `x_rep`, log variance, and KL.
Anew `Wx_log_var` remains diagnostic-only. Phase 13 adds no block KL and no
coordinate residual.

## D117 — Require matched capacity and exact source freezing

Status: Accepted

Content controls use the same adapter/gate architecture, trainable membership,
initialization policy, and parameter count. All source-loaded PVB and Anew
parameters remain frozen and bitwise checksummed. Any unavoidable mismatch must
be quantified and cannot support a capacity claim.

## D118 — Shuffle only within each sample and deterministically

Status: Accepted

The shuffled H-block control uses record-and-seed-deterministic permutations
within each sample. It must preserve block count, shape, and feature norms and
must never exchange information across samples.

## D119 — Match adapter and injection when comparing PVB and Anew features

Status: Accepted

Feature-source alignment is tested by routing shared-PVB H-block and the current
shape-matched Anew H-block through the same rank-32 adapter, post-merge
injection, optimizer protocol, and checkpoint-selection rule.

## D120 — Lock all Phase 13 checkpoints using valid reconstruction only

Status: Accepted

Valid `rec_total` selects each checkpoint. Test remains unseen until all
variants and hypotheses are locked, then runs once on identical complete paired
views and seeds. Because the expected effect is small, per-record deltas and
bootstrap confidence intervals accompany aggregate metrics.

## D121 — Defer any stochastic H-block to a separate future phase

Status: Accepted

A Gaussian block latent is considered only if Phase 13 first shows that
deterministic record-specific block information adds value. Any future version
must have a separate latent and KL schedule with warmup/free bits, enter through
a zero gate, and never replace the PVB posterior. It is not implemented or
trained in Phase 13.

## D122 — Defer semantic Phase 11B because fragment coverage is unresolved

Status: Accepted

The Phase 11B branch is skipped for now. The five damaged SDF records and the
508 complete-chemistry tokenizer failures show that Anew fragment metadata is
not complete enough for a faithful semantic experiment. T1111 and T1112-T1117
remain deferred; no fragment fallback is silently promoted.

## D123 — Run Phase 13 on protein-only views while PDB acquisition is asynchronous

Status: Accepted

Phase 13 uses the already validated protein-only Phase 9/10/11A views and is
not blocked by the unfinished PDB materialization. The PDB mirror may continue
in the background, but its partial tree is not an input to Phase 13 and does
not change the single-active-task invariant.

## D124 — Preserve disk headroom before PDB materialization

Status: Accepted

The current /data4 snapshot has approximately 410 GB free. The partial PDB
tree is about 9.3 GB and 46,119 files, versus an estimated 53.6 GB complete
remote tree. This is sufficient for the raw mirror and planned intermediates,
but full materialization must recheck free space and avoid duplicate copies or
unbounded temporary archives.

## D125 — Start Phase 13 with matched PVB-only controls

Status: Accepted

The first Phase 13 training set compares the PVB off baseline with real,
sample-local shuffled, fixed constant, and atom-no-pool controls. This isolates
record-specific content, adapter capacity, and pooling before adding the more
expensive Anew feature-source or injection-location comparisons.

## D126 — Use one explicit registry for Phase 13 adapter variants

Status: Accepted

The public registry in `utils/phase13_ablation.py` maps experiment names to
internal adapter values. Existing `off`, `anew_block`,
`anew_block_pvb_posterior`, and Phase 11A loading semantics remain unchanged;
no legacy mode is silently redirected to a Phase 13 variant.

## D127 — Stage-A locked-adapter gap is dependency evidence only

Status: Accepted

Using the same locked Phase 11A adapter weights on valid, real H-block input
produced `rec_total` 0.218320, versus 0.221847 for shuffled, 0.236439 for
constant, and 0.246207 for atom-no-pool, with identical PVB-like KL. This
supports record-specific H-block dependence and/or pooling dependence in the
locked adapter, but it is an out-of-distribution control and cannot replace
matched retraining or justify a test claim.

## D128 — Complete Phase 13 training before the one-time test traversal

Status: Accepted

The four matched controls completed valid-only training under one protocol with
17,185 trainable adapter/gate parameters and unchanged PVB source checksums.
Their checkpoints are locked by valid `rec_total`; test remains unread until a
single paired evaluation. Phase 11B remains skipped because Anew fragment
coverage is unresolved, and Phase 12 PDB materialization is not an input to
these protein-only ablations.

## D129 — Phase 13 four-control result is capacity-dominant with a small information signal

Status: Accepted

In the valid-only selected, source-frozen, equal-parameter comparison, all
adapter controls improved PVB off. Real shared H-block beat sample-local
shuffled and constant controls by approximately `0.00065–0.00070`
batch-mean `rec_total` on both valid and test, supporting a small
record-specific contribution. The much larger common improvement is compatible
with adapter/injection capacity. Atom-no-pool was best, so the current
atom-to-block pooling path is not established as optimal. Phase 9's KL issue
was absent because every Phase 13 control retained the PVB posterior. This does
not support making H-block Gaussian.

The aggregate paired evaluator did not emit per-record vectors. Because test
was intentionally traversed once, no second test pass is authorized for
post-hoc bootstrap. T1304/T1305 feature-source and injection-location
comparisons remain unrun and are not claimed by this result.

## D130 — Close the Phase 13 four-control tranche and defer unrun extensions

Status: Accepted

The PVB-matched real/shuffled/constant/atom-no-pool tranche is closed with
complete valid-only selection, one paired valid/test aggregate traversal, and
final target/source/artifact audits. T1304/T1305 feature-source and
injection-location comparisons were not run and remain TODO for a separately
registered experiment that emits per-record metrics before its one-time test
pass. Phase 11B remains deferred because Anew fragment coverage is unresolved.
