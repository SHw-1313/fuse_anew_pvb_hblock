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
