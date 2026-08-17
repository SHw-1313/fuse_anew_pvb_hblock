# Phase 13 T1300 protocol and provenance

Date: 2026-08-14

## Scope

Phase 11B is deferred because Anew fragment metadata coverage is unresolved.
Phase 13 uses only the existing protein-only Phase 9/10/11A views and does not
read the partial PDB mirror.

## Frozen source and target revisions

- Read-only PVB: commit `c08e5e3cd49d45c6d748387e78224843bd356f50`, clean at audit.
- Read-only AnewOmni: commit `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`, clean at audit.
- Target HEAD at protocol freeze: `fbf8302d57942dbc41a52e0e1019ecb8c0287687`
  (`fix KL loss`); target already contained preserved user/Phase 9–12 changes.
- Phase 9 PVB full checkpoint:
  `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt`
  SHA256 `4f0ad88356c7159cd5d0b9641b6c1e5c5f97a87ed95e7748c8189e1a110d1a77`.
- Phase 11A adapter checkpoint SHA256:
  `fecb7371033bb2dc5f82d865890f182fb41991c43104b3b302533d0f8dcab08f`.
- Phase 11A lock SHA256:
  `131138fa13701543d17c71ba2153ffd062ce798c8149f9ced7dbc61c95821178`.
- Phase 11A paired report SHA256:
  `ae078b126a3919936bc3a5b99c7f5cbaa7fa85d16b29012b9d6ca84f79064d1f`.

## Frozen dataset and protocol

- Input root:
  `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/pdbind_protein_only`.
- Views: exact existing protein-only train/valid/test block views.
- Batch protocol: padded-cost budget 4,000,000; maximum 4,096 atoms; maximum 8
  items; no silent truncation.
- Seeds: train/validation `20260810`; formal paired evaluation seeds
  `20260810`, `20260811`, and `20260812`.
- Training: maximum 5 epochs, minimum 3, patience 2, projector/gate LR
  1e-3, PVB source frozen, gradient clipping 1.0.
- Selection: minimum valid `rec_total = rec_vel + rec_drf`, batch mean. Test is
  not used during training or checkpoint selection.
- Source role: `pvb_full`, expected coverage 195/195. Every variant has 17,185
  trainable adapter/gate parameters in 7 tensors and one optimizer group.

## Registered variants

- `real`: current Phase 11A shared-PVB H-block.
- `shuffled`: deterministic permutation of pooled blocks within each sample.
- `constant`: one fixed nonzero vector shared by all blocks and records.
- `atom_no_pool`: same adapter applied directly to detached atom features.

All variants preserve PVB `x_rep`, PVB log variance, and PVB KL. Anew
`Wx_log_var`, block KL, coordinate residuals, and fragment semantics are
excluded.

## Phase 12 live data boundary

At the audit, `/data4/PVB/pdb` contained 46,119 files and about 9.3 GB,
against the remote estimate of 242,490 regular files and 53.6 GB. No full or
half PVB materialization was ready. `/data4` had about 410 GB free of 3.6 TB.
The partial tree is asynchronous and is not an input to Phase 13.

## Commands and tests

Commands were run after `enter-container` and `conda activate torch-ito`:

- target/source status and SHA capture;
- checkpoint SHA256 capture;
- `python -m py_compile` on modified model, adapter, runner, and tests;
- focused Phase 13 adapter tests and Phase 11A gate-zero tests;
- four-variant `pvb_full` coverage/freeze/optimizer audit.

Results:

- focused tests: 7/7 passed;
- all four variants: PVB coverage 195/195, source-frozen trainable count
  17,185, exact projector/gate optimizer membership;
- Phase 11A gate-zero full-objective and inference parity: passed;
- no source repository modification detected.

New artifacts are isolated under
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13`.

