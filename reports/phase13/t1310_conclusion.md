# Phase 13 four-control tranche conclusion

Date: 2026-08-14

## Scope

This closes the Phase 13 PVB-matched information/capacity/pooling tranche. It
does not close the deferred Phase 11B semantic fragment experiment, and it
does not claim the unrun T1304 Anew feature-source or T1305 injection-location
comparisons.

## Final evidence

- PVB source: `c08e5e3cd49d45c6d748387e78224843bd356f50`, clean.
- AnewOmni source: `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`, clean.
- Full target suite: `59` tests passed in `17.805 s`.
- `python -m compileall -q .`: passed.
- `python train.py --help`: passed.
- `python infer_prot.py --help`: passed.
- `git diff --check`: passed.
- Phase 9 PVB checkpoint SHA256 remains
  `4f0ad88356c7159cd5d0b9641b6c1e5c5f97a87ed95e7748c8189e1a110d1a77`.
- Paired Phase 13 JSON SHA256 is
  `1353962086cbfbb512ea29d53fed89c66429f73c813eeba5ed2d906968ccda53`.

## Conclusion

The four controls have equal source-frozen trainable capacity (17,185 adapter/
gate parameters). Every adapter control improved PVB off, but real shared
H-block was only about `0.00065–0.00070` better than shuffled/constant on both
valid and test. The dominant common gain is therefore compatible with
adapter/injection capacity; the small real-versus-control gap supports a
record-specific H-block contribution. Atom-no-pool was better than pooled real
by about `0.00203` valid and `0.00195` test, so the current pooling path is not
established as optimal. All variants preserved PVB KL, so Phase 9's KL failure
is not present here.

No result supports making H-block Gaussian. The next stochastic experiment,
if approved, must add a separate block latent and KL schedule without changing
the PVB posterior. Before that, a new paired budget should compare Anew versus
PVB feature sources and injection locations with the evaluator extended to emit
per-record vectors before its one-time test traversal.

## Data boundary

T1201 chemistry is complete for the current audit: the 508 records have
readable SDF bond information but fail Anew tokenizer mapping, and five damaged
SDF repair candidates were not promoted. T1111/Phase 11B remains deferred.

The latest PDB snapshot was about 50,189 files / 11G under `/data4/PVB/pdb`,
versus the remote estimate of 242,490 files / 53.6GB. The shared 3.6T
filesystem had 409G free (89% used). This is enough for bounded raw download,
but full EPT/PVB materialization must recheck space and avoid duplicate full
archives. No Phase 13 result used the partial PDB tree.

## Next phase proposal

1. Keep Phase 11B deferred until fragment coverage/mapping is resolved.
2. Register a new paired experiment for Anew-vs-PVB feature source and
   post-merge-vs-cross-attention injection, with valid-only locking and
   per-record output captured during the single test pass.
3. Only if the deterministic record-specific effect survives those controls,
   consider a separately gated stochastic block latent with KL warmup/free bits.
