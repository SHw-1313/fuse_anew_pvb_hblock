# T1201 chemistry-source and `xyz2mol` audit

Date: 2026-08-14
Purpose: determine whether tokenizer fallback is caused by missing chemistry
files, and apply the four-hour feasibility gate before any full conversion.

## Scope correction — 2026-08-14

The original PDBBind timing in this report is superseded. It did not record a
proof that the benchmark input was ligand-only, so it cannot be used to claim
that protein coordinates caused the approximately one-second timings. The
corrected probes below read only the conformer from each PDBBind .sdf; no
corresponding protein .pdb coordinates are loaded. This correction preserves
the old numbers as historical evidence instead of silently deleting them.

## Source inventory

- PCQM4Mv2: `/data4/PVB/pcqm4mv2/raw/pcqm4m-v2-train.sdf` exists (about
  9.7 GB); the source is not missing SDF records.
- ANI-1x: `/data4/PVB/ani1x/raw/ani1x-release.h5` exists (about 5.6 GB), with
  3,114 molecule groups and 4,956,005 conformers. Groups contain atomic
  numbers, coordinates, energies, forces, and charges, but no explicit bond
  order/aromaticity dataset.
- PDBBind: `/data4/PVB/pdbbind/raw/pdbbind_v11_pocket_aligned_fill_missing.zip`
  exists and the extracted tree contains 15,487 SDF records. Five records
  from the blocked Phase 11 audit contain at least one `UNSPECIFIED` bond.

Therefore `tokenizer fallback` cannot be classified as simply "SDF is absent".
The available evidence separates three cases: PCQM has SDF chemistry, ANI has
coordinates without explicit bond labels, and some PDBBind SDF records have
present-but-unspecified bond orders or tokenizer mapping assertions.

## `xyz2mol` installation and benchmark

The configured pip index had no normal `xyz2mol` distribution. The official
source was installed into `torch-ito` from
`https://github.com/jensengroup/xyz2mol` at commit
`9ec591dd01bb4793fc221c526d5f7a19c05d0aca` (reported package version
`0.1.2`). Inputs were integer atomic numbers and float64 Cartesian coordinates;
the initial string-element probe was discarded because it violated the API
contract.

Representative results:

| Source/probe | Sample | Success/failure | Mean time | Projection |
| --- | ---: | ---: | ---: | --- |
| PCQM SDF | 50 molecules, mean 29.96 atoms | 50/0 | 2.745 ms/record | not a blocker at this rate |
| ANI unique groups | 100 groups, one conformer/group | 99/1 | 3.187 ms/record | about 10 s for 3,114 groups |
| ANI all conformers | 4,956,005 conformers | benchmarked from the group rate | 3.187 ms/record | about 4.4 h, before I/O and failures |
| PDBBind SDF (historical, superseded) | bounded sample of 20 | 18 completed, 2 timed out at 5 s | 1.253 s/record | not used for the corrected gate |

An unbounded 100-record PDBBind probe ran for more than ten minutes without a
bounded result and was terminated. The five Phase 11 unknown-bond examples
(`4ie2`, `4q3s`, `3skk`, `3sjt`, `4hxq`) were tested with the available
charge and both charged-fragment settings; none produced a usable inferred bond
graph for the unspecified records.

## Gate result

Full PDBBind `xyz2mol` reconstruction is intentionally **abandoned** under
the four-hour gate. No raw dataset was modified and no guessed bond order was
written. ANI conversion is only computationally plausible once per unique
molecule group, with topology reused across conformers, but the source still
does not provide authoritative bond-order/aromaticity labels and therefore it
does not unblock the Phase 11 semantic gate.

The Phase 11 blocker remains: five train SDF bonds are `UNSPECIFIED` and 508
records trigger the upstream tokenizer fallback/assertion path. Any future
repair must use an authoritative chemistry source or a separately approved
semantic decision.

## Commands and reproducibility

The audit was run after `enter-container` and `conda activate torch-ito` in
the target container. The relevant commands were:

```bash
pip show xyz2mol
pip install --no-input git+https://github.com/jensengroup/xyz2mol.git
python  # bounded PCQM, ANI, PDBBind, and five-record probes
```

The PDBBind benchmark used a per-record five-second timeout; the full
conversion was not launched after the projected runtime exceeded four hours.

## Corrected ligand-only benchmark and method comparison

The five Phase 11 damaged records are ligand-only SDF records with 14–22
atoms. Their SDF atom tables contain an explicit UNSPECIFIED bond, but no
protein atoms are present in the input passed to either inference routine.
Using only the SDF atom numbers and 3D conformer coordinates, xyz2mol gave
the following bounded results:

| Record | Ligand atoms | xyz2mol wall time | Result |
| --- | ---: | ---: | --- |
| 4ie2.sdf | 17 | 0.279 s | returned zero molecules |
| 4q3s.sdf | 21 | 1.578 s | returned zero molecules |
| 3skk.sdf | 16 | 0.063 s | returned zero molecules |
| 3sjt.sdf | 14 | 0.061 s | returned zero molecules |
| 4hxq.sdf | 22 | 1.594 s | returned zero molecules |

The corrected damaged-record mean is about 0.715 s per ligand-only case, but
none of the five outputs is usable. A separate 20-record diagnostic on SDF
coordinates only produced 5 calls that returned and 15 three-second bounded
timeouts. This shows that the slow behavior can occur for ligand-only
coordinate inference; it is not evidence that protein coordinates were
included. It also confirms that coordinate inference must not be applied to
the 15,487 valid SDF records.

For ANI-1x, the source contains 3,114 unique molecular groups and 4,956,005
conformers. A 100-group xyz2mol probe remains approximately 3.187 ms per
unique group (99 non-empty returns and one failure in the earlier probe), so
the correct processing unit is the unique group, with its inferred topology
reused for all conformers. Running inference independently for all conformers
would be unnecessary and projects to roughly 4.4 hours before I/O.

RDKit provides rdDetermineBonds, whose official documentation identifies it
as a C++ implementation of the xyz2mol algorithm. In a small 10-group ANI
probe it returned graphs in roughly 0.04 ms per call, but this is a speed
alternative rather than an independent chemistry oracle. It must therefore be
validated against known SDF graphs and checked for valence, charge, aromaticity,
and atom-order consistency. The five malformed PDBBind records failed fast or
returned no usable graph under the direct RDKit/coordinate-only probe because
of their incomplete valence/charge state. Open Babel's ConnectTheDots() plus
PerceiveBondOrders() is a second heuristic candidate, but it was not accepted
as authoritative or used to modify raw data.

## Corrected gate decision

- Complete, readable SDF bond orders and aromaticity are authoritative and are
  used directly; no xyz2mol pass is planned for PCQM4Mv2 or valid PDBBind SDFs.
- Only damaged/unspecified SDF ligand records may enter a bounded repair path,
  and the repair input must contain ligand atoms and coordinates only.
- ANI-1x is a separate coordinate-only workstream. Benchmark RDKit and xyz2mol
  on unique groups, reuse accepted topology across conformers, and stop if
  validation or the four-hour gate fails.
- The five damaged PDBBind records remain unresolved. No inferred bonds were
  written, and T1111 remains blocked pending an authoritative repair or an
  explicitly approved fallback policy.

## Follow-up repair probe — five damaged SDFs and the 508 fallback records

Date: 2026-08-14

Five ligand-only repair candidates were generated by changing exactly one
`0`/`UNSPECIFIED` bond order to `SINGLE` in each of `3sjt`, `3skk`, `4q3s`,
`4ie2`, and `4hxq`; all coordinates, atom order, charges, radical records,
and other bonds were preserved. `xyz2mol` and RDKit `rdDetermineBonds` gave no
validated graph for these five. Open Babel preserved the known graph only for
`4hxq`, but it still failed RDKit valence validation. The candidates pass Anew
tokenization only with `sanitize=False`; normal RDKit sanitization fails on
four-coordinate boron, so none was promoted to the semantic dataset or raw SDF.

### What the 508 records are

All 508 records are from the half PDBBind view: train 480, valid 15, test 13.
They contain 15,772 ligand atoms and 17,178 ligand bonds in total. The source
is `/data/pvb_cross_dataset_20260810/blocks/pdbbind` and the raw SDF paths are
from `manifests/pdbbind_half.csv`; the five unspecified-bond records are
disjoint from this set.

These 508 SDFs are readable, have complete explicit bond types (zero
`UNSPECIFIED` bonds), and have exact PVB/SDF heavy-atom ordering. The failure
is the Anew tokenizer `get_submol_atom_map()` assertion, not missing SDF
chemistry. The existing atom-level fallback is a semantic compatibility
fallback, not a bond-repair result.

The all-508 bounded probe found: RDKit returned/sanitized 84 records, with 68
tokenizer passes and zero exact raw-graph preservations; Open Babel sanitized
436, with 50 tokenizer passes and only one exact graph (`5c28_holo`), whose
tokenizer still failed. The other 49 Open Babel passes changed the source
graph, so neither tool safely repairs this population. `xyz2mol` is not run on
complete authoritative SDF graphs.

Artifacts: `fallback_method_probe.json` SHA256
`483c027676c9739a5cb5da6b7e9db02973fb3b6b586163c0735c0ab3e7bb1b51`;
`five_single_repair_summary.json` SHA256
`97fb45e5fab0b5e4fe7ebcf02b71d16892f4886edd321a28a845f324710c869d`.
