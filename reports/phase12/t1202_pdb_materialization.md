# T1202 PDB acquisition and materialization log

## Live status audit — 2026-08-14 Phase 13 handoff

The latest container-side snapshot found 49,398 files and 10,561,005,045
bytes (about 10 GB) under `/data4/PVB/pdb`. The acquisition session remains
external/asynchronous; its silent PTY poll did not report completion, and
newest partial-transfer files are still changing. No full or half PVB
materialization has been declared ready.

The remote estimate remains 242,490 regular files and 53,593,441,967 bytes,
so the current tree is approximately 20.4% complete by file count and 19.7%
by raw bytes. `df -h` reports a 3.6 TB filesystem with about 409 GB free
(89% used). This is currently sufficient for the remaining raw mirror and
planned materialization only if duplicate archives and unbounded temporary
copies are avoided. Recheck free space before materialization.

Phase 13 does not consume this partial tree. It uses the existing
protein-only Phase 9/10/11A views until T1202 completes and passes its
manifest/integrity checks.

Date started: 2026-08-14
Status: PDB mirror in progress.

## Acquisition provenance

- Official template: `https://files.wwpdb.org/pub/pdb/software/rsyncPDB.sh`
- Downloaded template: `/data4/PVB/pdb/rsyncPDB.sh`
- Template SHA256: `984c74abfbfa1de2e987be654e944e578b4e08e52cf966a1be08d80db1255294`
- Remote module: `rsync.wwpdb.org::ftp/data/structures/divided/pdb/`
- Port: `33444`
- Destination: `/data4/PVB/pdb/`
- The template's commented command includes `--delete`; the actual command
  deliberately omits `--delete` and uses `--partial` for resumability.

The initial read-only dry run reported 243,756 files (242,490 regular files,
1,266 directories) and total source size 53,593,441,967 bytes. The first
single-connection transfer was stopped after its low observed throughput and
left partial files intact. It was replaced by 12 non-overlapping two-character
prefix lanes using the same official rsync module and resumable options, then
restarted as 36 prefix lanes after a measured throughput check.
## Parallel throughput check

The 36-lane pool was started without `-z` because the remote files are
already gzip-compressed. In a one-minute sample, the tree grew from about
476,201,351 to 647,679,590 bytes (about 171 MB/min, 2.86 MB/s) and from
2,777 to 3,724 regular files. The early residual projection is about 5.2 hours,
so the 36-lane pool is retained and will be re-measured after a longer sample.


## Current validation

The prefix filter was independently dry-run for prefix `0`: 361 files,
324 regular files, 37 directories, and 84,505,477 bytes. No regular files
needed transfer in that prefix at the time of the dry run because they were
already present or partial-resume state was sufficient.

Phase 11 tests continue to use their existing materialized protein-only views;
the downloaded/raw PDB tree is not referenced by Phase 11 runners.
## Format check before full processing

For `pdb260l.ent.gz`, the existing EPT parser produced one chain with 162
residues and EPT data shape `X=(1299, 3)` including its global node. The
original PVB parser, using the same raw file and a temporary NumPy
`np.compat.long` shim, produced one record with 1,285 atoms and 2,614 directed
bond edges. This confirms that the EPT mmap is an intermediate and that the
PVB-compatible output must be materialized separately.


## Pending

After the mirror completes, run the existing scripts under
`/data4/PVB/pdb/ept_release/jiaor17-EPT-35f45d5/scripts/process_data`,
record the resulting EPT index/data hashes, generate reproducible structure-level
train/valid/test manifests, and materialize the requested PVB-compatible full
and half data in fresh output directories. Existing outputs are not deleted.

## Live acquisition refresh — 2026-08-14

The 36-prefix pool was restarted with the same non-overlapping filters after an
interruption; partial files were retained. The current host-side snapshot is
24,019 regular .ent.gz files and approximately 5.02 GB. EPT/PVB materialization

## Live acquisition refresh — 2026-08-14 Phase 13 paired-evaluation handoff

A later container-side snapshot during Phase 13 evaluation found 50,189 files
and approximately 11G under `/data4/PVB/pdb`. The asynchronous external
acquisition process is still active; no full raw mirror, EPT materialization,
or PVB full/half dataset is ready. The remote estimate remains 242,490 regular
files and 53,593,441,967 bytes, so the visible tree is still partial.

The shared filesystem snapshot was:

```text
df -h /data4 /output /workspace
3.6T total, 409G available, 89% used
```

The current free space is sufficient for completing the raw mirror and planned
materialization only with bounded temporary storage and no duplicate full
archives. Recheck space immediately before EPT/PVB conversion. Phase 13 did
not read this tree.

