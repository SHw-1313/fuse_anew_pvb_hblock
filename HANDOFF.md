# Handoff

## Current state

- Current phase: Phase 13 — H-block information, capacity, and injection ablations
- Current task: none — Phase 13 four-control tranche is closed
- Status: `DONE` for T1300-T1310 tranche; T1304/T1305 remain explicit TODO extensions; Phase 11B is deferred; T1202 PDB acquisition continues asynchronously; T1111 remains blocked
- Last updated: 2026-08-14 (final Phase 13 audit completed)
- Agent/model: Codex / GPT-5
- Container: `sihao-dev`, entered with `enter-container`
- Conda environment: torch-ito

## Repository locations

- Read-only PVB: `/workspace/PVB`
- Read-only AnewOmni: `/workspace/AnewOmni`
- Writable target: `/workspace/fuse_anew_pvb_hblock`

## Source revisions

| Source | Commit SHA | Dirty before work? | Modified by this task? |
|---|---|---:|---:|
| PVB | `c08e5e3cd49d45c6d748387e78224843bd356f50` | No | Must remain no |
| AnewOmni | `926e99818ea18cf9d9b2064ce0319fe691b7a1f1` | No | Must remain no |

## Target revision

- Branch: `phase11/pvb-shared-hblock`
- Commit: Phase 10 parent `fbf8302d57942dbc41a52e0e1019ecb8c0287687`; working tree has Phase 11 changes
- Working tree status: dirty with Phase 10 code, diagnostics, documentation, and checkpoint-audit changes; Phase 9 artifacts remain under `/output` and must not be overwritten

## Completed tasks

- T000 — verified container entry and activated `torch-ito`.
- T001 — recorded Python, PyTorch, CUDA, GPU, torch-scatter, and xFormers status.
- T002 — recorded clean source revisions.
- T003 — confirmed source repositories are untouched.
- T004 — bootstrapped target from PVB with runtime-artifact exclusions.
- T005 — initialized an independent target Git repository.
- T006 — created the four live planning documents.
- T007 — passed the unchanged PVB import/CLI smoke suite.
- T008 — added the PVB train-step profiling script.
- T009 — added the deterministic one-batch overfit script.
- T010 — recorded a full-dimension 64-atom baseline profile.
- T100–T112 — vendored the minimal Anew implementation and passed source parity.
- T200–T208 — added explicit block metadata, block-safe cropping, and protein-only rejection.
- T300–T311 — added the faithful Anew H-block encoder and passed pooling/SE(3)/gradient tests.
- T400–T409 — added zero-gated decoder conditioning and passed parity/gradient/overfit gates.
- T500–T506 — added explicit PVB/Anew/resume checkpoint migration, coverage reports, and trainer state saving.
- T600–T609 — added staged freezing, optimizer groups, BF16 option, n*n budgeting, and four-scale profiles.
- T700–T706 — passed the focused target suite, protein train/inference smoke tests, source/dependency audits, and final changed-file/benchmark review.
- T800 — upgraded profiling with CUDA-event warmup, repeated measurements, component timings, and memory statistics.
- T801 — completed the six-scale synthetic `off`/`anew_block` CUDA matrix; no 2000-atom discontinuity was observed.
- T802 — profiled 100 real PDBBind records per split; found legacy metadata, mixed protein/ligand blocks, and a zero-group default budget.
- T803 — audited legacy checkpoint coverage, added source-key tracking/freeze stages and a durable protein-only real-batch path comparison.
- T804 — captured CUDA operator traces on real protein-only representatives near 1024/2000 atoms.
- T805 — attributed the dominant scaling to dense EPT attention, block-candidate construction, and real-batch graph/decoder/backward costs.
- T806 — completed the stale-length/max-padding audit; Phase 8 gate passed without changing faithful EPT.
- T900–T901 — downloaded/provenanced the official Anew release and selected the prior shape-matched fused state dict for the requested run.
- T902–T903 — materialized exact-length protein-only PDBBind views and passed full metadata, bond, ID, and batch-isolation validation.
- T904–T908 — generated per-key checkpoint provenance, implemented and audited source-frozen optimizer membership, passed all 28 target tests, and passed the real fixed-batch overfit/checksum smoke.
- T909 — trained one complete source-frozen epoch on all `6413` train records, selected by complete valid loss, and preserved all source checksums.
- T910 — evaluated the existing PVB checkpoint on complete original valid/test splits only.
- T911–T912 — evaluated PVB `off` and fused H-block on paired protein-only valid/test views and recorded three-seed batch/atom-weighted metrics.
- T913 — verified bitwise source-key preservation and exact non-source optimizer membership after training.
- T914 — updated the live plan/handoff/decision records; Phase 9 gate passed.
- T1000 — initialized Phase 10 documentation/provenance; target/source SHAs and Phase 9 artifact hashes were verified, and no code or Phase 9 artifact changed.
- T1001 — reproduced the Phase 9 loss formula and KL-dominated posterior provenance; the Phase 10 audit artifact was written without changing Phase 9 artifacts.
- T1002 — added and audited the complete PVB checkpoint role; real coverage is 195/195 and the coverage artifact is protected under Phase 10 output.
- T1003 — added the explicit `anew_block_pvb_posterior` mode/configuration path; existing `off` and legacy `anew_block` semantics remain unchanged and focused tests passed.
- T1004 — routed corrected training/inference/realization through PVB posterior plus Anew H-block conditioning; Anew variance/coordinates remain diagnostic-only.
- T1005 — passed complete gate-zero parity and confirmed legacy Phase 9 role loading.
- T1006 — audited corrected `pvb_full`/Anew source coverage, bitwise checksums, exact complement, and optimizer membership.
- T1007 — added reusable posterior/conditioning diagnostics; fixed-batch results show PVB-like KL and no Anew variance gradient in the corrected loss.
- T1008 — passed the full target suite, compile/CLI checks, all three protein smoke modes, and source cleanliness; fixed the stale smoke-mode choice list.
- T1009 — passed the corrected fixed-real-batch gate with exact source checksums, optimizer complement, decreasing reconstruction, PVB-like KL, and no dropped atoms.
- T1010 — completed four complete validation epochs on the exact train/valid views; the best valid reconstruction checkpoint was produced before the long-running session stopped during the next epoch.
- T1011 — locked the epoch-3 corrected checkpoint using valid rec_total only; test remained unevaluated.
- T1012 — completed all six paired valid/test traversals with identical counts; recovered the report from the completed evaluator log after a final JSON serialization defect, without rerunning test.
- T1013 — passed the final target/source/artifact/runtime audits; Phase 9 hashes remain unchanged and the final audit is recorded.
- T1014 — closed Phase 10 documentation, marked Gate P10 PASSED, and recorded the bounded paired-performance interpretation plus the unimplemented Phase 11 proposal.
- T1300 — froze the Phase 13 protein-only ablation protocol, registry scope, seeds, valid-only selection rule, and protected artifact boundary; focused gates passed before training.
- T1301-T1307 — added the explicit registry and four matched PVB controls; passed legacy/gate-zero, source-freezing, optimizer, capacity, and fixed-real-batch gates.
- T1308 — completed four identical valid-only runs: real `0.218319582`, shuffled `0.218891819`, constant `0.218855551`, atom-no-pool `0.216537594`; all used 17,185 trainable parameters, preserved source checksums, and left test untouched.
- T1309 — completed the single paired valid/test aggregate evaluation; all five models had identical traversal counts and PVB-like KL. Real beat shuffled/constant by about `0.00065–0.00070` `rec_total`, all controls beat off, and atom-no-pool was best; no second test traversal was performed for per-record bootstrap.

## In-progress task

- T1310 — close the four-control Phase 13 tranche and record the next ablation phase; this is the sole `IN_PROGRESS` task.
- Intended outcome: preserve the paired result and state the information-versus-capacity/pooling conclusion honestly, while leaving T1304/T1305 as explicit unrun extensions.
- Source files being reused: `reports/phase13/t1309_paired_eval.md`, `scripts/phase13_paired_eval.py`, exact protein-only Phase 9 views, and the Phase 13 registry.
- Target files: `PLAN.md`, `TASKS.md`, `DECISIONS.md`, `HANDOFF.md`, and the Phase 13 conclusion/audit report.
- Current evidence: paired artifact SHA256 `1353962086cbfbb512ea29d53fed89c66429f73c813eeba5ed2d906968ccda53`; valid `367/847,978/354`, test `167/334,142/155`, identical for every model; PDB snapshot `50,189` files/11G and 409G free, still incomplete.
- Remaining work: run final target/source/artifact audits, record Phase 12 status and exact limitations, then decide whether the Anew feature/injection comparisons become a separate phase.

## Commands already run

```bash
# Host/workspace audit
pwd
ls -la
command -v enter-container

# In sihao-dev, with torch-ito active
conda env list
conda activate torch-ito
python --version
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
python -c 'import importlib.util; print(importlib.util.find_spec("torch_scatter")); print(importlib.util.find_spec("xformers"))'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Source provenance
git -C /workspace/PVB rev-parse HEAD
git -C /workspace/PVB status --porcelain=v1
git -C /workspace/AnewOmni rev-parse HEAD
git -C /workspace/AnewOmni status --porcelain=v1

# Target bootstrap
rsync -a --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.cache/' --exclude='checkpoints/' --exclude='checkpoint/' --exclude='datasets/' --exclude='logs/' --exclude='results/' --exclude='outputs/' --exclude='wandb/' --exclude='ckpt/' --exclude='.venv/' --exclude='venv/' /workspace/PVB/ /workspace/fuse_anew_pvb_hblock/
git -C /workspace/fuse_anew_pvb_hblock init -b main

# Phase 1–4 validation
python -m unittest -v tests.test_anew_vendor_parity tests.test_block_metadata tests.test_anew_block_encoder tests.test_fusion
python -m compileall -q module data tests third_party train.py
python -m scripts.overfit_one_batch --device cuda --atoms 16 --samples 2 --steps 5 --hidden-dim 32 --ffn-dim 64 --rbf-dim 8 --heads 4 --layers 2 --k-neighbors 8
python -m scripts.overfit_one_batch --device cuda --atoms 16 --samples 2 --steps 5 --hidden-dim 32 --ffn-dim 64 --rbf-dim 8 --heads 4 --layers 2 --k-neighbors 8 --fusion-mode anew_block
python -m unittest -v tests.test_checkpoints tests.test_training_stages
python -m scripts.profile_components --device cuda --atoms 256 --samples 1 --steps 2
python -m scripts.profile_components --device cuda --atoms 256 --samples 1 --steps 2 --fusion-mode anew_block
python -m unittest discover -s tests -v
python -m compileall -q .
python -m scripts.protein_smoke --device cuda --fusion-mode off
python -m scripts.protein_smoke --device cuda --fusion-mode anew_block
python train.py --help
python infer_prot.py --help

# Phase 8 T800
python -m py_compile scripts/profile_components.py
python -m scripts.profile_components --device cpu --atoms 8 --samples 1 --steps 2 --warmup-steps 1 --hidden-dim 8 --ffn-dim 16 --rbf-dim 4 --heads 2 --layers 1 --k-neighbors 4
python -m scripts.profile_components --device cpu --atoms 8 --samples 1 --steps 1 --warmup-steps 1 --hidden-dim 8 --ffn-dim 16 --rbf-dim 4 --heads 2 --layers 1 --k-neighbors 4 --fusion-mode anew_block --anew-hidden-dim 8 --anew-ffn-dim 8 --anew-edge-size 4 --anew-rbf-dim 4 --anew-layers 1 --anew-heads 2 --anew-k-neighbors 2
python -m scripts.profile_components --device cuda --atoms 16 --samples 1 --steps 2 --warmup-steps 1 --hidden-dim 16 --ffn-dim 32 --rbf-dim 4 --heads 4 --layers 1 --k-neighbors 8
python -m scripts.profile_components --device cuda --atoms 16 --samples 1 --steps 2 --warmup-steps 1 --hidden-dim 16 --ffn-dim 32 --rbf-dim 4 --heads 4 --layers 1 --k-neighbors 8 --fusion-mode anew_block --anew-hidden-dim 16 --anew-ffn-dim 16 --anew-edge-size 8 --anew-rbf-dim 4 --anew-layers 1 --anew-heads 4 --anew-k-neighbors 2

# Phase 8 T801 synthetic matrix (all commands used cuda:5, warmup 10, measured 20)
python -m scripts.profile_components --device cuda:5 --atoms {512,1024,1536,1800,2000,2048} --samples 1 --steps 20 --warmup-steps 10
python -m scripts.profile_components --device cuda:5 --atoms {512,1024,1536,1800,2000,2048} --samples 1 --steps 20 --warmup-steps 10 --fusion-mode anew_block
python -m unittest discover -s tests -v

# Phase 8 T802 real PDBBind batch profile
python -m py_compile scripts/profile_real_batches.py
python -m scripts.profile_real_batches --dataset-root /data/pvb_cross_dataset_20260810/blocks --dataset pdbbind --max-records 100 --budgets 2000 4000000 8000000 --max-groups 8 > /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/real_batches/pdbbind_profile.json
python -m py_compile data/protein_view.py scripts/profile_training_paths.py train.py trainer/abs_trainer.py utils/fusion_training.py
CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_training_paths --dataset-root /data/pvb_cross_dataset_20260810/blocks/pdbbind --split train --record-index 0 --device cuda:0 --pvb-checkpoint /tmp/performance_v1_pvb_state.pt --anew-checkpoint /tmp/performance_v1_fuse_state.pt --warmup-steps 1 --steps 1 > /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/train_record0_paths.json
python -m unittest tests.test_checkpoints tests.test_training_stages -v
# Phase 8 T804/T805 operator traces (cuda:5 mapped to cuda:0)
python -m py_compile scripts/profile_operator.py
CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_operator --dataset-root /data/pvb_cross_dataset_20260810/blocks/pdbbind --split train --record-index 24 --fusion-mode anew_block --device cuda:0 --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt --anew-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt --warmup-steps 1 --row-limit 500 --output-dir /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/operator
CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_operator --dataset-root /data/pvb_cross_dataset_20260810/blocks/pdbbind --split train --record-index 0 --fusion-mode anew_block --device cuda:0 --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt --anew-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt --warmup-steps 1 --row-limit 500 --output-dir /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/operator
# Phase 9 T900/T901 checkpoint provenance (container workdir `/workspace/AnewOmni` for the serialized upstream object)
curl -L --fail --retry 3 --connect-timeout 15 --max-time 1800 -o /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_model.ckpt https://github.com/bytedance/AnewOmni/releases/download/init/model.ckpt
stat -c '%s %n' /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_model.ckpt
sha256sum /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_model.ckpt
cd /workspace/AnewOmni; python -c 'import torch; p=torch.load("/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_model.ckpt", map_location="cpu", weights_only=False); print(type(p).__module__, type(p).__name__, len(p.state_dict()))'
cd /workspace/fuse_anew_pvb_hblock; sha256sum /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/anew_official_encoder_state_dict.pt
# Phase 9 T902/T903 exact protein-only materialization and validation
PYTHONPATH=. python scripts/materialize_protein_view.py --manifest /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/materialization_manifest.json
PYTHONPATH=. python scripts/profile_real_batches.py --dataset-root /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data --dataset pdbind_protein_only --max-records 0 --budgets 4000000 8000000 --max-groups 8 > /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/materialized_profile.json
PYTHONPATH=. python -c 'load materialized train/valid/test with UniDataset; collate first/middle/last records; assert block_batch[atom_block_id] == abid and bond endpoints stay in one sample' > /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/materialized_validation.json
# Phase 9 T904-T908 provenance and source-frozen smoke
PYTHONPATH=. python -m scripts.audit_fused_provenance --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt --anew-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt
CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_training_paths --dataset-root /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/pdbind_protein_only --split train --record-index 0 --device cuda:0 --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt --anew-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt --warmup-steps 1 --steps 2 > /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/source_frozen_smoke.json
CUDA_VISIBLE_DEVICES=5 python -m scripts.source_frozen_overfit --dataset-root /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/pdbind_protein_only --split train --record-index 0 --device cuda:0 --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt --anew-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt --steps 20 --pvb-lr 1e-3 --projector-lr 1e-3 > /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/profiles/source_frozen_overfit_lr1e3.json
```

# Phase 10 T1001
python -m py_compile scripts/audit_phase9_loss.py
python -m scripts.audit_phase9_loss --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase9_loss_audit.json
sha256sum /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase9_loss_audit.json
# Phase 10 T1002
python -m py_compile utils/checkpoint.py tests/test_checkpoints.py scripts/profile_training_paths.py train.py scripts/audit_pvb_full_checkpoint.py
python -m unittest -v tests.test_checkpoints
python -m scripts.audit_pvb_full_checkpoint --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json
sha256sum /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json
# Phase 10 T1003
python -m py_compile module/model.py train.py scripts/profile_training_paths.py tests/test_fusion.py
python -m unittest -v tests.test_fusion tests.test_checkpoints
# Phase 10 T1004
python -m py_compile module/model.py utils/fusion_training.py tests/test_fusion.py
python -m unittest -v tests.test_fusion
# corrected inference/realization finite smoke with sde_step=2
# Phase 10 T1005
python -m py_compile tests/test_fusion.py
python -m unittest -v tests.test_fusion
# Phase 10 T1006
python -m py_compile scripts/audit_fused_provenance.py tests/test_training_stages.py
python -m unittest -v tests.test_training_stages
python -m scripts.audit_fused_provenance --fusion-mode anew_block_pvb_posterior --pvb-role pvb_full --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt --anew-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/legacy_fused_state_dict.pt --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/source_frozen_provenance.json
sha256sum /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/source_frozen_provenance.json
# Phase 10 T1010/T1011
python -m scripts.phase10_training_report --log /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/train_corrected.log --checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/anew_block_pvb_posterior_best.ckpt --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_train_interrupted.json
python -m scripts.phase10_lock_checkpoint --checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/anew_block_pvb_posterior_best.ckpt --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/phase10_best.lock.json
python -m scripts.phase10_paired_eval --device cuda:0 --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_paired_valid_test.json --eval-seeds 20260810 20260811 20260812
python -m scripts.recover_phase10_paired_eval --log /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/phase10_paired_eval.log --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase10_paired_valid_test.json

```

## Validation results

| Test or benchmark | Command | Result | Notes |
| --- | --- | --- | --- |
| PVB baseline | `python -m compileall -q .`; import/CLI smoke | Passed | Unmodified copied PVB tree |
| Vendor parity | `python -m unittest -v tests.test_anew_vendor_parity` | Passed | 2 source parity tests |
| Block metadata | `python -m unittest -v tests.test_block_metadata` | Passed | 4 metadata/cropping tests |
| Gate-zero parity | `python -m unittest -v tests.test_fusion` | Passed | decoder parity within `1e-6` |
| One-batch overfit | `python -m scripts.overfit_one_batch ...` | Passed | off `29.2191 → 3.1783`; fused `11.6374 → 0.8612` |
| SE(3) | `python -m unittest -v tests.test_anew_block_encoder` | Passed | atom state/coordinate equivariance |
| Checkpoint migration | `python -m unittest -v tests.test_checkpoints` | Passed | PVB/Anew/resume/legacy namespace/coverage fixtures |
| Staged training | `python -m unittest -v tests.test_training_stages` | Passed | adapter, source_frozen key union, Stage A/B, groups, diagnostics, n*n budget |
| BF16 smoke | CUDA autocast forward/backward probe | Passed | finite fused loss and gradients |
| Training smoke | `python -m scripts.protein_smoke --device cuda --fusion-mode off`; `... --fusion-mode anew_block` | Passed | Synthetic protein train step; losses `13.7539` and `4.1669`, finite gradients |
| Inference smoke | Same protein smoke commands | Passed | Both modes returned finite `[16, 3]` generated coordinates |
| Focused target suite | `python -m unittest discover -s tests -v` | Passed | 28 tests in 15.751 s |
| Compile/import/CLI | `python -m compileall -q .`; `python train.py --help`; `python infer_prot.py --help` | Passed | Target compiles and both entrypoints import |
| Profiler protocol T800 | CPU/CUDA smoke with warmup and event timing | Passed | Finite off/fused runs; raw measurements and p50/p90/std/CV emitted |
| Synthetic scaling T801 | 12 CUDA profiles; 10 warmup + 20 measured steps | Passed | All finite; no OOM; no 2000-atom discontinuity |
| Real batch profile T802 | `python -m scripts.profile_real_batches ...` | Passed | 100 PDBBind records per split; legacy metadata, protein/ligand block counts, bonds, padding, and dynamic budgets recorded |
| Protein-only view T803 | py_compile and real train record 0 check | Passed | 2147 atoms, 282 complete residue blocks, 4368 remapped bonds; no unsupported block IDs |
| Checkpoint coverage T803 | role-scoped loader audit | Passed | PVB 150/150, Anew 68/68, shape mismatches 0; unexpected legacy keys reported |
| Execution paths T803 | scripts/profile_training_paths | Passed | all-trainable/adapter/forward-only finite; adapter gradients finite |
| Operator profiler T804 | `python -m scripts.profile_operator ... --record-index 24/0 --row-limit 500` | Passed | 911/2147 atom real protein-only traces; nested KNN/EPT scopes and Chrome traces |
| Root-cause attribution T805 | JSON event extraction and source inspection | Passed | dense `[1,4,N,N]` attention, block-KNN, PVB graph/decoder, backward, and memory evidence |
| Batch-budget/padding audit T806 | `python -m scripts.profile_real_batches ...` plus sampled `get_len`/atom-count comparison | Passed | stale length properties, max-N padding amplification, no faithful EPT rewrite; Phase 8 gate passed |
| Official checkpoint T900 | download, `stat`, `sha256sum`, source-object audit | Passed | 722,534,016 bytes; SHA256 `961599ac...a74c1`; 786 state keys; official 512/64/6/8 encoder |
| Architecture/coverage T901 | official role audit plus prior fused state audit | Passed with explicit selection | official 1/68 and 67 shape mismatches; prior fused PVB 150/150 and Anew 68/68 |
| Materialized protein-only view T902 | materialize_protein_view.py | Passed | 6413/367/167 records; source IDs preserved; exact get_len verified; complete protein blocks |
| Materialized integrity T903 | materialized_validation.json and target unit suite | Passed | explicit metadata, valid remapped bonds, and batch isolation passed for all splits |
| Exact-length batch profile T903 | profile_real_batches.py on materialized root | Passed | 4e6/8e6 candidate groups train/valid/test = 1710/85/63 and 3398/207/61 |
| Provenance T904 | scripts/audit_fused_provenance.py | Passed | PVB 150/150, Anew 68/68; union 218 state keys; all assertions true |
| Source-frozen T905/T906 | profile_training_paths.py source-frozen mode | Passed | 50 complement tensors / 1,858,692 params; optimizer ID set exact; 45 inactive legacy PVB tensors explicitly reported |
| Frozen tests T907 | timeout 120 python -m unittest discover -s tests -v | Passed | 28 tests; finite gradients, exact-complement, and empty-bond tests |
| Source-frozen overfit T908 | scripts/source_frozen_overfit.py --steps 20 | Passed | fixed-batch loss 1.0472314 → 1.0452697; source checksums unchanged |
| Formal source-frozen train T909 | scripts/phase9_train_eval.py --mode train_fused | Passed | 6413/6413 records, 6203 batches, 14,666,461 atoms, 3929 oversized singletons; valid 367/367 |
| Original PVB evaluation T910 | scripts/phase9_train_eval.py --mode eval_pvb | Passed | complete original valid/test: pcqm4mv2, ani1x, pdbbind; no truncation |
| Paired evaluation T911/T912 | scripts/phase9_train_eval.py --mode eval_pvb_protein/eval_fused | Passed | same protein-only PDBBind valid/test counts; three-seed loss/KL/velocity/drift reports |
| Source freeze audit T913 | checkpoint payload and source checksum comparison | Passed | 218 source checksums equal; exact 50-tensor complement optimizer |
| Phase 9 gate T914 | `git diff --check`, source status, artifact SHA256 audit | Passed | T909–T914 complete; PVB/Anew source worktrees clean |
| Phase 10 documentation T1000 | target/source status, SHA audit, complete document read | Passed | only live docs changed; Phase 9 artifacts preserved |
| Phase 10 loss audit T1001 | `python -m scripts.audit_phase9_loss ...` | Passed | formula error `1.24e-8`; legacy/PVB KL ratio `157.18x` batch and `157.97x` atom; PVB posterior keys absent from legacy role |
| Phase 10 full-PVB coverage T1002 | `python -m scripts.audit_pvb_full_checkpoint ...` | Passed | `195/195`; zero missing/unexpected/shape mismatches; encoder 42; posterior 3 |
| Phase 10 corrected mode T1003 | `python -m unittest -v tests.test_fusion tests.test_checkpoints` | Passed | 11 focused tests; corrected mode constructs with zero gate; legacy mode/checkpoint fixtures remain green |
| Phase 10 posterior routing T1004 | `python -m unittest -v tests.test_fusion` plus corrected inference/realization smoke | Passed | 5 fusion tests; PVB posterior preserved; Anew variance-head mutation has no loss effect/gradient; finite corrected inference and realization |
| Phase 10 complete parity T1005 | `python -m unittest -v tests.test_fusion` | Passed | 6 tests; full objective/inference parity within `1e-6`; legacy loader `150/150` + `68/68` |
| Phase 10 source freeze T1006 | `python -m scripts.audit_fused_provenance --fusion-mode anew_block_pvb_posterior --pvb-role pvb_full ...` | Passed | `195/195` + `68/68`; 263 bitwise source state matches; exact five-tensor/33,281-parameter optimizer complement |
| Phase 10 diagnostics T1007 | `python -m unittest -v tests.test_phase10_diagnostics`; `python -m scripts.phase10_diagnostics ... --no-update` | Passed | PVB/Anew quantiles and KL separated; PVB KL `0.006921472`; Anew diagnostic KL `1.018065`; no Anew variance gradient; artifact SHA `6b203235...83641ea` |
| Phase 10 focused suite T1008 | `python -m unittest discover -s tests -v`; compile/CLI; three `scripts.protein_smoke` modes | Passed | 36 tests in 44.304 s; finite all-mode smoke; source worktrees clean; unit log SHA `a894bb52...c5d204` |
| Phase 10 fixed-real-batch T1009 | `python -m scripts.source_frozen_overfit ... --pvb-role pvb_full --fusion-mode anew_block_pvb_posterior --steps 20` | Passed | `rec_total 0.308612682 → 0.307434760`; PVB KL `~0.007063`; exact source/optimizer gate; artifact SHA `ed51fda...7acb4e2` |
| Phase 10 train T1010 | phase10_train_eval.py with 5/3/2 protocol | Passed with recorded interruption | Four complete valid epochs; best rec_total=0.2215431160951233 at epoch 3; report SHA 7cd906217...e10b9dc6; no test use |
| Phase 10 lock T1011 | phase10_lock_checkpoint.py | Passed | checkpoint SHA 5ad3b769...e4d204; lock SHA b9d574ad...14e09; valid-only and test_evaluated=false |
| Phase 10 paired evaluation T1012 | phase10_paired_eval.py plus recovery | Passed with recorded writer recovery | Six complete traversals; valid 367/367 and test 167/167 with identical counts; report SHA 965353475...2006c52; test executed once |
| Phase 10 final audit T1013 | python -m scripts.audit_phase10_final ... | Passed | 36 tests; compile/CLI help; source worktrees clean; Phase 9 hashes unchanged; runtime sibling-dependency scan passed; audit SHA e48d251...fdfd050 |
| Phase 10 close T1014 | final live-document review and post-close unit/diff checks | Passed | Gate P10 PASSED; 36 post-close tests; unit-log SHA `7818b714...aa7af`; T1000–T1014 complete; Phase 11 official-Anew alignment proposed only |

## Performance results

### Phase 10 T1001 audit

The audit artifact is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/profiles/phase9_loss_audit.json`
with SHA256
`6ef70eed26b5cb4c3144047f657c4d664a77f67b715b07f3bf9a407b984a6273`.
It records the exact loss recomputation, posterior provenance, source-frozen
key union/complement, Anew variance-key status, and source revisions. The
Phase 9 diagnosis is confirmed: reconstruction improves, but the legacy
Anew-derived posterior causes the total-loss increase.

### Phase 10 T1002 checkpoint coverage

The role report is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/checkpoints/pvb_full_coverage.json`,
SHA256
`1b5210793836f7ccc69f38de41a18cab855285673279992c51af85c90b0ff976`.
It records `195/195` expected keys, source-to-target mapping, per-prefix
counts, missing/unexpected/shape mismatch lists, source checkpoint hash, and
PVB/Anew/target revisions. The legacy `pvb` loader was not changed.

Stable post-warmup step from `scripts/profile_components.py` on one A100; graph/encoder/decoder values are forward components.

| Mode | Atoms | Graph | Encoder | Decoder | Forward | Backward | Step | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PVB baseline | 64 | — | — | — | 0.02282 s | 0.02611 s | 0.04893 s | 697,924,608 B |
| PVB baseline | 256 | 0.03172 s | 0.00497 s | 0.01953 s | 0.08040 s | 0.06572 s | 0.14612 s | 2,361,583,616 B |
| PVB baseline | 512 | 0.02972 s | 0.02405 s | 0.04456 s | 0.10814 s | 0.08315 s | 0.19129 s | 4,593,067,008 B |
| PVB baseline | 1024 | 0.02801 s | 0.01021 s | 0.05976 s | 0.12035 s | 0.12455 s | 0.24490 s | 9,034,408,960 B |
| PVB baseline | 2000 | 0.00789 s | 0.01513 s | 0.04912 s | 0.07482 s | 0.09180 s | 0.16661 s | 17,511,942,656 B |
| Anew H-block | 256 | 0.03078 s | 0.10455 s | 0.04235 s | 0.18486 s | 0.07258 s | 0.25744 s | 2,610,635,776 B |
| Anew H-block | 512 | 0.00359 s | 0.03939 s | 0.01917 s | 0.06556 s | 0.07295 s | 0.13851 s | 4,715,552,256 B |
| Anew H-block | 1024 | 0.00272 s | 0.02794 s | 0.02646 s | 0.05959 s | 0.08203 s | 0.14162 s | 8,943,741,440 B |
| Anew H-block | 2000 | 0.00326 s | 0.05010 s | 0.04776 s | 0.10420 s | 0.15587 s | 0.26007 s | 17,345,196,544 B |

### T801 repeated CUDA profiles

All values below are p50 on `cuda:5`; peak memory is allocated memory.

| Mode | Atoms | Encoder | Decoder | Forward | Backward | Step | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PVB `off` | 512 | 0.005533 s | 0.015940 s | 0.029872 s | 0.039006 s | 0.074834 s | 4.278 GiB |
| PVB `off` | 1024 | 0.008206 s | 0.026607 s | 0.044941 s | 0.058416 s | 0.109251 s | 8.414 GiB |
| PVB `off` | 1536 | 0.010892 s | 0.038037 s | 0.059403 s | 0.081497 s | 0.145547 s | 12.533 GiB |
| PVB `off` | 1800 | 0.012641 s | 0.043653 s | 0.068106 s | 0.094398 s | 0.167887 s | 14.664 GiB |
| PVB `off` | 2000 | 0.013642 s | 0.048489 s | 0.073489 s | 0.100427 s | 0.179374 s | 16.309 GiB |
| PVB `off` | 2048 | 0.013772 s | 0.048539 s | 0.072287 s | 0.097331 s | 0.175023 s | 16.664 GiB |
| Anew H-block | 512 | 0.026329 s | 0.016772 s | 0.048533 s | 0.054676 s | 0.110349 s | 4.392 GiB |
| Anew H-block | 1024 | 0.029690 s | 0.026923 s | 0.063967 s | 0.087629 s | 0.157562 s | 8.330 GiB |
| Anew H-block | 1536 | 0.042863 s | 0.038312 s | 0.088212 s | 0.125481 s | 0.221191 s | 12.378 GiB |
| Anew H-block | 1800 | 0.050477 s | 0.043868 s | 0.101666 s | 0.150250 s | 0.259745 s | 14.501 GiB |
| Anew H-block | 2000 | 0.053302 s | 0.048038 s | 0.108027 s | 0.161762 s | 0.279082 s | 16.154 GiB |
| Anew H-block | 2048 | 0.056928 s | 0.050089 s | 0.113748 s | 0.168469 s | 0.289102 s | 16.512 GiB |

### T802 real PDBBind batch shape

The following uses 100 inspected records per split; atom/block/bond values are p50/p99.

| Split | Records | Atoms | Blocks | Bonds | Unsupported atoms p50 | Groups at n*n=2000 | Groups at n*n=4e6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6413 | 2228.5/4894.8 | 305.5/661.3 | 4562/9918.7 | 26 | 0 | 3667 |
| valid | 367 | 2193/5395.2 | 304/713.7 | 4496/11071 | 25 | 0 | 213 |
| test | 167 | 1738/4818.9 | 250/684.2 | 3550/9754.4 | 29 | 0 | 72 |
### T803 real protein-only execution paths

Train record 0 after complete-block protein filtering; times are one warmup plus one measured CUDA step on A100 `cuda:5`.

| Mode | Atoms | Blocks | Step | Peak allocated | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-trainable | 2147 | 282 | 169.3 ms | 14.75 GiB | 10,949,066 |
| strict adapter | 2147 | 282 | 114.1 ms | 9.53 GiB | 33,281 |
| forward-only | 2147 | 282 | 68.3 ms | 1.81 GiB | 0 |


### T804/T805 operator-level real-batch traces

The following are one warmup plus one profiled forward/backward/optimizer step on A100 `cuda:5`; the operator profiler uses a row limit of 500.

| Mode | Atoms | Blocks | Padded N | Anew KNN | Anew EPT | PVB graph | PVB decoder | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anew H-block | 911 | 109 | 912 | 51.9 ms | 35.5 ms | 38.1 ms | 88.2 ms | 6.31 GiB |
| Anew H-block | 2147 | 282 | 2152 | 53.9 ms | 65.9 ms | 39.8 ms | 156.2 ms | 14.75 GiB |

### T902/T903 exact-length materialized batching

The profile uses all records in each materialized split. Padding ratios are from
the representative groups emitted by the profiler, not a replacement for a
full model benchmark.

| Split | Records | Exact atoms p50/max | 4e6 groups/skipped | 8e6 groups/skipped | Representative padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 6413 | 2190/6855 | 1710/3929 | 3398/1299 | 0.46–1.00 |
| valid | 367 | 2217/6443 | 85/239 | 207/72 | 0.55–1.00 |
| test | 167 | 1711/6083 | 63/70 | 61/37 | 0.50–1.00 |

### T904–T908 source-frozen provenance and smoke

| Item | Result |
| --- | --- |
| PVB/Anew matched source keys | `150/150` and `68/68`; union `218` state keys / `217` parameter tensors |
| Source/complement parameters | source `9,090,374`; complement `1,858,692` across `50` tensors |
| Effective gradient-bearing complement | `5` projector/gate tensors; `45` legacy PVB encoder/prior tensors have no forward path in `anew_block` |
| Real fixed batch | `2147` atoms / `282` blocks / `4368` bonds; all finite |
| Source-frozen overfit | loss `1.0472314 → 1.0452697` in 20 fixed-draw steps |
| Frozen checksum / optimizer audit | source unchanged; optimizer exactly equals the non-source complement |

### T909–T914 formal training and evaluation

| Item | Result |
| --- | --- |
| Fused source-frozen train | `6413/6413` records; `6203` batches; `14,666,461` atoms; `3929` oversized singletons |
| Fused valid selection | `367/367` records; `354` batches; best batch loss `1.0267511`; `239` oversized singletons |
| Original PVB valid/test | valid `168929/285072/367`; test `168930/161913/167` for `pcqm4mv2/ani1x/pdbbind` |
| Paired PVB `off` / fused | both valid `367` and test `167` records; batches `354/155` |
| Source freeze | `218` checksums equal before/after; optimizer is exact `50`-tensor complement |
| Selected checkpoint | `source_frozen_epoch1_best.ckpt`, SHA256 `c9df6928268c4c8a5f27779067b83703af1a15d92f187f570bb454baa2441d57` |

Paired three-seed aggregate metrics (mean ± std):

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

### Phase 10 T1010/T1011 training and lock

| Item | Result |
| --- | --- |
| Complete valid epochs | 0, 1, 2, 3; valid records 367/367, 354 batches, 847978 atoms |
| Valid rec_total batch mean | 0.222021 → 0.221780 → 0.222065 → 0.221543 |
| Best checkpoint | epoch 3, step 24797, SHA 5ad3b769...e4d204 |
| Lock manifest | SHA b9d574ad...14e09; test_evaluated=false |
| Source checksum status | unchanged; pvb_full=195/195, Anew 68/68 |
| Training interruption | fifth epoch stopped before validation; retained in the training report |

### Phase 10 T1012 paired evaluation

Batch-mean metrics are loss / KL / rec_vel / rec_drf / rec_total.

| Split | PVB off | Phase 9 legacy | Phase 10 corrected |
| --- | --- | --- | --- |
| valid | 0.273862 / 0.006958 / 0.102848 / 0.165448 / 0.268296 | 1.064952 / 1.093584 / 0.064430 / 0.125655 / 0.190084 | 0.264250 / 0.006958 / 0.099335 / 0.159349 / 0.258684 |
| test | 0.255278 / 0.007027 / 0.098625 / 0.151031 / 0.249657 | 1.043928 / 1.089811 / 0.060861 / 0.111218 / 0.172079 | 0.246216 / 0.007027 / 0.095238 / 0.145356 / 0.240595 |

Atom-weighted metrics are in the recovered report. The corrected mode
preserves PVB KL and improves reconstruction/total loss over PVB off on both
splits. Legacy reconstruction is lower, but its block posterior KL remains
about two orders of magnitude too high. The first evaluator completed all
six traversals but failed during final JSON serialization because resume
metadata contained optimizer tensors; recovery used the emitted aggregate
lines and did not rerun test. Per-seed detail is therefore not present in the
recovered report; fixed seeds and aggregate mean/std are recorded.

## Files changed

* `PLAN.md`
* `TASKS.md`
* `HANDOFF.md`
* `DECISIONS.md`
* PVB baseline files copied into the target from the clean source revision
* `third_party/anewomni/`
* `data/block_metadata.py`
* `data/protein_view.py`
* `module/anew_block_encoder.py`
* `utils/checkpoint.py`
* `utils/fusion_training.py`
* `module/graph.py`
* `train.py` and `trainer/abs_trainer.py`
* `config/train.yaml`
* `tests/test_checkpoints.py`, `tests/test_training_stages.py`, and `tests/test_graph_empty_bonds.py`
* scripts/profile_components.py, scripts/profile_training_paths.py, scripts/profile_operator.py, scripts/profile_real_batches.py, scripts/materialize_protein_view.py, scripts/audit_fused_provenance.py, scripts/source_frozen_overfit.py, and scripts/phase9_train_eval.py
* scripts/audit_pvb_full_checkpoint.py and Phase 10 T1001/T1002 artifacts under /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/
* scripts/audit_phase9_loss.py, scripts/audit_pvb_full_checkpoint.py, scripts/audit_phase10_final.py, scripts/phase10_train_eval.py, scripts/phase10_training_report.py, scripts/phase10_lock_checkpoint.py, scripts/phase10_paired_eval.py, and Phase 10 artifacts under /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10, scripts/recover_phase10_paired_eval.py
* phase9 materialization manifests/profiles/validation and source-frozen smoke artifacts under /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/
* `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase8/operator/` profiler reports/traces (external artifacts)

## Decisions added or changed

* D001–D058; D045–D058 define the KL diagnosis, complete PVB posterior contract, legacy-mode preservation, pvb_full loading, full-objective parity, Anew variance diagnostics, valid-only locking, one exact paired evaluator, the bounded paired result, and the separate official-Anew Phase 11 boundary. T1012 also records the JSON-writer recovery and the no-rerun-test rule.

## Known problems

* xFormers is not installed in `torch-ito`; it is optional and not required for the faithful baseline.
* Root-level `python -m unittest discover -v` also discovers two unrelated optional PVB packages: `ept.models` expects its legacy `utils.register` import layout and `simulation` requires `openmm`. The authoritative target suite is `python -m unittest discover -s tests -v`, which passes all 36 fused-repository tests.
* The copied optional PVB converters/simulation retain their original local `sys.path.append('..')` compatibility lines; no target runtime code imports `/workspace/PVB` or `/workspace/AnewOmni`, and Anew parity tests invoke the source only in isolated subprocesses.
* Repeated synthetic CUDA timing is complete and shows no 2000-atom discontinuity. T804–T806 now attribute the real-batch cost to dense EPT attention, repeated block-candidate construction, stale dynamic-batch lengths, PVB graph/decoder/backward, and shape-dependent memory; the faithful path remains unchanged.
* The official Anew checkpoint is downloaded and audited but is incompatible with the current fused architecture (`1/68` coverage, `67` shape mismatches); it remains provenance-only. The prior shape-matched fused state dict is the selected Anew source for this run.
* The formal one-epoch fused training and complete evaluation have finished. PVB remained evaluation-only on the original valid/test splits; the paired fused result is a diagnostic baseline and is worse than PVB `off` on loss.
* The original PDBBind mmap records remain legacy and untouched; the derived views now carry explicit metadata and exact protein-only lengths.
* `fusion.stage=source_frozen` now records and freezes the union of matched checkpoint keys; the strict `adapter` stage is the projector/gate-only mode used in T803. In the current block path, 45 complement parameters belong to bypassed PVB encoder/prior modules and receive no gradient; this is recorded in every source-frozen smoke report.
* T1001 confirms that the legacy `anew_block` path uses Anew `Wx_log_var` for PVB KL/reparameterization while the legacy PVB role omits the PVB encoder and posterior heads; T1002 must add the explicit full-PVB role before corrected fusion training.
* Formal T909 training and T910–T912 evaluation are complete. The selected fused H-block checkpoint is reproducible and source-frozen, but its paired valid/test loss is substantially worse than PVB `off`; no improvement claim is made.
* The exact-length profile makes 4e6/8e6 the current budget candidates: train/valid/test form 1710/85/63 and 3398/207/61 groups, respectively, with skipped oversized records and tracked max-padded-N amplification.
* Raw PDBBind records include unsupported ligand element blocks; they must not be passed directly to the protein-only Anew encoder.
* Phase 10 training completed four full validation epochs but the fifth epoch ended before validation. The epoch-3 checkpoint is locked from valid reconstruction only; this is an execution interruption, not a test-tuned result.
* The first Phase 10 paired evaluator completed all six traversals but could not serialize tensor-valued resume metadata. The report was recovered from its aggregate log without a second evaluation; the evaluator now filters metadata to JSON-safe fields. The recovered report intentionally lacks per-seed detail.

## Blockers

* None; Phase 10 is complete and no task is active. T1010–T1014 have complete evidence, including the recorded paired-evaluation writer recovery and final audit.

## Exact next action

1. Future Phase 11 objective: design and document exact official-Anew 512-hidden/64-head/six-layer/eight-radial-setting alignment with full checkpoint coverage and parity gates before any code edit; expected result is a separate approved plan, and any coverage or parity failure must be recorded as a blocker.

## Resume instructions

1. Enter the container.
2. Activate `torch-ito`.
3. Read `PLAN.md`.
4. Read `DECISIONS.md`.
5. Read this file.
6. Phase 10 is closed; do not rerun its training or test evaluation.
7. If Phase 11 is authorized, create its task/document baseline before editing code; do not redo completed Phase 10 tasks.

## Phase 10 initialization record

T1000 starts with the target at commit
`044b30f2340f19c91d11d2d555817cc4a3765c6d`, PVB at
`c08e5e3cd49d45c6d748387e78224843bd356f50`, and AnewOmni at
`926e99818ea18cf9d9b2064ce0319fe691b7a1f1`. The Phase 9 output directory
remains the protected baseline; Phase 10 writes, when authorized, only under
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10`.
T1000 initialization, T1001 diagnosis, T1002 pvb_full coverage, T1003
mode plumbing, T1004 posterior routing, T1005 parity, T1006 source
freezing, T1007 diagnostics, T1008 focused tests/smoke, and T1009 fixed-batch
validation are complete; T1010, T1011, T1012, T1013, and T1014 are complete and
no task is active. Phase 10 artifacts are protected under the Phase 10 output
tree; Phase 9 artifacts remain unchanged. The valid-only lock is immutable;
test was evaluated once, and the paired report records the writer recovery.

## Phase 11 handoff — authoritative latest state

This section supersedes older `Current state`, `In-progress task`, `Exact next
action`, and `Resume instructions` sections above while preserving them as
historical records.

### Current state

- Current phase: Phase 11 — lightweight shared-PVB H-block fusion
- Current task: T1110 — vendor the minimal Anew tokenizer and vocabulary assets
- Status: `IN_PROGRESS`
- T1100 status: `DONE`; branch/provenance and collision-free decision numbering verified
- Phase 10 status: complete; preserve its final commit, checkpoint, metrics,
  reports, and hashes before Phase 11 implementation
- Phase 11 branch: `phase11/pvb-shared-hblock`
- Phase 11 parent: final Phase 10 commit `fbf8302d57942dbc41a52e0e1019ecb8c0287687`
- Container: enter with `enter-container`
- Conda environment: `torch-ito`

### Repository locations

```text
Read-only PVB:      /workspace/PVB
Read-only AnewOmni: /workspace/AnewOmni
Writable target:   /workspace/fuse_anew_pvb_hblock
Phase 11 outputs:  /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11
```

Do not modify either source repository. Do not overwrite any Phase 9 or Phase
10 artifact.

### Intended architecture

```text
pretrained PVB TorchMD encoder -> h_atom
    |-> unchanged PVB posterior / x_rep / KL
    `-> detached h_atom
        -> sum/sqrt(N) pooling
        -> rank-32 adapter
        -> zero structural gate
        -> one post-cross decoder injection

optional Phase 11B semantic branch:
Anew principal-subgraph ID
    -> exact frozen official Anew block embedding, if provenance passes
    -> low-rank projection
    -> zero semantic gate
    -> combine with structural H-block condition
```

The new mode is `pvb_shared_hblock`. Existing `off`, Phase 9 legacy, and Phase
10 posterior-preserving modes remain unchanged.

### T1102 implementation result

- Status: `DONE`
- Target: `module/model.py`, `config/train.yaml`,
  `scripts/profile_train_step.py`, and `tests/test_phase11_shared.py`.
- `pvb_shared_hblock` is a distinct PVB-only mode; it constructs no Anew
  encoder/EPT. PVB `encode()` preserves its historical two-value return and
  exposes `h_atom`, `vec_atom`, and `log_var_pvb` only through
  `return_state=True`.
- PVB full-role checkpoint coverage is `195/195` with no mismatches. Focused
  tests passed: 17/17. Pooling, adapter, and decoder injection remain for
  T1103/T1104.


### T1103 implementation result

- Status: `DONE`
- Target: `module/shared_hblock.py`, `module/model.py`,
  `utils/fusion_training.py`, and `tests/test_phase11_shared.py`.
- The adapter reuses vendored Anew `std_conserve_scatter_mean`, consumes
  detached PVB `h_atom`, and broadcasts with explicit `atom_block_id`.
- The bottleneck is `LayerNorm -> Linear(32) -> SiLU -> Linear`; the final
  projection uses ordinary initialization and `shared_hblock_gate` is zero
  initialized, preserving a first-step gate gradient. At hidden size 256
  it has 7 parameter tensors and 17,185 parameters.
- Validation: focused shared/stage tests passed 14/14; compilation passed;
  legacy fusion and stage tests remained green. Decoder injection is complete
  under T1104.

### T1104 implementation result

- Status: `DONE`
- Target: `module/torchmd_et.py`, `module/model.py`, and
  `tests/test_phase11_shared.py`.
- `post_cross_condition` is added once after the decoder `x0`/`xt` merge;
  legacy `block_condition` semantics remain unchanged. Training, inference,

  and realization use it only for `pvb_shared_hblock`.
- Validation: focused Phase 11 tests passed 10/10; fusion/staged-training
  regression passed 14/14; compilation and `git diff --check` passed, including
  full-objective parity and SE(3) checks.

### T1105 provenance result

- Status: `DONE`
- Target: `scripts/audit_phase11_shared_provenance.py`,
  `tests/test_checkpoints.py`, and `tests/test_training_stages.py`.
- Real `pvb_full` coverage is `195/195` with no missing, unexpected, or
  shape-mismatched keys. All 195 source parameter tensors match checkpoint
  bytes and remain unchanged through source freezing.
- The exact non-source complement is seven shared adapter/gate tensors and
  17,185 parameters in one `projector_gate` optimizer group; no Anew encoder
  is constructed. The provenance artifact is
  `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/`
  `t1105_shared_provenance.json`, SHA256
  `a7209f4e51176e2daa90e55ebeb3404ebf0868b587209a3ba95b9044b0a6269a`.

### T1106 parity and compatibility result

- Status: `DONE`
- Target: `tests/test_phase11_shared.py`.
- Validation: `python -m py_compile tests/test_phase11_shared.py` and
  `python -m unittest -v tests.test_phase11_shared` passed 12/12.
- Gate-zero matched PVB `off` for posterior state, `x_rep`, KL, full stochastic
  objective, inference source sample, and decoder output; protected Phase 9
  legacy and Phase 10 corrected checkpoints loaded with complete coverage.

### T1101 audit result

- Status: `DONE`
- Report: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/profiles/t1101_audit.json`
- Real item: train record 0, 2147 atoms, 282 blocks, 4368 bonds.
- Forward-only diagnostic: PVB `off` 149.38 ms; Phase 9 legacy Anew 102.72 ms;
  Phase 10 corrected PVB-posterior plus Anew EPT 282.05 ms on an A100.
- Finding: Anew EPT materializes dense `[B, N, N]` attention and the corrected
  Phase 10 path runs both encoders. Phase 11A must reuse PVB `h_atom`, pool it,
  and never invoke Anew EPT. PVB `btype` remains separate from new semantic
  pooling metadata.
- Coverage/checksums: PVB full role 195/195; Anew role 68/68; source-frozen
  checksums unchanged; sampled real records used explicit block metadata.

### T1107 profile and fixed-batch result

- Status: `DONE`
- Target: `scripts/profile_training_paths.py`, `scripts/source_frozen_overfit.py`, and `phase11/shared_hblock_protein/` JSON artifacts.
- Real item: train record 0, 2147 atoms, 282 blocks, 4368 bonds; dataset/raw/view counts were identical.
- Profile: shared forward-only 79.029 ms / 1.813 GiB and source-frozen 117.069 ms / 9.527 GiB; Phase 10 corrected was 85.729 ms / 1.809 GiB and 138.208 ms / 9.531 GiB on the same A100/GPU slot. Shared has 17,185 trainable parameters and no Anew encoder.
- Fixed batch: 20/20 finite updates; `rec_total` 0.308612704 -> 0.308233753; gate gradient from step 0, projector gradient after step 1, exact optimizer membership, and source checksums all passed.
- Artifacts: `t1107_profile_off.json` `0af7aa5757ae517c91f41006c0730920e91a62d85b76cb9d73ecf2cbf18f988c`; `t1107_profile_phase10.json` `f87d42371174b919b20b0c14fdcdb634ef74292259d453cde44f1aaf26ecd56`; `t1107_profile_shared.json` `126b76db1d601decd875c56c873e1e0e0a5445320a0a4a6344b457879d667cfb`; `t1107_overfit_shared.json` `8f93740bbcbc99f391266c897c72407ab2b30ad0faaceb820602e0bf5165dfcb`.

### T1108 training/evaluation progress

- Status: `DONE`
- Runner: `scripts/phase11_shared_train_eval.py` trains only the seven shared
  adapter/gate tensors after complete `pvb_full` loading, traverses exact
  protein-only train/valid views, selects minimum valid `rec_total`, and writes
  a lock manifest with `test_evaluated=false`.
- Paired evaluator: `scripts/phase11_paired_eval.py` refuses an unlocked or
  already-tested manifest and evaluates PVB off, Phase 9 legacy, Phase 10
  corrected, and Phase 11 shared on the same valid/test views and three seeds.
- Smoke artifact: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/`
  `shared_hblock_protein/t1108_smoke.json` (debug-only, one train/valid item;
  not a quality result). The first smoke found a runner aggregation indexing
  error; it was fixed and the rerun passed with valid-only selection.
- Formal train/valid and the locked paired evaluation are complete. The
  paired report records the only Phase 11 test traversal; no test result was
  used to change the selected checkpoint.
- Formal-run storage audit: epochs 0–2 each completed train (6,200 steps /
  14,660,149 atoms) and valid (354 batches / 847,978 atoms); valid
  `rec_total` was `0.218641275`, `0.218670571`, then `0.218561299`, with KL
  `0.006957666` throughout. Replacement of the improved epoch-2 checkpoint
  failed because `/output` had `Avail 0`; the task-owned checkpoint was removed,
  Phase 9/10 artifacts were not touched, and the runner now deletes only its
  own old checkpoint before replacement. The final format is now an
  adapter-only checkpoint with external immutable `pvb_full` source state; it
  is about 109 KiB in the format smoke (`pvb_full=195/195`, adapter=`7/7`). The
  lock records `checkpoint_kind=phase11_adapter_only` and
  `optimizer_state_saved=false`; optimizer membership remains verified during
  the actual training run.

### T1108 formal train/valid result

- Status: `DONE_PENDING_PAIRED_EVAL`; the task remains the sole `IN_PROGRESS`
  task until the locked valid/test comparison is complete.
- The formal run completed 5 epochs / 30,984 steps, traversing 6,413 train
  items / 14,666,461 atoms and 367 valid items / 847,978 atoms. Validation
  batch-mean `rec_total` by epoch was `0.2186412669`, `0.2186705688`,
  `0.2185613004`, `0.2184535972`, `0.2183195718`; epoch 4 was selected using
  valid reconstruction only.
- Final valid metrics: loss `0.2238857052`, KL `0.0069576662`, `rec_vel`
  `0.0911244328`, `rec_drf` `0.1271951389`, `rec_total` `0.2183195718`;
  atom-weighted `rec_total` `0.2156988460`.
- Checkpoint: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/pvb_shared_hblock_best.ckpt`, SHA256 `fecb7371033bb2dc5f82d865890f182fb41991c43104b3b302533d0f8dcab08f`.
- Lock: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/phase11_shared_hblock_best.lock.json`, SHA256 `131138fa13701543d17c71ba2153ffd062ce798c8149f9ced7dbc61c95821178`, `status=locked`, `test_evaluated=false`, `checkpoint_kind=phase11_adapter_only`.
- External PVB source: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt`, SHA256 `4f0ad88356c7159cd5d0b9641b6c1e5c5f97a87ed95e7748c8189e1a110d1a77`; source coverage `195/195`, source checksums unchanged, exact adapter complement `7` tensors / `17,185` parameters, and no Anew encoder or Anew variance loss path.

### T1108 locked paired result

- The one-time evaluator used seeds `20260810`, `20260811`, `20260812` and
  identical views: valid 367 items / 847,978 atoms / 354 batches; test 167
  items / 334,142 atoms / 155 batches. All four models had identical traversal
  counts and complete checkpoint coverage.
- Batch-mean `loss / KL / rec_vel / rec_drf / rec_total`:

  | Model | Valid | Test |
  | --- | --- | --- |
  | PVB off | 0.273862243 / 0.006957666 / 0.102848231 / 0.165447878 / 0.268296109 | 0.255278479 / 0.007027225 / 0.098625390 / 0.151031311 / 0.249656701 |
  | Phase 9 legacy | 1.064952904 / 1.093585077 / 0.064429840 / 0.125654988 / 0.190084828 | 1.043927739 / 1.089811224 / 0.060861075 / 0.111217675 / 0.172078751 |
  | Phase 10 corrected | 0.264249999 / 0.006957666 / 0.099334824 / 0.159349042 / 0.258683866 | 0.246216296 / 0.007027225 / 0.095238273 / 0.145356245 / 0.240594518 |
  | Phase 11 shared | 0.260572391 / 0.006957666 / 0.097893420 / 0.157112839 / 0.255006258 | 0.242662904 / 0.007027225 / 0.093872097 / 0.143169028 / 0.237041126 |

- Atom-weighted `loss / KL / rec_vel / rec_drf / rec_total` for Phase 11
  shared was valid `0.251991546 / 0.006926987 / 0.095785901 / 0.150664055 /
  0.246449956` and test `0.240526811 / 0.006980063 / 0.093844615 /
  0.141098147 / 0.234942761`.
- Report: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/shared_hblock_protein/t1108_paired_valid_test.json`, SHA256 `ae078b126a3919936bc3a5b99c7f5cbaa7fa85d16b29012b9d6ca84f79064d1f`.
- Interpretation: Gate P11A passed. Shared improves `rec_total` versus PVB off
  and Phase 10 corrected on both splits without changing PVB KL. Legacy remains
  reconstruction-improving but KL-dominated and is not claimed as a fair total
  loss improvement. Phase 9/10 checkpoint hashes remain unchanged.

### T1109 result

- Status: `DONE`; official embedding provenance passed without changing the
  fused runtime model.
- Full official key: `base_model.autoencoder.embedding.block_embedding.weight`.
  Derived/extracted key: `embedding.block_embedding.weight`. Shape is `437 x 512`,
  dtype `torch.float32`, and full/derived/extracted tensor SHA256 is
  `2ba7c22abf1ca550d354d282e7c4ed2278ab972789ce223bf50db31abb69ddf`.
- Anew vocabulary order has 437 block types and 119 atom types; amino-acid IDs
  are exactly 1–20 after `UNK` at 0. Vocabulary order SHA256 is
  `b7e157f2f6cb62e673301430333fa4b6988573f68237dfb8b31fe806b4f133a1`.
- Artifacts: extracted table
  `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/anew_official_block_embedding.pt`
  SHA256 `ad31c380a2c6ab8cca24457948bdd8979829569150044c4022de3a938246f26c`;
  provenance JSON SHA256
  `2e2525fd578389072219d51fdf2698ab1de1d9f49e81bb14dcbd1502d55810ee`.
- `--verify-only` reproduction and `tests.test_phase11_official_embedding`
  passed 1/1. Official semantic table reuse is approved; tokenizer/vendor and
  semantic model integration are not yet complete.

### T1110 active task

- Status: `IN_PROGRESS`; this is the sole active task.
- Intended outcome: vendor only the minimal tokenizer, vocabulary implementation,
  required chemical helpers, and exact tokenizer asset under
  `third_party/anewomni/`, with package-relative imports and source parity.
- Source files: pinned Anew tokenizer/vocabulary files and dependencies at
  commit `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`.
- Target files: namespaced files under `third_party/anewomni/data/bioparse/` and
  `third_party/anewomni/utils/`, plus focused parity tests.
- Remaining work: copy complete source files, adapt imports only, record source
  file hashes, and compare tokenization/vocabulary outputs on representative
  molecules. Do not add semantic model conditioning yet.


### Required execution order

1. T1100 is complete: the actual Phase 10 parent SHA, source SHAs, target
   status, remote, branch, and preserved artifact hashes are recorded above.
2. All four live documents have been read completely; the appended decision
   block was mechanically renumbered from D053–D066 to D059–D072.
3. T1101 is complete; its source, block-semantics, cost, and real-batch
   audit are recorded above and in the Phase 11 profile artifacts.
4. T1106-T1108 are complete and Gate P11A passed; the locked paired report is
   the sole test traversal and Phase 9/10 artifacts remain immutable.
5. T1109 is complete with exact official embedding provenance; T1110 is the sole
   active task and must finish tokenizer/vocabulary parity before semantic model
   integration.
6. Continue T1111-T1116 only after T1110 parity passes, then close in T1117.

Only one task may be `IN_PROGRESS`. Update the task evidence, decisions, and
this handoff after each completed task.

### Mandatory invariants

- PVB posterior, `x_rep`, and KL are unchanged.
- Shared source features are detached in the adapter-only experiment.
- No Anew EPT call occurs in `pvb_shared_hblock`.
- Legacy PVB `btype` is not replaced with Anew fragment IDs.
- Fragment metadata is computed offline from complete bond chemistry.
- Gate-zero full-objective and inference parity pass before training.
- Source-loaded parameters remain bitwise unchanged.
- Test is not used for tuning.
- Earlier modes and checkpoints remain reproducible.

### Exact next action

```text
cd /workspace/fuse_anew_pvb_hblock
T1111 objective: complete the CPU-only semantic fragment metadata audit and
prove that bond order, atom assignment, and fallback behavior are explicit.
```

Phase 11A is closed and T1110 vendor parity passed. T1111 is blocked until
authoritative bond orders are available for the five affected train SDF bonds
and the 508 tokenizer mapping assertions are resolved or explicitly accepted.

### Blockers

- T1111 is blocked by five train SDF bonds with `UNSPECIFIED` order and 508
  explicit tokenizer fallbacks affecting 15,772 atoms; no semantic branch may start.
- Phase 9, Phase 10, and Phase 11A artifacts remain immutable.

### Resume instructions

1. Enter the container and activate `torch-ito`.
2. Change to `/workspace/fuse_anew_pvb_hblock`.
3. Read `PLAN.md`, `DECISIONS.md`, `TASKS.md`, and this latest handoff section.
4. Verify the branch and current task state; T1111 is `BLOCKED` and no task is
   `IN_PROGRESS` until the chemistry blocker is resolved.
5. Continue that task only; do not redo completed Phase 9 or Phase 10 work.

## Phase 11 live update — T1110 complete, T1111 blocked

- Current phase: Phase 11 — lightweight shared-PVB H-block fusion
- Current task: T1111 — materialize and validate Anew semantic fragment metadata (blocked)
- Status: `BLOCKED`; T1110 is `DONE`; no task is active.
- Last updated: 2026-08-14

### T1110 result

T1110 passed. Thirteen pinned Anew tokenizer/vocabulary/helper files were
vendored under `third_party/anewomni/`. All mapped files compare byte-identically
after the three documented import-only adaptations. Source/target vocabulary and
tokenization parity passed on four representative SMILES; the focused vendor
and official-embedding tests passed 3/3. The provenance report is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/t1110_anew_vendor_provenance.json`
with SHA256 `ad1ec26af7eb4282b4c31d060d3adeba5873de8ab67d51e6df29093e6e40ecc4`.

### T1111 blocked task

- Status: `BLOCKED`; no semantic model integration may start. T1111 found five train SDF bonds with `UNSPECIFIED` order and 508 explicit tokenizer fallbacks (15,772 atoms); no semantic dataset was created and original mmap/SDF data is untouched.
- Source files: raw PVB/PDBBind views, half-PVB manifest, Anew tokenizer/vocabulary.
- Target files: `scripts/audit_phase11_semantic_data.py` and `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/t1111_semantic_data_audit.json` (SHA256 `62ee4655760f887a7d66bc096cda5c0277ef18cc70606e09d8cdf9b9ab3c1031`); no semantic dataset was created and original mmap/SDF data is untouched.

### Verification at blocker

- Focused vendor/official/shared Phase 11 tests: 15/15 passed.
- `python -m py_compile scripts/audit_phase11_semantic_data.py scripts/audit_anew_vendor.py` passed.
- `git diff --check` passed; PVB and AnewOmni source worktrees remain clean.
- Full target discovery was manually interrupted after approximately 15 minutes without output; it is not claimed as passed.

### Blocker and exact next action

```text
Obtain authoritative bond-order values for the five UNSPECIFIED train SDF bonds
and decide how to resolve the 508 Anew tokenizer mapping assertions. Re-run the
full CPU audit; only after it passes may T1112 semantic model integration start.
```

The official semantic table and vendor behavior are provenance-approved, but
T1111 is blocked by source chemistry/fallback quality. The Phase 11A shared PVB
branch remains valid; Phase 9/10/11A artifacts remain immutable.

## Phase 12 live update — T1200 coordinator active

- Current phase: Phase 12 — source chemistry recovery and PVB dataset materialization
- Current task: T1200 — coordinate the chemistry/xyz2mol and PDB/full-half lanes
- Status: `IN_PROGRESS`; T1201 chemistry audit is complete; T1202 remains the active parallel data lane under T1200.
- Last updated: 2026-08-14 (chemistry scope corrected)

### Parallel worker A — T1201 chemistry source and xyz2mol

- Scope: inspect `/data4/PVB/pcqm4mv2`, `/data4/PVB/ani1x`, and
  `/data4/PVB/pdbbind`; use complete SDF chemistry directly; benchmark only
  damaged SDF ligand coordinates and ANI unique molecular groups. Protein or
  receptor coordinates must never enter a PDBBind bond-assignment probe.
- Stop gate: if projected full processing exceeds four hours, stop and document
  the measured rate; do not launch full conversion.
- Output: separate Phase 12 audit/reconstruction path only; raw sources remain untouched.

### Parallel worker B — T1202 PDB and PVB full/half materialization

- Scope: inspect `rsyncPDB.sh` and `/data4/PVB/pdb/ept_release`, download into
  `/data4/PVB/pdb/`, produce full PVB data first, then derive half data under
  `/data4/users/sihao/data` or a documented cross-dataset subdirectory.
- Constraint: processed PDB data is not used by Phase 11 tests.
- Output: split manifests, counts, provenance, and hashes; no silent overwrite.
- Acquisition report: `reports/phase12/t1202_pdb_materialization.md`.
- Current: official mirror is running in 36 non-overlapping prefix lanes without
  transport compression and with `--partial`; no `--delete` is used; Phase 11
  inputs remain separate. The resumed pool's latest sample is 24,019 files and
  about 5.02 GB; materialization remains blocked until the mirror completes.


### T1201 corrected result and current next action

- The initial T1201 DONE scope is superseded because its PDBBind timing did not
  prove a ligand-only input boundary; the corrected T1201 audit is now complete.
- The corrected probe passed only SDF ligand atoms/conformer coordinates. The
  five damaged records have 14–22 atoms, took 0.061–1.594 seconds, and all
  returned zero usable xyz2mol molecules. A separate 20-record ligand-only
  diagnostic had 5 calls that returned and 15 bounded timeouts. This rules out using the
  old timing to infer protein-coordinate inclusion, but also rules out applying
  inference to valid SDF records.
- ANI has no explicit bond-order/aromaticity arrays; process its 3,114 unique
  molecular groups separately and reuse a validated topology across 4,956,005
  conformers. RDKit rdDetermineBonds is a faster C++ xyz2mol implementation,
  while Open Babel is a second heuristic; neither is an independent authority.
- T1111 remains blocked by five UNSPECIFIED train SDF bonds and 508 tokenizer

Phase 9/10/11A artifacts and existing Phase 11 protein-only views remain immutable.

The target helper `scripts/phase12_make_pdb_manifests.py` is ready to create
structure-level full/half manifests once the EPT index exists.

### Phase 12 exact next action

Wait for the 36-prefix official rsync pool to exit successfully, then create
the clean symlink-only EPT input directory and run the existing
`process_PDB_monomer.py` into a new `phase12_full/ept_processed_pdb` directory.
Expected result: source file count matches the dry run and an EPT `index.txt`
exists; if either check fails, stop before PVB materialization and record the
failed prefix/log rather than treating a partial mirror as complete.
## Latest update — T1111 prerequisite probe and P9/P10/P11A schematic

- Current phase: Phase 12 data workstreams remain active; Phase 11A is complete,
  while semantic Phase 11B is blocked at T1111.
- Current task: T1200 is the sole IN_PROGRESS task; T1111 is BLOCKED and
  T1112–T1117 are TODO.
- Last updated: 2026-08-14 after the T1111 prerequisite probe and report update.

### T1111 prerequisite result

- The existing CPU audit remains authoritative:
  /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/t1111_semantic_data_audit.json
- It covered 6,947 records, 219,314 ligand atoms, and 232,741 ligand bonds;
  PVB/SDF heavy-atom order was exact for all records.
- The gate is still blocked by five train SDF bonds with UNSPECIFIED order and
  508 tokenizer-fallback records affecting 15,772 atoms. The corrected Phase 12
  ligand-only probes returned no usable graph for the five damaged records.
- Audit SHA256:
  62ee4655760f887a7d66bc096cda5c0277ef18cc70606e09d8cdf9b9ab3c1031
- No semantic dataset was materialized and no T1112–T1117 model task was started.

### P9/P10/P11A report

- Schematic and conclusions:
  reports/phase11/p9_p10_p11a_architecture.md
- Report SHA256: 7d8ab3cefcdb3820ce0b70c192ed64063ed19cb4cc97c1a3207b4b9e324fabd8
- P9 legacy lowers reconstruction but uses Anew block variance in the PVB
  posterior, producing valid KL 1.093585077 and total loss 1.064952904.
- P10 restores the complete PVB posterior and uses Anew H-block only for
  decoder conditioning: valid KL 0.006957666, rec_total 0.258683866.
- P11A pools shared PVB h_atom with variance-preserving block normalization and
  trains only the adapter/gate: valid KL 0.006957666, rec_total 0.255006258.
- P11A test batch means are KL 0.007027225 and rec_total 0.237041126.
  It is not semantic fragment fusion and does not close Gate P11.

### Exact next action

1. Keep T1111 blocked; do not infer the five missing bond orders or silently
   accept the 508 fallback records.
2. Keep T1200 active while the resumable PDB mirror completes; do not use
   Phase 12 processed PDB data in Phase 11 tests.
3. Resume T1112 only after an authoritative chemistry/fallback policy makes
   the complete T1111 audit pass.

## Latest chemistry follow-up — T1201 repair probe

- Five PDBBind train SDFs (`3sjt`, `3skk`, `4q3s`, `4ie2`, `4hxq`) each have
  one existing `UNSPECIFIED` bond. Isolated `0 -> SINGLE` candidates preserve
  the raw graph and pass Anew tokenization only with `sanitize=False`; normal
  RDKit sanitization fails on four-coordinate boron, so no candidate is promoted.
- The 508 fallback records are all PDBBind half-view records: train/valid/test
  `480/15/13`; their SDF bond types and PVB/SDF atom order are complete. The
  failure is Anew `get_submol_atom_map()` assertion, not missing SDF chemistry.
- All-508 probe: RDKit 68 tokenizer passes/0 exact graphs; Open Babel 50 passes/1
  exact graph, but the exact record still fails tokenizer. No geometry rewrite is
  accepted. T1111 remains BLOCKED.

### Artifacts

- Candidates: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase12/t1201_repair_audit/repaired_sdf_candidates/`
- Candidate summary SHA256: `97fb45e5fab0b5e4fe7ebcf02b71d16892f4886edd321a28a845f324710c869d`
- Full fallback probe SHA256: `483c027676c9739a5cb5da6b7e9db02973fb3b6b586163c0735c0ab3e7bb1b51`

### Exact next action

1. Continue T1200 by waiting for the resumable PDB mirror; do not materialize from a partial tree.
2. Keep the five candidates isolated and do not alter raw SDFs or Phase 11 inputs.
3. Do not start T1112 until an authoritative chemistry or explicitly approved tokenizer policy closes T1111.

## Active Phase 13 — H-block ablation plan

Phase 13 is now active. T1301 is the sole `IN_PROGRESS` task; T1302-T1310
remain `TODO`. T1300 is complete; four matched PVB-only adapter controls are
training under the protected Phase 13 artifact root.

### Motivation and registered baseline

- Phase 9 valid: KL 1.093585077, `rec_total` 0.190084828. Reconstruction
  improved, but the Anew block variance replaced the PVB posterior and made the
  result KL-confounded.
- Phase 10 valid: KL 0.006957666, `rec_total` 0.258683866.
- Phase 11A valid: KL 0.006957666, `rec_total` 0.255006258; test
  `rec_total` 0.237041126.
- Phase 11A trains 17,185 parameters versus Phase 10's 33,281. The small
  Phase 11A gain is not explained by a larger raw parameter count.

### Planned comparisons

1. Current real shared-PVB H-block versus deterministic sample-local shuffled
   H-block and a constant/zero-input branch with the same adapter/gate capacity.
2. Pooled PVB H-block versus detached PVB atom features with no block pooling.
3. Shared-PVB versus shape-matched Anew H-block through the identical rank-32
   post-merge adapter.
4. Post-merge versus cross-attention injection while holding feature source,
   pooling, adapter, initialization, and optimizer protocol fixed.

All variants preserve the complete PVB posterior. Gaussian H-block, block KL,
Anew log-variance use, coordinate residuals, and source unfreezing are excluded.

### Phase 13 gate and artifacts

- New artifacts: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13`.
- Gate zero must match `off` across the full objective and PVB KL.
- Source checksums, optimizer membership, parameter-count matching, deterministic
  batch-isolated shuffling, and fixed-real-batch gradients must pass before
  training.
- Checkpoints are selected by valid `rec_total` only. Locked variants receive
  one paired complete valid/test evaluation with three registered seeds,
  per-record deltas, and bootstrap confidence intervals.
- Phase 9/10/11/12 artifacts remain protected.

### Exact next action

1. Monitor the four Phase 13 training JSON files until every valid-only run
   completes and writes its lock manifest.
2. Expected result: each control has complete traversal, unchanged PVB source
   checksums, and a valid-selected checkpoint; test remains untouched.
3. If a training, source-checksum, lock, or data-identity gate fails, stop the
   ablation run and document the blocker; the asynchronous PDB tree is not a fallback.

### Phase 12 live data audit — 2026-08-14

- Chemistry: T1201 is complete; the 508 records are Anew tokenizer mapping
  failures with complete SDF chemistry, not missing-bond records. The five
  damaged records were not promoted.
- PDB acquisition: the partial tree now contains 49,398 files and
  10,561,005,045 bytes (about 10 GB); the remote estimate is 242,490 regular
  files and 53.6 GB. No materialized full/half PVB dataset is ready yet.
- Disk: `/data4` has about 409 GB free of 3.6 TB (89% used). This is currently
  sufficient, but materialization must recheck headroom and avoid duplicate
  raw/intermediate archives.
- Phase 13 input rule: use only the existing protein-only Phase 9/10/11A
  views; never read the partial PDB tree in these experiments.

## Latest Phase 13 handoff — four-control tranche complete

- Current phase: Phase 13 — H-block information, capacity, and pooling ablations.
- Current task: none — Phase 13 four-control tranche is closed.
- Status: `DONE` for T1300-T1310 tranche; T1304/T1305 remain explicitly unrun
  TODO extensions; Phase 11B is deferred; T1202 remains asynchronous; T1111
  remains BLOCKED.
- Last updated: 2026-08-14 after the final Phase 13 audit.

### Phase 13 result

- Valid-only training completed for `pvb_shared_real`, `pvb_shared_shuffled`,
  `pvb_shared_constant`, and `pvb_atom_no_pool`; each has 17,185 trainable
  adapter/gate parameters and unchanged PVB source checksums.
- One paired evaluator run completed valid then test with seeds
  `20260810/20260811/20260812`. Every model had valid `367/847,978/354` and
  test `167/334,142/155` items/atoms/batches, with identical oversized counts.
- Batch-mean `rec_total` valid/test: off `0.268296111/0.249656701`, real
  `0.255006258/0.237041124`, shuffled `0.255652213/0.237688745`, constant
  `0.255703018/0.237712422`, atom-no-pool `0.252977388/0.235095688`.
- Real beats shuffled/constant by approximately `0.00065–0.00070`; all
  adapter controls beat off; atom-no-pool is best. KL remains PVB-like for all
  variants. Interpretation: common adapter/injection capacity dominates, with
  a small record-specific H-block signal; pooling is not established as
  optimal. No Gaussian H-block conclusion is justified.
- Aggregate artifact:
  `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/evaluation/paired_valid_test.json`
  SHA256 `1353962086cbfbb512ea29d53fed89c66429f73c813eeba5ed2d906968ccda53`.
  The evaluator did not emit per-record vectors; test was not rerun for
  bootstrap.

### Phase 12 current snapshot

- Chemistry T1201 remains complete: the 508 cases have complete SDF chemistry
  but fail Anew tokenizer mapping; the five damaged SDF repair candidates were
  not promoted. T1111/Phase 11B remains deferred.
- PDB tree snapshot: 50,189 files and approximately 11G under `/data4/PVB/pdb`;
  remote estimate 242,490 files / 53.6GB; no full/half EPT/PVB materialization.
- Disk: 3.6T filesystem, 409G available, 89% used. This is enough for bounded
  continuation, but space must be rechecked before materialization and duplicate
  full archives must be avoided. Phase 13 did not use the partial PDB tree.

### Exact next action

1. For a future session, register a new paired experiment for T1304/T1305 with
   per-record output captured before its single test traversal.
2. Keep Phase 11B deferred until Anew fragment coverage/mapping is resolved.
3. Do not add a stochastic block latent until the deterministic feature-source
   and injection comparisons are complete.
