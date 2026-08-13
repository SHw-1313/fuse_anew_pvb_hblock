# Handoff

## Current state

- Current phase: Phase 10 — PVB-posterior-preserving H-block conditioning
- Current task: T1014 — completed; no active task
- Status: COMPLETE; Phase 10 Gate P10 PASSED; T1000–T1014 complete
- Last updated: 2026-08-13
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

- Branch: `main`
- Commit: `044b30f2340f19c91d11d2d555817cc4a3765c6d` (first evaluation)
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

## In-progress task

- None — Phase 10 is closed; no task is IN_PROGRESS.
- Intended outcome: N/A for Phase 10. A future Phase 11 must begin with its own task/document initialization.
- Source files being reused: Phase 10 audit/report/checkpoint manifests and the official Anew provenance audit.
- Target files: no further Phase 10 edits; Phase 9 and Phase 10 artifacts remain immutable.
- Completed evidence: final audit SHA e48d25108041f0948d7cbb94acd5dada53694d3b0d3da46269aeb2541fdfd050; target tests 36/36; source worktrees clean; protected Phase 9 hashes unchanged.
- Remaining work: none for Phase 10. The proposed Phase 11 official-Anew alignment is intentionally not implemented.

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
