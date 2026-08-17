# T1309 — Phase 13 paired valid/test evaluation

Date: 2026-08-14

## Scope and lock

Phase 11B was skipped because Anew fragment coverage is unresolved. Phase 13
used only the existing complete protein-only Phase 9 views. Four matched
adapter controls were trained with PVB source parameters frozen and selected by
valid `rec_total` only. Test was not read during training or checkpoint
selection.

The one-time evaluator was:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.phase13_paired_eval \
  --device cuda:0 \
  --output /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/evaluation/paired_valid_test.json \
  --fused-data-root /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/pdbind_protein_only \
  --pvb-checkpoint /output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/pvb_state_dict.pt \
  --variant-checkpoint pvb_shared_real=/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/checkpoints/real_best.ckpt \
  --variant-checkpoint pvb_shared_shuffled=/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/checkpoints/shuffled_best.ckpt \
  --variant-checkpoint pvb_shared_constant=/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/checkpoints/constant_best.ckpt \
  --variant-checkpoint pvb_atom_no_pool=/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/checkpoints/atom_no_pool_best.ckpt \
  --eval-seeds 20260810 20260811 20260812
```

The JSON artifact is
`/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase13/evaluation/paired_valid_test.json`.
Its SHA256 is
`1353962086cbfbb512ea29d53fed89c66429f73c813eeba5ed2d906968ccda53`.

## Paired traversal identity

| Split | Items | Atoms | Batches | Oversized batches |
| --- | ---: | ---: | ---: | ---: |
| valid | 367 | 847,978 | 354 | 239 |
| test | 167 | 334,142 | 155 | 70 |

All five models had identical counts, batch counts, atom counts, and oversized
counts on each split. Seeds were `20260810`, `20260811`, and `20260812`.

## Batch-mean metrics

`rec_total = rec_vel + rec_drf`. The values below average the three seed
summaries; KL is the PVB posterior KL and is not affected by the H-block
controls.

| Model | Split | Loss | KL | rec_vel | rec_drf | rec_total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PVB off | valid | 0.273862246 | 0.006957666 | 0.102848232 | 0.165447880 | 0.268296111 |
| shared real | valid | 0.260572391 | 0.006957666 | 0.097893420 | 0.157112838 | 0.255006258 |
| shared shuffled | valid | 0.261218345 | 0.006957666 | 0.098213449 | 0.157438763 | 0.255652213 |
| shared constant | valid | 0.261269151 | 0.006957666 | 0.098168437 | 0.157534581 | 0.255703018 |
| atom no pool | valid | 0.258543521 | 0.006957666 | 0.097224042 | 0.155753346 | 0.252977388 |
| PVB off | test | 0.255278479 | 0.007027225 | 0.098625392 | 0.151031309 | 0.249656701 |
| shared real | test | 0.242662902 | 0.007027225 | 0.093872097 | 0.143169027 | 0.237041124 |
| shared shuffled | test | 0.243310523 | 0.007027225 | 0.094209237 | 0.143479508 | 0.237688745 |
| shared constant | test | 0.243334200 | 0.007027225 | 0.094157408 | 0.143555014 | 0.237712422 |
| atom no pool | test | 0.240717467 | 0.007027224 | 0.093273264 | 0.141822424 | 0.235095688 |

The atom-weighted metrics are stored in the JSON artifact. For reference,
the valid/test atom-weighted `rec_total` values are respectively:

- PVB off: `0.259520650` / `0.247445792`.
- Shared real: `0.246449956` / `0.234942759`.
- Shared shuffled: `0.247124438` / `0.235617015`.
- Shared constant: `0.247155104` / `0.235623556`.
- Atom no pool: `0.244462153` / `0.233088890`.

## Interpretation

1. All three pooled adapter controls improve over PVB off by roughly
   `0.0126–0.0133` batch-mean `rec_total` on valid/test. Because the controls
   have the same 17,185 trainable adapter/gate parameters and the off model has
   no such trainable path, this improvement is compatible with adapter/injection
   capacity being the dominant effect.
2. Real shared H-block beats the sample-local shuffled control by
   `0.000645955` on valid and `0.000647621` on test batch-mean `rec_total`.
   It beats the constant control by `0.000696760` on valid and `0.000671298` on
   test. This is a small but consistent signal supporting record-specific
   information beyond matched capacity.
3. Atom no pool is best in this experiment, by `0.002028870` valid and
   `0.001945436` test versus shared real. This does not prove that the block
   representation is useless: it indicates that the current variance-preserving
   atom-to-block pooling/conditioning path loses useful atom-level information,
   or that the no-pool control is a different and more expressive injection.
   A separately matched injection/feature-source experiment is still needed.
4. KL is identical to PVB off within numerical noise for every control. The
   Phase 9 KL degradation is therefore not present in Phase 13; Anew
   `Wx_log_var` is absent from this loss graph.

These results do not support the claim that the H-block should be made
Gaussian. They support first isolating deterministic information, pooling, and
injection placement. A stochastic block latent remains a separate future
experiment with its own KL schedule and zero gate.

## Limitations

This evaluator stores complete paired batch-mean and atom-weighted aggregates,
not per-record metric vectors; therefore no post-hoc per-record bootstrap was
claimed. Test was deliberately not rerun to generate such vectors because the
protocol requires one test traversal after locking. The paired aggregate
conclusion above is valid, but a future experiment may add per-record output to
the evaluator before its one-time test run.

## Phase 12 status snapshot

At the same work session, `/data4/PVB/pdb` contained approximately 50,189 files
and 11G. `df -h` reported 409G free on the shared 3.6T filesystem (89% used).
The remote mirror estimate remains about 242,490 files and 53.6GB. The raw
mirror has enough current headroom, but PVB/EPT materialization must recheck
space after acquisition and avoid duplicate full copies. The PDB tree was not
used in this evaluation.
