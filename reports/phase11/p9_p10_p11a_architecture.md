# P9 → P10 → P11A: H-block conditioning, posterior semantics, and loss

This is the Phase 11A schematic.  `P11` below means the completed shared-PVB
adapter (`pvb_shared_hblock`); semantic Phase 11B remains blocked at T1111.

~~~mermaid
flowchart LR
  I[Protein-only PDBBind view<br/>x0, atom types, blocks]

  subgraph P9[Phase 9 · legacy anew_block]
    P9A[Anew BlockEmbedding + EPT]
    P9B[H_atom → H_block]
    P9V[Anew Wx_log_var<br/>broadcast to atoms]
    P9R[x_rep and KL<br/>PVB decoder prior mismatch]
    P9C[H_block projection + zero gate]
    P9D[PVB velocity/drift decoder]
    P9L[loss = 0.8·KL_Anew + rec_vel + rec_drf<br/>valid: 1.064953 = 0.8×1.093585 + 0.190085]
    P9A --> P9B
    P9B --> P9V --> P9R --> P9D
    P9B --> P9C --> P9D --> P9L
  end

  subgraph P10[Phase 10 · anew_block_pvb_posterior]
    P10P[PVB encoder + W_vec_log_var]
    P10R[x_rep and KL_PVB]
    P10A[Anew BlockEmbedding + EPT]
    P10B[H_atom → H_block]
    P10C[block projection + zero gate]
    P10D[PVB decoder]
    P10L[loss = 0.8·KL_PVB + rec_vel + rec_drf<br/>valid: 0.264250 = 0.8×0.006958 + 0.258684]
    P10P --> P10R --> P10D
    P10A --> P10B --> P10C --> P10D --> P10L
    P10R -. posterior is independent .-> P10D
  end

  subgraph P11[Phase 11A · pvb_shared_hblock]
    P11P[PVB encoder + W_vec_log_var]
    P11R[x_rep and KL_PVB]
    P11H[PVB h_atom]
    P11B[sum(H_atom) / sqrt(block_length)<br/>shared H_block]
    P11C[rank-32 adapter + zero gate]
    P11D[PVB decoder<br/>one post-merge condition]
    P11L[loss = 0.8·KL_PVB + rec_vel + rec_drf<br/>valid: 0.260572 = 0.8×0.006958 + 0.255006]
    P11P --> P11R --> P11D
    P11P --> P11H --> P11B --> P11C --> P11D --> P11L
    P11R -. posterior is unchanged .-> P11D
  end

  I --> P9A
  I --> P10P
  I --> P11P
~~~

All reconstruction values below are batch means from the locked three-seed
paired report.  `rec_total = rec_vel + rec_drf`; the atom-weighted values are
included for completeness.

| Model | Split | Loss | KL | rec_vel | rec_drf | rec_total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PVB off | valid | 0.273862243 | 0.006957666 | 0.102848231 | 0.165447878 | 0.268296109 |
| P9 legacy | valid | 1.064952904 | 1.093585077 | 0.064429840 | 0.125654988 | 0.190084828 |
| P10 corrected | valid | 0.264249999 | 0.006957666 | 0.099334824 | 0.159349042 | 0.258683866 |
| P11A shared | valid | 0.260572391 | 0.006957666 | 0.097893420 | 0.157112839 | 0.255006258 |
| PVB off | test | 0.255278479 | 0.007027225 | 0.098625390 | 0.151031311 | 0.249656701 |
| P9 legacy | test | 1.043927739 | 1.089811224 | 0.060861075 | 0.111217675 | 0.172078751 |
| P10 corrected | test | 0.246216296 | 0.007027225 | 0.095238273 | 0.145356245 | 0.240594518 |
| P11A shared | test | 0.242662904 | 0.007027225 | 0.093872097 | 0.143169028 | 0.237041126 |

## Conclusions

1. **P9 isolated the posterior mismatch.**  H-block conditioning reduced both
   reconstruction terms, but broadcasting Anew `Wx_log_var` into PVB's
   posterior made KL about 157× larger on valid.  The lower reconstruction
   number therefore did not represent a lower total objective.
2. **P10 fixed the objective contract.**  PVB supplies `x_rep` and KL, while
   Anew H-block only conditions the decoder.  KL returned to PVB scale and the
   paired reconstruction improved over PVB off, at the cost of running a
   second dense Anew EPT path.
3. **P11A removed the redundant encoder without changing the posterior.**  It
   pools the already-available PVB `h_atom`, uses the Anew variance-preserving
   block normalization, and trains only the rank-32 projection/gate (7 tensors,
   17,185 parameters).  It improves `rec_total` over both PVB off and P10 on
   valid and test while retaining PVB KL.
4. **Scope boundary.**  P11A is not semantic Anew fragment integration: it
   does not construct Anew EPT and does not use Anew `Wx_log_var`.  T1111 is
   still blocked by five unspecified train bonds and 508 tokenizer fallback
   records, so T1112–T1117 must not be marked complete.

## Provenance

- Metrics: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/shared_hblock_protein/t1108_paired_valid_test.json`
- Phase 11A checkpoint: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/pvb_shared_hblock_best.ckpt`
- T1111 blocker audit: `/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/t1111_semantic_data_audit.json`
- Source revisions: PVB `c08e5e3cd49d45c6d748387e78224843bd356f50`; AnewOmni `926e99818ea18cf9d9b2064ce0319fe691b7a1f1`.
