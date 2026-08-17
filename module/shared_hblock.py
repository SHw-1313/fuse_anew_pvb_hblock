"""Lightweight PVB-feature to block-feature adapter for Phase 11A.

The pooling operation is imported from the vendored Anew implementation so
that the shared path uses the same variance-preserving convention as Anew's
reference encoder.  Phase 13 adds deterministic information and pooling
controls without changing the Phase 11A default semantics.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from third_party.anewomni.utils.gnn_utils import std_conserve_scatter_mean
from utils.phase13_ablation import PHASE13_ADAPTER_VARIANTS


def pool_pvb_h_atom(
    h_atom: torch.Tensor,
    atom_block_id: torch.Tensor,
    block_lengths: torch.Tensor,
) -> torch.Tensor:
    """Pool PVB scalar atom features using Anew's variance-preserving rule.

    ``atom_block_id`` and ``block_lengths`` are explicit integer metadata.
    The vendored Anew helper performs ``scatter_sum / sqrt(count)``. The
    supplied lengths remain explicit metadata for the block contract; they are
    not inferred from coordinates or floating-point block centers in forward.
    """

    if h_atom.ndim != 2:
        raise ValueError(f"h_atom must have shape [N_atom, hidden], got {tuple(h_atom.shape)}")
    if atom_block_id.ndim != 1 or atom_block_id.shape[0] != h_atom.shape[0]:
        raise ValueError("atom_block_id must be one entry per atom")
    if atom_block_id.dtype not in (torch.int32, torch.int64):
        raise TypeError("atom_block_id must be an integer tensor")
    if block_lengths.ndim != 1:
        raise ValueError("block_lengths must be a one-dimensional tensor")
    if block_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_lengths must be an integer tensor")
    if atom_block_id.numel() == 0:
        if block_lengths.numel() != 0:
            raise ValueError("empty atom_block_id requires empty block_lengths")
        return h_atom.new_empty((0, h_atom.shape[-1]))

    # Phase 11A is adapter-only: the pretrained PVB encoder is a source
    # feature provider and receives no gradient through this branch.
    return std_conserve_scatter_mean(h_atom.detach(), atom_block_id.long(), dim=0)


class SharedHBlockAdapter(nn.Module):
    """Rank-constrained adapter with explicit Phase 13 ablation controls.

    real is the Phase 11A behavior. shuffled permutes pooled blocks only within
    each sample. constant replaces record-specific features by one fixed
    deterministic vector. atom_no_pool applies the same adapter directly to
    detached atom features. All controls retain identical trainable
    projection/gate parameterization.
    """

    VALID_VARIANTS = frozenset(PHASE13_ADAPTER_VARIANTS)

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 32,
        variant: str = "real",
        shuffle_seed: int = 20260810,
    ):
        super().__init__()
        if hidden_dim <= 0 or rank <= 0:
            raise ValueError("hidden_dim and rank must be positive")
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"unsupported shared H-block variant {variant!r}; "
                f"expected one of {sorted(self.VALID_VARIANTS)}"
            )
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.variant = str(variant)
        self.shuffle_seed = int(shuffle_seed)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.rank),
            nn.SiLU(),
            nn.Linear(self.rank, self.hidden_dim),
        )
        # The model-level scalar gate makes the complete branch an exact
        # no-op at initialization while preserving a nonzero first-step gate
        # gradient; adapter gradients begin after the gate moves off zero.

    def _shuffle_within_sample(
        self,
        h_block: torch.Tensor,
        block_batch: torch.Tensor,
    ) -> torch.Tensor:
        if block_batch.ndim != 1 or block_batch.shape[0] != h_block.shape[0]:
            raise ValueError("block_batch must contain one entry per block")
        shuffled = h_block.clone()
        for sample_id in torch.unique(
            block_batch.detach().long(), sorted=True
        ).tolist():
            indices = torch.where(block_batch.long() == int(sample_id))[0]
            if indices.numel() < 2:
                continue
            generator = torch.Generator(device=h_block.device)
            generator.manual_seed(self.shuffle_seed + 104729 * int(sample_id))
            permutation = torch.randperm(
                indices.numel(), device=h_block.device, generator=generator
            )
            shuffled[indices] = h_block[indices[permutation]]
        return shuffled

    def _constant_features(
        self, block_count: int, reference: torch.Tensor
    ) -> torch.Tensor:
        if block_count == 0:
            return reference.new_empty((0, self.hidden_dim))
        # A fixed nonzero vector keeps the gate gradient informative while
        # carrying no record-specific or cross-sample feature content.
        values = torch.linspace(
            -1.0,
            1.0,
            self.hidden_dim,
            device=reference.device,
            dtype=reference.dtype,
        )
        return values.unsqueeze(0).expand(block_count, -1)

    def forward(
        self,
        h_atom: torch.Tensor,
        atom_block_id: torch.Tensor,
        block_lengths: torch.Tensor,
        block_batch: torch.Tensor | None = None,
        variant: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        chosen = self.variant if variant is None else str(variant)
        if chosen not in self.VALID_VARIANTS:
            raise ValueError(
                f"unsupported shared H-block variant {chosen!r}; "
                f"expected one of {sorted(self.VALID_VARIANTS)}"
            )
        h_block = pool_pvb_h_atom(h_atom, atom_block_id, block_lengths)
        if chosen == "shuffled":
            if block_batch is None:
                raise ValueError("shuffled H-block control requires block_batch")
            h_source = self._shuffle_within_sample(h_block, block_batch)
            condition_source = h_source
            condition_block = self.projection(h_source)
            condition_atom = condition_block.index_select(
                0, atom_block_id.long()
            )
        elif chosen == "constant":
            h_source = self._constant_features(h_block.shape[0], h_block)
            condition_source = h_source
            condition_block = self.projection(h_source)
            condition_atom = condition_block.index_select(
                0, atom_block_id.long()
            )
        elif chosen == "atom_no_pool":
            # Keep the pooled tensor available for shape diagnostics, but do
            # not use it to form the decoder condition in this control.
            condition_block = h_block.new_empty((0, self.hidden_dim))
            condition_source = h_atom.detach()
            condition_atom = self.projection(condition_source)
        else:
            condition_source = h_block
            condition_block = self.projection(condition_source)
            condition_atom = condition_block.index_select(
                0, atom_block_id.long()
            )
        return {
            "H_block": h_block,
            "condition_block": condition_block,
            "condition_atom": condition_atom,
            "atom_block_id": atom_block_id.long(),
            "condition_source": condition_source,
            "block_lengths": block_lengths.long(),
            "variant": chosen,
        }
