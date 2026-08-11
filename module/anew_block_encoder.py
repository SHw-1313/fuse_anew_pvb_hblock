"""Faithful Anew atom-to-block encoder for the first fusion milestone.

The encoding sequence follows ``CondIterAutoEncoderEdge.encode`` from
AnewOmni's ``models/IterVAE/model_edge.py``:

``BlockEmbedding -> EPT -> H_atom/X_atom -> block pooling``.

This module accepts explicit integer metadata.  It never reconstructs block
membership from coordinates.  ``X_atom`` and ``X_block`` are diagnostic
outputs; the PVB bridge continues to use ``x0`` until a later, separately
gated coordinate experiment.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch_scatter import scatter_mean

from third_party.anewomni.models.modules.EPT.ept import XTransEncoderAct
from third_party.anewomni.models.modules.GET.tools import knn_edges
from third_party.anewomni.models.modules.nn import BlockEmbedding
from third_party.anewomni.utils.gnn_utils import std_conserve_scatter_mean
from utils.bio_utils import ATOM_TYPE, NUM_ATOM_TYPE, RES_TYPE_3


# Verified against AnewOmni/data/bioparse/const.py and
# AnewOmni/data/bioparse/vocab.py at source commit 926e99818ea18cf9d9b2064ce0319fe691b7a1f1.
ANEW_AA_ORDER = (
    "GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "HIS",
    "ASN", "GLU", "LYS", "GLN", "MET", "ARG", "SER", "THR", "CYS", "PRO",
)
ANEW_NUM_ATOM_TYPES = 119  # dummy + the 118-entry periodic table
ANEW_NUM_BLOCK_TYPES = 437  # verified from VOCAB.get_num_block_type()

if tuple(RES_TYPE_3) != ANEW_AA_ORDER:
    raise RuntimeError(
        "PVB and Anew amino-acid block orders differ; refusing an implicit vocabulary mapping"
    )
if len(ATOM_TYPE) != ANEW_NUM_ATOM_TYPES - 1 or NUM_ATOM_TYPE != len(ATOM_TYPE):
    raise RuntimeError("PVB/Anew periodic-table vocabulary sizes differ")


def map_pvb_atom_types(atom_type: torch.Tensor) -> torch.Tensor:
    """Map PVB zero-based periodic-table IDs to Anew IDs with its dummy offset."""

    atom_type = atom_type.long()
    if atom_type.numel() and (atom_type.min() < 0 or atom_type.max() >= len(ATOM_TYPE)):
        raise ValueError("PVB atom IDs are outside the verified periodic-table vocabulary")
    return atom_type + 1


def map_pvb_protein_block_types(block_type: torch.Tensor) -> torch.Tensor:
    """Map PVB residue block IDs to Anew's verified amino-acid IDs.

    Element-level PVB blocks identify molecular atoms and are intentionally
    rejected here; they are not semantically interchangeable with Anew's
    learned fragment vocabulary.
    """

    block_type = block_type.long()
    residue_offset = len(ATOM_TYPE)
    if block_type.numel() == 0:
        return block_type
    residue_mask = (block_type >= residue_offset) & (
        block_type < residue_offset + len(ANEW_AA_ORDER)
    )
    if not bool(residue_mask.all()):
        bad = block_type[~residue_mask].detach().cpu().tolist()
        raise ValueError(
            "Anew block fusion is protein-only: unsupported PVB molecular/element "
            f"block IDs {bad}; no arbitrary ligand offset mapping is permitted"
        )
    return block_type - residue_offset + 1


def _validate_global_metadata(
    atom_block_id: torch.Tensor,
    block_type: torch.Tensor,
    block_batch: torch.Tensor,
    block_lengths: torch.Tensor,
    num_atoms: int,
) -> None:
    if atom_block_id.ndim != 1 or atom_block_id.shape[0] != num_atoms:
        raise ValueError("atom_block_id must have shape [N_atom]")
    if block_type.ndim != 1 or block_batch.ndim != 1 or block_lengths.ndim != 1:
        raise ValueError("block_type, block_batch, and block_lengths must be one-dimensional")
    num_blocks = block_type.shape[0]
    if block_batch.shape[0] != num_blocks or block_lengths.shape[0] != num_blocks:
        raise ValueError("block metadata arrays must have the same number of blocks")
    if num_atoms == 0 or num_blocks == 0:
        raise ValueError("AnewBlockEncoder does not accept an empty graph")
    if atom_block_id.min() < 0 or atom_block_id.max() >= num_blocks:
        raise ValueError("atom_block_id contains an out-of-range global block ID")
    if torch.any(atom_block_id[1:] < atom_block_id[:-1]):
        raise ValueError("atom_block_id must be sorted so each block is contiguous")
    expected_ids = torch.arange(num_blocks, device=atom_block_id.device)
    if not torch.equal(torch.unique(atom_block_id), expected_ids):
        raise ValueError("atom_block_id must contain contiguous global IDs")
    observed_lengths = torch.bincount(atom_block_id, minlength=num_blocks)
    if not torch.equal(observed_lengths, block_lengths.long()):
        raise ValueError("block_lengths do not match atom_block_id")
    if block_batch.min() < 0 or torch.any(block_batch[1:] < block_batch[:-1]):
        raise ValueError("block_batch must be sorted by sample")
    atom_batch = block_batch[atom_block_id]
    if atom_batch.numel() and torch.any(atom_batch[1:] < atom_batch[:-1]):
        raise ValueError("block metadata would mix samples within the atom sequence")


class AnewBlockEncoder(nn.Module):
    """Anew's BlockEmbedding/EPT path returning atom and pooled block states."""

    def __init__(
        self,
        hidden_size: int = 512,
        ffn_size: int = 512,
        edge_size: int = 64,
        n_rbf: int = 64,
        cutoff: float = 10.0,
        n_layers: int = 6,
        n_head: int = 8,
        k_neighbors: int = 9,
        sparse_k: Optional[int] = 3,
        efficient: bool = False,
        vector_act: str = "layernorm",
        num_block_type: int = ANEW_NUM_BLOCK_TYPES,
        num_atom_type: int = ANEW_NUM_ATOM_TYPES,
    ) -> None:
        super().__init__()
        if num_atom_type < ANEW_NUM_ATOM_TYPES:
            raise ValueError(f"num_atom_type must be at least {ANEW_NUM_ATOM_TYPES}")
        if num_block_type < max(ANEW_AA_ORDER.__len__() + 1, 21):
            raise ValueError("num_block_type must include Anew's dummy and 20 amino-acid types")

        self.hidden_size = hidden_size
        self.edge_size = edge_size
        self.k_neighbors = k_neighbors
        self.embedding = BlockEmbedding(num_block_type, num_atom_type, hidden_size)
        # Keep Anew's explicit embedding-to-hidden projection so official
        # checkpoints can migrate without silently dropping this parameter.
        self.enc_embed2hidden = nn.Linear(hidden_size, hidden_size)
        self.block_edge_embedding = nn.Embedding(3, edge_size)
        self.atom_edge_embedding = nn.Embedding(5, edge_size)
        self.encoder = XTransEncoderAct(
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            n_rbf=n_rbf,
            cutoff=cutoff,
            edge_size=edge_size,
            n_layers=n_layers,
            n_head=n_head,
            pre_norm=True,
            use_edge_feat=True,
            sparse_k=sparse_k,
            efficient=efficient,
            vector_act=vector_act,
        )
        # Anew's reference predicts an isotropic coordinate log variance from
        # the pooled scalar state. It is diagnostic in milestone one.
        self.Wx_log_var = nn.Linear(hidden_size, 1)

    def _block_edges(
        self,
        x_atom: torch.Tensor,
        atom_block_id: torch.Tensor,
        block_batch: torch.Tensor,
        block_edge_index: Optional[torch.Tensor],
        block_edge_type: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_edge_index is None:
            block_edge_index = knn_edges(
                atom_block_id,
                block_batch,
                x_atom.unsqueeze(1),
                self.k_neighbors,
            )
            block_edge_type = torch.zeros(
                block_edge_index.shape[1], dtype=torch.long, device=x_atom.device
            )
        else:
            block_edge_index = block_edge_index.long().to(x_atom.device)
            if block_edge_index.ndim != 2 or block_edge_index.shape[0] != 2:
                raise ValueError("block_edge_index must have shape [2, E_block]")
            if block_edge_type is None:
                block_edge_type = torch.zeros(
                    block_edge_index.shape[1], dtype=torch.long, device=x_atom.device
                )
            else:
                block_edge_type = block_edge_type.long().to(x_atom.device)
        if block_edge_type.shape[0] != block_edge_index.shape[1]:
            raise ValueError("block_edge_type must have one entry per block edge")
        if block_edge_type.numel() and (block_edge_type.min() < 0 or block_edge_type.max() >= 3):
            raise ValueError("block edge types must be in Anew's [0, 1, 2] vocabulary")
        return block_edge_index, self.block_edge_embedding(block_edge_type)

    def forward(
        self,
        x_atom: torch.Tensor,
        atom_type: torch.Tensor,
        block_type: torch.Tensor,
        atom_block_id: torch.Tensor,
        block_batch: torch.Tensor,
        block_lengths: torch.Tensor,
        bond_index: Optional[torch.Tensor] = None,
        bond_type: Optional[torch.Tensor] = None,
        block_edge_index: Optional[torch.Tensor] = None,
        block_edge_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode one collated batch using explicit atom/block metadata."""

        if x_atom.ndim != 2 or x_atom.shape[-1] != 3:
            raise ValueError("x_atom must have shape [N_atom, 3]")
        device = x_atom.device
        atom_type = atom_type.long().to(device)
        block_type = block_type.long().to(device)
        atom_block_id = atom_block_id.long().to(device)
        block_batch = block_batch.long().to(device)
        block_lengths = block_lengths.long().to(device)
        _validate_global_metadata(
            atom_block_id, block_type, block_batch, block_lengths, x_atom.shape[0]
        )

        anew_atom_type = map_pvb_atom_types(atom_type)
        anew_block_type = map_pvb_protein_block_types(block_type)
        H = self.embedding(anew_block_type, anew_atom_type, atom_block_id)
        H = self.enc_embed2hidden(H)
        edges, edge_attr = self._block_edges(
            x_atom, atom_block_id, block_batch, block_edge_index, block_edge_type
        )

        topo_edges = None
        topo_edge_attr = None
        if bond_index is not None and bond_index.numel() > 0:
            topo_edges = bond_index.long().to(device)
            if topo_edges.ndim != 2 or topo_edges.shape[0] != 2:
                raise ValueError("bond_index must have shape [2, E_bond]")
            if bond_type is None:
                # PVB's topology stores connectivity but no bond order. The
                # protein path therefore uses Anew's single-bond embedding.
                bond_type = torch.ones(topo_edges.shape[1], dtype=torch.long, device=device)
            else:
                bond_type = bond_type.long().to(device)
            if bond_type.shape[0] != topo_edges.shape[1]:
                raise ValueError("bond_type must have one entry per bond")
            if bond_type.numel() and (bond_type.min() < 0 or bond_type.max() >= 5):
                raise ValueError("bond_type must be in Anew's [0, 1, 2, 3, 4] vocabulary")
            topo_edge_attr = self.atom_edge_embedding(bond_type)

        H_atom, X_atom = self.encoder(
            H,
            x_atom,
            atom_block_id,
            block_batch,
            edges,
            edge_attr,
            topo_edges=topo_edges,
            topo_edge_attr=topo_edge_attr,
            attn_mask=None,
        )
        H_block = std_conserve_scatter_mean(H_atom, atom_block_id, dim=0)
        X_block = scatter_mean(X_atom, atom_block_id, dim=0, dim_size=block_type.shape[0])
        log_var_block = self.Wx_log_var(H_block)

        return {
            "H_atom": H_atom,
            "X_atom": X_atom,
            "H_block": H_block,
            "X_block": X_block,
            "log_var_block": log_var_block,
            "atom_block_id": atom_block_id,
            "block_batch": block_batch,
            "block_lengths": block_lengths,
        }
