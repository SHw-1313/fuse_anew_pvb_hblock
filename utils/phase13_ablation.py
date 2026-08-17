"""Explicit registry for the Phase 13 PVB-H-block ablation controls.

The registry is intentionally small and descriptive.  Existing fusion modes
are not aliases for these controls and remain unchanged.  A caller must pass
the selected variant explicitly when constructing ``pvb_shared_hblock``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase13Variant:
    name: str
    adapter_variant: str
    source: str
    pooling: str
    description: str


PHASE13_VARIANTS = (
    Phase13Variant(
        name="pvb_shared_real",
        adapter_variant="real",
        source="detached PVB atom features",
        pooling="Anew variance-preserving atom-to-block pooling",
        description="Matched PVB H-block content from the same record.",
    ),
    Phase13Variant(
        name="pvb_shared_shuffled",
        adapter_variant="shuffled",
        source="detached PVB atom features",
        pooling="within-sample shuffled Anew variance-preserving blocks",
        description="Record-local content permutation control.",
    ),
    Phase13Variant(
        name="pvb_shared_constant",
        adapter_variant="constant",
        source="fixed non-record-specific vector",
        pooling="same adapter and broadcast path",
        description="Capacity and injection control without record-specific content.",
    ),
    Phase13Variant(
        name="pvb_atom_no_pool",
        adapter_variant="atom_no_pool",
        source="detached PVB atom features",
        pooling="no atom-to-block pooling",
        description="Pooling contribution control with the same adapter family.",
    ),
)

PHASE13_ADAPTER_VARIANTS = tuple(item.adapter_variant for item in PHASE13_VARIANTS)
_BY_NAME = {item.name: item for item in PHASE13_VARIANTS}
_BY_ADAPTER = {item.adapter_variant: item for item in PHASE13_VARIANTS}


def get_phase13_variant(name: str) -> Phase13Variant:
    """Return a registered experiment by public name or adapter variant."""

    key = str(name)
    if key in _BY_NAME:
        return _BY_NAME[key]
    if key in _BY_ADAPTER:
        return _BY_ADAPTER[key]
    valid = sorted(set(_BY_NAME) | set(_BY_ADAPTER))
    raise ValueError(f"unknown Phase 13 variant {name!r}; expected one of {valid}")


def phase13_variant_names() -> tuple[str, ...]:
    """Return stable public names for CLI choices and reports."""

    return tuple(item.name for item in PHASE13_VARIANTS)


def phase13_adapter_variant_names() -> tuple[str, ...]:
    """Return the internal adapter values accepted by model construction."""

    return PHASE13_ADAPTER_VARIANTS

