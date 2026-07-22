"""Shared constants for the paper verification: key neuron identities, the canonical
T4/T5 subtype set, the lobula-plate layer<->direction convention, and the paper's
right-hemisphere scope.

Source: Figure_Ground_Circuit.pdf ("Finding Vision-Behavior Circuit through
Connectomics"), FlyWire FAFB v783, male-cns:v1.0. The analysed sheet is the
RIGHT-hemisphere LLPC1 reached by right-side T4a postsynaptic to the LEFT VCH
(Methods / S3 Laterality).
"""

from __future__ import annotations

from dataclasses import dataclass

VERSION = "783"

# The paper studies left VCH (root 720575940627706398; same root the predecessor used)
# whose dendrite+axon lie in the RIGHT optic lobe, gating the right-hemisphere sheet.
VCH_ROOT = 720_575_940_627_706_398

# Scope: the right-hemisphere exemplar (S3). Sheet cells, T4a drivers and LLPC1 are all
# on the right; descending/central readouts are matched by type.
SHEET_SIDE = "right"

# ---------------------------------------------------------------------------
# Bilateral / mirror configuration (Stage 8). Additive: VCH_ROOT and SHEET_SIDE above
# remain the right-side defaults so existing right-hemisphere code/tests are untouched.
# ---------------------------------------------------------------------------
# Centrifugal horizontal cells CROSS hemispheres: a VCH/DCH soma on one side has its
# arbor in the OTHER optic lobe and gates THAT lobe's sheet. So the paper's LEFT-soma VCH
# gates the RIGHT sheet; the RIGHT-soma VCH gates the LEFT sheet.
# Root ids verified against neurons.parquet (both GABA, super_class=visual_centrifugal).
VCH_ROOT_BY_SOMA = {
    "left": 720_575_940_627_706_398,   # == VCH_ROOT; gates the RIGHT sheet (paper)
    "right": 720_575_940_627_502_338,  # gates the LEFT sheet (Stage 8)
}
DCH_ROOT_BY_SOMA = {
    "left": 720_575_940_639_209_956,   # gates the RIGHT sheet
    "right": 720_575_940_631_147_776,  # gates the LEFT sheet
}
# Which SOMA side gates which SHEET side, and the body-side complement map.
GATING_SOMA_FOR_SHEET = {"right": "left", "left": "right"}
OPPOSITE_SIDE = {"right": "left", "left": "right"}


@dataclass(frozen=True)
class SideConfig:
    """One hemisphere's figure-ground scope: which sheet, gated by which crossing VCH/DCH.

    ``RIGHT`` reproduces the paper exactly (the existing hardcoded behaviour); ``LEFT`` is
    the Stage-8 mirror candidate. ``exemplar_t4`` is the layer-a (front-to-back) drive
    channel the paper followed; ``opponent_layer_letter`` is the layer-b channel that drives
    the opponent LPi15 inhibition (and is the negative-control channel for the sheet drive).
    """

    sheet_side: str
    gating_soma_side: str
    vch_root: int
    dch_root: int
    exemplar_t4: str = "T4a"
    opponent_layer_letter: str = "b"

    @classmethod
    def for_side(cls, sheet_side: str) -> "SideConfig":
        soma = GATING_SOMA_FOR_SHEET[sheet_side]
        return cls(
            sheet_side=sheet_side,
            gating_soma_side=soma,
            vch_root=VCH_ROOT_BY_SOMA[soma],
            dch_root=DCH_ROOT_BY_SOMA[soma],
        )


# The paper's right-hemisphere scope (default everywhere) and the left mirror candidate.
RIGHT = SideConfig.for_side("right")
LEFT = SideConfig.for_side("left")

# Canonical T4/T5 subtypes (exact cell_type strings). a/b/c/d index the four
# lobula-plate layers / cardinal directions.
CANONICAL_T4T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

# Lobula-plate layer <-> motion-direction convention (Fischbach & Dittrich 1989;
# Maisak 2013; Shinomiya 2019; the paper's S1 "Directional subtypes"). Layer-a (layer 1)
# = front-to-back; layer-b (layer 2) = back-to-front; c = upward; d = downward. VCH and
# LLPC1 read layer-a; the opponent LPi15 reads layer-b. This convention is taken as
# given (literature), and the verified quantity is the *fraction* of drive from each
# layer, not the convention itself.
T4T5_LAYER = {"a": 1, "b": 2, "c": 3, "d": 4}
LAYER_DIRECTION = {"a": "front_to_back", "b": "back_to_front", "c": "upward", "d": "downward"}

# Per-synapse neurotransmitter probability columns (argmax -> NT name).
NT_COLS = ("gaba", "ach", "glut", "oct", "ser", "da")
NT_TO_CANONICAL = {
    "gaba": "gaba", "ach": "acetylcholine", "glut": "glutamate",
    "oct": "octopamine", "ser": "serotonin", "da": "dopamine",
}

# The eight largest inhibitory T4/T5 targets screened in Table 1 / Fig 3.
INHIBITOR_SCREEN_TYPES = ("VCH", "DCH", "CT1", "LPi15", "LPi14", "Am1", "LT33", "Li14")

# Cholinergic columnar visual-projection "sibling sheets" LLPC1 is compared against
# (Fig S6b / Table S1): the direction-sibling sheets and the looming pathway.
COLUMNAR_PROJECTION_SHEETS = ("LLPC1", "LLPC2", "LLPC3", "LPC1", "LPC2", "LPLC2")

# Nod-type relay cells (S10.1 / Table S10).
NOD_TYPES = ("Nod1", "Nod2", "Nod3", "Nod5", "LPT42_Nod4")

# The seven figure-driven wing-steering DN types (Fig 6 / Table S14).
WING_STEERING_DNS = ("DNa04", "DNbe001", "DNge107", "DNbe005", "DNp26", "DNg32", "DNge094")


def t4t5_subtype_letter(cell_type: str) -> str | None:
    """'T4a'->'a', 'T5b'->'b'; None for non-canonical."""
    if isinstance(cell_type, str) and len(cell_type) == 3 and cell_type[:2] in ("T4", "T5"):
        return cell_type[2]
    return None
