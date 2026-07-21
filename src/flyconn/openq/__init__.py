"""Stage-6 — the THREE open anatomical questions left by ``Figure_Ground_Circuit.pdf``.

The paper verified the VCH->T4/T5->LLPC1->Nod1->DNp26->wing-steering figure-ground
course-control circuit, but verified it only for ONE exemplar configuration: the RIGHT
LLPC1 sheet, the front-to-back (layer-a) T4a channel, split across two animals of
different sex (female FlyWire FAFB for the optic-lobe/DN side, male MCNS for the motor
side). That leaves three falsifiable questions this package answers end-to-end from the
connectome, each returning a neutral verdict
(CONFIRMED / CONFIRMED_WITH_CAVEAT / REFUTED / UNVERIFIABLE):

  * Q1 SINGLE-ANIMAL CLOSURE (``q1_single_animal``) — does the LLPC1->Nod1->DNp26 readout
    reproduce WITHIN the male MCNS alone (the cells all exist there), closing the circuit
    in one animal rather than two?
  * Q2 BILATERAL / 4-DIRECTION GENERALIZATION (``q2_bilateral``) — do the headline numbers
    (VCH gating, retinotopic pooling, Nod1 dominance) hold for the LEFT LLPC1 sheet and the
    other directional channels, or are they specific to the shown right-sheet/layer-a
    exemplar?
  * Q3 ESCAPE-ROUTE CENSUS TO MUSCLE (``q3_escape``) — enumerate the LPLC2/LC4 looming/
    escape descending census (giant fibre DNp01/DNp03/DNp04/DNp06 ...) to the MUSCLE level
    and test whether the escape output is ANATOMICALLY SEPARABLE from the LLPC1 wing-
    steering output (distinct DNs, distinct muscles), as the paper claims.

Each ``run_q*`` is a pure analysis function over the real ``flyconn.paper`` sources
(``make_source`` / ``NeuronMeta`` for FlyWire; ``malecns.client`` for MCNS) — no new data
access is introduced. They emit ``ClaimResult``s via ``flyconn.motif.compare`` so the
verdicts use the same tolerance policy as every prior stage.
"""

from __future__ import annotations

VERSION = "1.0"
