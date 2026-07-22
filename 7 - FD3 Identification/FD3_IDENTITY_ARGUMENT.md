# Why LPT42_Nod4 is FD3 — the firm argument, and the global case

**Question.** (1) Between **Nod3** and **LPT42_Nod4**, why is LPT42_Nod4 the one that is Egelhaaf's
FD3? (2) Is LPT42_Nod4, *globally* across the whole FlyWire connectome, the most likely FD3?

**Answer in one line.** LPT42_Nod4 is the FlyWire connectome's **unique** best functional/anatomical
correlate of Egelhaaf-1985 FD3 — it is the only cell carrying the full FD3 conjunction (regressive
layer-b + FD1-relative frontal gap + clean heterolateral contra-POF projection + bounded cholinergic
small-field). Against Nod3 the case is decisive (~95/100); as the global best match it is strong
(~88/100); as a literal biological identity it is probable-but-qualified (~62/100), because the
modern cell-typing authority declined the FD one-for-one mapping and the absolute receptive-field
geometry is only corroborated differentially, not confirmed absolutely.

All numbers below are measured on FlyWire FAFB **v783** (offline primary, live cross-check), using the
Family-K / KD machinery (`src/flyconn/paper/derive/k_fd3_lpt42.py`). The literature ground truth is
Egelhaaf, M. (1985) *Biol. Cybern.* 52:195–209, "The FD-cells", read directly (`Egelhaaf FD Cell.pdf`,
FD3 pp. 202–204).

---

## Part 0 — The FD3 signature (Egelhaaf 1985, verbatim discriminators)

Ranked by discriminating power for isolating FD3 from FD1/FD2/FD4:

| # | Property | FD3 value | FD1 | FD2 | FD4 | Discriminates FD3 from |
|---|---|---|---|---|---|---|
| **1** | **Frontal gap** — no excitatory input in the most frontal field | **YES, frontal margin ~20°** — *"the only FD-unit which does not receive excitatory input in the most frontal part of the eye"* (p.202) | no (margin ~−10°) | no (margin −10 to −5°) | no (spans whole eye) | **ALL THREE — unique to FD3** |
| 2 | RF peak (horizontal) | **40–50°** (fronto-lateral) | ~10° | 0–10° | scattered 50–80° | FD1, FD2 (not FD4) |
| 3 | Preferred direction / layer | **regressive / layer-b** | progressive/a | regressive/b | progressive/a | FD1, FD4 (not FD2) |
| 4 | Output projection side | **contralateral POF (heterolateral)** | contra | **ipsilateral POF (homolateral)** | contra | **FD2** |
| 5 | RF half-max width | ~62°±7° | narrow | narrow-frontal | 80–110° | corroborating |
| 6 | Vertical extent | entire vertical field | — | entire | most of vertical | class gate |
| 7 | Contra inhibition | **bidirectional** (either direction) | unidirectional | — | — | FD1 |
| 8 | Small-field selectivity, NT | figure≫ground, cholinergic output | same | same | same | class gate (shared) |

The load-bearing point: **the frontal gap (property 1) is the one feature Egelhaaf explicitly certifies
as unique to FD3.** A cell that has it is separated from FD1, FD2 and FD4 simultaneously. Everything else
either only separates FD3 from a subset (props 2–4, 7) or is a shared class gate that carries zero
within-FD-family discriminating power (props 3, 6, 8 are shared with FD2).

---

## Part 1 — LPT42_Nod4 vs Nod3: the firm argument

Stated as a differential diagnosis — the discriminating findings, ranked, decisive one first.

**The decisive finding: the frontal gap (property 1).** This is Egelhaaf's pathognomonic FD3 sign.
Measured against the FD1 anchor's frontal band:

- **LPT42_Nod4 carries the frontal gap on both sides** — occupancy of the frontal band ~0.001–0.003
  (essentially empty), on both left and right cells independently.
- **Nod3 fills the frontal band** — occupancy ~0.13–0.39. Nod3 **fails** the one necessary feature
  that would make it FD3.

Because a lateral RF *requires* a frontal gap and a frontal RF *precludes* one, this is corroborated by
its retinotopic dual:

**Corroborating finding — RF lateralization (property 2).** RF centroid offset from the FD1 anchor:
**LPT42_Nod4 = +12.7 columns lateral** (fronto-lateral, consistent with the 40–50° peak);
**Nod3 = +3.1 columns** (barely lateral — sitting in the frontal FD1/FD2 cluster). The two independent
RF measurements agree exactly.

**Corroborating finding — projection side (property 4).** FD3 is a *clean heterolateral* output whose
axon crosses to the contralateral POF. **LPT42_Nod4 projects 90.3% contralateral (offline) / 80.5%
(live)** — a clean crossing. **Nod3 is 44.7% / 39.2% — bilateral/mixed**, which is *structurally
incompatible* with FD3's defining heterolateral axon. (The annotation-free spatial contralaterality,
which does not depend on partner side-labels, is 91.3% for LPT42_Nod4 — confirming the number is real,
not an annotation artifact.)

**The constructive tie-breaker.** Scoring each candidate against all four FD signatures
(`FD_SIGNATURES`: FD3 = {layer-b, lateral-gap, heterolateral}), **LPT42_Nod4 best-matches FD3 at 3/3
axes**, while **Nod3's own best fit is FD2 (3/3), not FD3 (1/3)**. The right description of Nod3 is not
"an atypical FD3" — it is *a different cell that most resembles FD2* (though not a clean FD2 either;
it is intermediate).

**The one axis where Nod3 keeps pace, and why it doesn't rescue it.** Nod3 ties LPT42_Nod4 on
regressive/layer-b tuning (97.7% vs 98.6%) and cholinergic output (0.878 vs 0.894). But these are the
**class gates shared by every regressive FD cell**, including FD2 — they have zero power to separate
FD3 from FD2. A tie on the family-wide gates cannot overturn a failure on the Egelhaaf-certified-unique
frontal gap. **Verdict: LPT42_Nod4 is FD3; Nod3 is excluded (confidence ~95/100).**

---

## Part 2 — Is LPT42_Nod4 the global best FD3 match?

**Method.** Two independent global scans of the whole output-cell universe (all visual_projection /
visual_centrifugal / lobula-plate-tangential cell types, ~424–680 types), each cell scored against the
full FD3 conjunction: (a) dominant lobula-plate **layer-b** (regressive), (b) **frontal gap** relative
to FD1, (c) **heterolateral** (contra output ≥ 70%), (d) **bounded cholinergic small-field**. No
hand-picked candidate list — the funnel is applied to every type.

**Result: LPT42_Nod4 is the *unique* 4/4 conjunction hit in the connectome.** It remains the unique hit
even under just the three core gates restricted to low-copy (individually-identifiable) types. Every
same-class confusor is eliminated on an independent axis:

| Rival | Why it is not FD3 |
|---|---|
| **Nod3** | 44.7% contra (bilateral, not heterolateral); no frontal gap → FD2-shaped |
| **LPT21** | 1.5% contra (ipsilateral/homolateral); lateral dendrite → **FD2** |
| **H2** | 38% contra (not heterolateral); and it *feeds* the VCH/DCH wide-field inhibitors — the anti-FD (feedback) signature |
| **VS1** | ipsilateral; vertical-system, unclassified motion axis |
| **cML02 / LPT58** | wrong motion axis / not layer-b conjunction |

Note the honest mechanism: the connectome frontal-gap *feature* alone is **not** unique (H2, VS1, LPT58
also pass it, because it fires on any diffuse large-field input, not only a true small-field FD RF), so
the **contra ≥ 70% heterolaterality gate does the decisive separating work**, and it is only
LPT42_Nod4 that satisfies the *full conjunction*. This is well-managed — every rival that passes one
gate fails another independent one — but it means the global uniqueness rests on the conjunction, not on
any single "magic" feature. **Confidence LPT42_Nod4 is the global best FD3 match: ~88/100.**

### Independent confirmation — a deterministic funnel over all 8,806 connectome cell types

A second, completely independent scan (fast edge-table pass over the whole connectome, not the
agent-driven rescan above) reproduces the uniqueness cleanly. Of **8,806 cell types**: **288** have a
resolvable T4/T5 motion layer → **87** are dominant layer-b (regressive) → **11** are also
heterolateral (contra ≥ 70%). Applying the FD-output definition (**excitatory/cholinergic + low-copy /
individually-identifiable, n ≤ 6**) leaves exactly **two**: **LPT42_Nod4** and **MeMe_e13** — and
MeMe_e13 is ruled out decisively:

| Survivor | Why it is / isn't FD3 |
|---|---|
| **LPT42_Nod4** | 98.6% layer-b (cleanly regressive), 90.3% contra, cholinergic, targets central-brain PLP/WED (a true heterolateral output), bilateral frontal gap → **FD3** |
| MeMe_e13 | `super_class=optic`, a **medulla→medulla intrinsic** cell whose top targets are all medulla-intrinsic (Pm10, C2, Li21) — projects *within the optic lobe*, not a lobula-plate output; only 41.5% layer-b (weak plurality, not cleanly regressive); frontal gap not bilateral → **not an FD output** |

The other 9 layer-b heterolateral cells are eliminated by neurotransmitter alone — they are
**GABAergic** wide-field / feedback cells (H1 99.3% layer-b but GABA; CT1 the giant amacrine; Li31,
mALC3/5, LT33) or dopaminergic descending neurons (DNc01/02) — i.e. *inhibitory* elements, not
Egelhaaf's excitatory FD outputs. Only H1 and LPT42_Nod4 are even cleanly layer-b, and H1 is GABAergic.
**LPT42_Nod4 is therefore the unique cell in the connectome that is a low-copy, cholinergic, cleanly
regressive (layer-b), heterolateral lobula-plate output element** — the exact conjunction that defines
an FD3-type figure detector. (Scan artifacts: `global_fd3_scan_*.csv`.)

---

## Part 3 — Residual caveats, ranked by materiality

These are stated plainly rather than smoothed over; none overturns the ranking above, but they bound how
strongly "FD3" can be asserted as a literal identity.

1. **[most material] "= FD3" is not literature-endorsed.** The modern FlyWire/FAFB cell-typing authority
   that *named* these Nod cells (Nern et al. optic-lobe typing; eLife 93659 / bioRxiv 2023.10.16.562634)
   explicitly declined the FD mapping: the FD cells are *"almost certainly homologous to some of the
   cells in the Noduli group, but one-for-one matches were hard to establish… we did not match the cells
   named as FD1/2/3/4."* LPT42_Nod4 carries `hemibrain_type=None, cell_class=None, fbbt_id=None`. Both
   "LPT42_Nod4 = FD3" and the anchor "Nod1 = FD1" are the pipeline's own derived inferences, not
   published identities.

2. **[material] Anchor circularity.** The RF axis (sign convention, frontal-band reference, azimuth
   calibration) is pinned to the assumed anchor Nod1 = FD1 — `assert_frontal_anchor` *enforces* rather
   than *tests* it. That anchor is measurably mis-calibrated: it places FD1 at ~35–48° when Egelhaaf
   states ~10° (a ~25–38° error at the reference point). **The Nod3 differential result survives this**
   (the frontal gap is anchor-band-independent under a fixed absolute band), but the absolute calibration
   does not.

3. **[material] Absolute RF geometry does not positively confirm FD3.** LPT42_Nod4's raw RF peak sits at
   ~100–106° (~86° even after re-anchoring FD1 to 10°) — roughly 2× FD3's stated 40–50°, reaching toward
   the caudal pole. On absolute geometry alone this reads *at least* as FD4-shaped as FD3-shaped. The
   identification never rested on absolute RF (it rests on the conjunction and the FD1-differential), but
   the RF axis only *corroborates* FD3; it never independently confirms the 40–50° peak.

4. **[moderate] The frontal-gap discriminator is degenerate in the connectome.** Because the feature
   fires on diffuse large-field input, the load-bearing separation collapses onto the single `contra ≥
   70%` cut — where the nearest rivals (H2 38%, Nod3 45%) sit ~25–45 pp below the line. Robust, but a
   single-axis, single-threshold decision at the crux.

5. **[minor] Medial-dendrite discriminator is fragile.** Frame-fair within-lobula-plate fraction is 0.38
   (mid-LOP, FD3-consistent) vs LPT21/FD2's 0.70 (lateral) — a modest ordering that excludes FD2 in
   conjunction but cannot carry the call alone.

6. **[minor / unclosed scope] The ~77k optic-intrinsic cells were not exhaustively screened** as output
   candidates. Excluding them as inputs-not-outputs is defensible but not proven — the only reason global
   uniqueness is not rated higher.

---

## Bottom line

> **LPT42_Nod4 is the FlyWire connectome's unique best functional/anatomical correlate of Egelhaaf's
> FD3** — regressive (layer-b), heterolaterally contra-POF-projecting, cholinergic, small-field, and
> uniquely carrying the FD1-relative frontal gap that excludes Nod3 (which is FD2-shaped, not FD3). This
> is a **derived correlate** resting on the assumed Nod1 = FD1 anchor and on connectome-measured
> physiology rather than on FD3's absolute receptive-field geometry — not a literature-endorsed
> one-for-one identity.
>
> - **FD3 over Nod3: firm (~95/100).** Decisive and multiply-redundant (frontal gap + lateralization +
>   heterolaterality + constructive best-fit all agree); every result survives the anchor caveat because
>   the gap is anchor-band-independent; Nod3's own best fit is FD2.
> - **Global best FD3 match: strong (~88/100).** Unique 4/4 conjunction hit; every rival eliminated on an
>   independent axis. Docked only for the un-screened optic-intrinsic universe.
> - **Is biologically FD3: probable-but-qualified (~62/100).** The conjunction is genuinely FD3-shaped and
>   beats every alternative, but the literature refuses the mapping, the anchor is assumed and
>   mis-calibrated, and the absolute RF geometry mildly favors FD4.

*Method note: verified via the Family-K/KD pipeline plus a 9-agent adversarial audit that attacked the
identification from the receptive-field, morphology, global-candidate, and literature/taxonomic angles —
all four attacks concluded the identification survives, while surfacing the caveats above. A deterministic
global funnel over all output cell types is included as `global_fd3_scan.*` for independent confirmation.*
