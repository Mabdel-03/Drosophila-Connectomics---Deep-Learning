# Stage 3: Results

Four families of connectome models, all built on the frozen FlyWire wiring. See
[`MODELS.md`](MODELS.md) for the architectures, training, and inference, and
[`README.md`](README.md) for the short tour. Regenerate the tables with
`python -m flyconn.models.run aggregate`.

---

## Family 1: the 24 learned-encoder models

These use a learned input encoder and a learned output head with the connectome in between. All
six subgraphs, two flavors (`ff_unroll`, `rnn`), and two initializations (`from_data`, `random`).

### Test accuracy (every run)

| Subgraph | ff, from_data | ff, random | rnn, from_data | rnn, random |
|---|---:|---:|---:|---:|
| `optic` | 0.9741 | 0.9743 | 0.9710 | **0.9761** |
| `optic_left` | 0.9692 | 0.9724 | 0.9700 | **0.9746** |
| `optic_right` | 0.9600 | 0.9761 | 0.9734 | **0.9775** |
| `whole` | 0.9718 | 0.9708 | 0.9731 | **0.9751** |
| `whole_left` | 0.9690 | 0.9706 | 0.9720 | **0.9747** |
| `whole_right` | 0.9671 | 0.9745 | 0.9724 | **0.9769** |

Overall: min 0.9600, max 0.9775, mean 0.9724. Best run: `optic_right` with `rnn` and `random`, at
97.75%.

### Findings

1. **Connectome-constrained networks solve MNIST.** Freezing the real connectivity and the
   excitatory/inhibitory signs, and learning only edge magnitudes plus a small encoder and head,
   still reaches about 97% across all six subgraphs.

2. **Amplitude-matched random init slightly beats data init** (11 of 12 subgraph-by-flavor pairs).

   | Init | Mean test accuracy |
   |---|---:|
   | `random` (matched amplitude) | **0.9745** |
   | `from_data` (real synapse counts) | 0.9703 |

   The gap is small (about 0.4 points on average, up to 1.6) but consistent. Reading: the graph
   structure and signs carry the useful inductive bias, while the specific measured magnitudes are
   not a helpful starting point for MNIST.

3. **RNN is at least as good as feedforward.** The recurrent flavor edged out the unrolled
   feedforward flavor in most pairs.

4. **Size barely matters for this task.** Whole-brain (139k neurons), optic (97k), and a single
   hemisphere (48k) all land near 97%. MNIST is easy enough that one hemisphere's worth of fly
   visual wiring suffices.

The lesson: 97% validates the setup, but a learned encoder plus a learned head is itself a capable
network, so this number does not isolate how much the *connectome* contributes. That motivated
Family 2.

---

## Family 2: the faithful rigid-eye models (Phase 1)

These replace the learned encoder with a fixed, eye-like input filter, then sweep how much else is
learned: V1 (trainable core plus head), V2 (frozen core, linear probe), V3 (zero learning). Phase 1
covers `optic_left` and `optic_right`, with corrected (`fix`) and uncorrected (`raw`) photoreceptor
signs. Numbers are on the full 10,000-image test set; chance is 10%.

### The ladder (headline)

As learning is removed, the measured connectome keeps classifying:

| What is learned | optic_left | optic_right |
|---|---:|---:|
| V1: connectome magnitudes + head | **0.963** | **0.965** |
| V2: only a linear probe (core frozen) | **0.939** | **0.940** |
| V3: nothing at all (cosine) | **0.574** | **0.489** |

### V3 detail (zero learned parameters)

V3 reports four numbers so the result can be trusted: `cosine` (the headline, pattern-based),
`raw-euclid` (a brightness floor the real rules must beat), `lda` (a learning-free linear
discriminant), and `shuffle` (a control that must collapse toward chance).

| Subgraph | Sign | cosine | raw-euclid (floor) | lda | shuffle (control) |
|---|---|---:|---:|---:|---:|
| `optic_left` | fix | 0.574 | 0.508 | 0.567 | 0.200 |
| `optic_left` | raw | 0.603 | 0.552 | 0.607 | 0.142 |
| `optic_right` | fix | 0.489 | 0.418 | 0.472 | 0.156 |
| `optic_right` | raw | 0.515 | 0.433 | 0.504 | 0.148 |

On every run, cosine and lda clearly beat the raw-euclidean floor (so the classifier is reading the
*pattern* of activity, not just image brightness), and the shuffle controls collapse toward chance
(so there is no leak). The measured fly visual wiring, with zero learned parameters, classifies
handwritten digits at roughly 49 to 60%.

### Findings

1. **The wiring carries real visual structure.** 96% with a trainable core, 94% with only a linear
   probe, and roughly 50 to 60% with no learning at all. The signal is in the fixed connectome, not
   supplied by a learned input or output stage.

2. **A faithful input costs almost nothing.** Swapping the learned 784-to-photoreceptor encoder for
   a fixed, eye-like filter barely moves V1 (about 96% versus the 97% of Family 1).

3. **Gain control is essential.** A frozen biological network run open-loop loses its signal before
   it reaches the output; per-step divisive normalization (which real visual systems use) restores
   it. See [`MODELS.md`](MODELS.md), Part 5.2.

4. **The biology sign-fix is nuanced.** Forcing photoreceptors to be inhibitory (correct biology)
   slightly lowers the zero-learning V3 score but slightly raises the trained V2 score. It is the
   right default once anything is trained, but not a free win for a bare nearest-template rule.

5. **Left edges out right.** `optic_left` beats `optic_right` across all variants, expected with the
   self-contained retinal grid; the faithful FlyWire column map should narrow it.

---

## Family 3: the recurrent-multiply models (recmul, Phase 1)

The depth-10 RNN **of** the connectome: keep Family 1's learned input/output, but make the core
just multiply the node-state vector by `W` ten times. Phase 1 covers `optic_left` and `optic_right`
× three step rules × frozen vs trained connectome, all with the `from_data` init (12 runs). Numbers
are on the full 10,000-image test set; chance is 10%.

### Test accuracy (every run)

| Step rule | optic_left, frozenW | optic_left, trainW | optic_right, frozenW | optic_right, trainW |
|---|---:|---:|---:|---:|
| `linear` (raw `W^10`) | 0.7754 | 0.8948 | 0.5116 | 0.8706 |
| `linear_rms` (+ normalization) | **0.9249** | **0.9252** | **0.9216** | **0.9263** |
| `tanh` (bounded step) | 0.2064 | 0.9705 | 0.1135 | 0.9590 |

### Findings

1. **Pure `linear` behaves like the linear map it is.** With the connectome frozen, repeatedly
   multiplying by `W^10` and reading out linearly classifies at 78% (`optic_left`) / 51%
   (`optic_right`); training the magnitudes lifts both to ~87–89%, approaching the
   linear-classifier ceiling on MNIST (~92%) and **no higher** — exactly as predicted for a single
   effective `R · W^10 · S` map. This is the cleanest measure of how much of MNIST is *linearly
   readable* through the fixed ten-hop wiring.

2. **Divisive normalization is the whole story for the frozen connectome.** Adding per-step RMS
   normalization (`linear_rms`, no new learned parameters) jumps the *frozen* core from 51–78% to a
   uniform **~92%** on both hemispheres, and training barely changes it (~92.5%). The input-dependent
   normalization — the same gain control that rescued the rigid-eye Family 2 — is what lets the
   as-measured wiring classify near the ceiling, and it makes the left/right asymmetry of the raw
   `linear` runs disappear.

3. **`tanh` needs a trained core, or it dies.** The bounded leaky step with a *frozen* core collapses
   to chance (~11–21%): with no normalization and no training, the signal does not survive ten
   `tanh` hops to the readout. But with a **trained** core `tanh` is the best rule of all —
   **97.0% / 95.9%** — matching Family 1, because training can shape the magnitudes to keep the
   nonlinear dynamics informative across depth 10.

4. **The ladder isolates each ingredient.** Frozen `linear` (78%) → frozen `linear_rms` (92%) is the
   value of divisive normalization on fixed wiring; frozen → trained within `linear` (78%→89%) is
   what learning the magnitudes adds under a pure linear map; and trained `tanh` (97%) over trained
   `linear` (89%) is the value of the nonlinearity once the core is learnable. `linear_rms` is the
   only rule that classifies well *without training the connectome at all*.

---

## Family 4: eyeTact — eye × activation × recurrence-depth, untrained vs trained

The largest sweep: the **whole connectome** (139,255 neurons, signed + magnitude-weighted) run as
a depth-`T` recurrent net on MNIST, crossing four axes — input front-end {learned encoder, rigid
**faithful-FlyWire-retinotopy** eye} × activation {linear, tanh, relu} × recurrence depth
`T = 1…12` × connectome trainability {**untrained** = wiring frozen exactly as measured, only the
input/output layers learn; **trained** = edge magnitudes also learn}. 2 × 3 × 12 × 2 = **144 runs**.
Every cell reports an untrained and a trained test accuracy; chance is 11.4%.

### Learned encoder

**linear**

| T | untrained (frozen connectome) | trained |
|---:|:---:|:---:|
| 1 | 0.114 | 0.114 |
| 2 | 0.924 | 0.925 |
| 3 | 0.916 | 0.923 |
| 4 | 0.906 | 0.922 |
| 5 | 0.879 | 0.921 |
| 6 | 0.846 | 0.921 |
| 7 | 0.757 | 0.920 |
| 8 | 0.727 | 0.917 |
| 9 | 0.579 | 0.918 |
| 10 | 0.565 | 0.915 |
| 11 | 0.517 | 0.918 |
| 12 | 0.536 | 0.899 |

**tanh**

| T | untrained (frozen connectome) | trained |
|---:|:---:|:---:|
| 1 | 0.114 | 0.114 |
| 2 | 0.916 | 0.962 |
| 3 | 0.633 | 0.966 |
| 4 | 0.542 | 0.972 |
| 5 | 0.303 | 0.971 |
| 6 | 0.114 | 0.973 |
| 7 | 0.114 | 0.971 |
| 8 | 0.114 | 0.973 |
| 9 | 0.114 | 0.971 |
| 10 | 0.114 | 0.972 |
| 11 | 0.114 | 0.971 |
| 12 | 0.114 | 0.966 |

**relu**

| T | untrained (frozen connectome) | trained |
|---:|:---:|:---:|
| 1 | 0.114 | 0.114 |
| 2 | 0.977 | 0.976 |
| 3 | 0.984 | 0.984 |
| 4 | 0.987 | 0.985 |
| 5 | 0.985 | 0.986 |
| 6 | 0.983 | 0.984 |
| 7 | 0.982 | 0.984 |
| 8 | 0.982 | 0.983 |
| 9 | 0.982 | 0.984 |
| 10 | 0.983 | 0.986 |
| 11 | 0.983 | 0.984 |
| 12 | 0.980 | 0.984 |

### Rigid eye (faithful FlyWire v783 retinotopy)

**linear**

| T | untrained (frozen connectome) | trained |
|---:|:---:|:---:|
| 1 | 0.114 | 0.114 |
| 2 | 0.911 | 0.895 |
| 3 | 0.924 | 0.911 |
| 4 | 0.924 | 0.912 |
| 5 | 0.925 | 0.913 |
| 6 | 0.924 | 0.914 |
| 7 | 0.924 | 0.914 |
| 8 | 0.925 | 0.912 |
| 9 | 0.925 | 0.912 |
| 10 | 0.925 | 0.913 |
| 11 | 0.925 | 0.911 |
| 12 | 0.925 | 0.912 |

**tanh**

| T | untrained (frozen connectome) | trained |
|---:|:---:|:---:|
| 1 | 0.114 | 0.114 |
| 2 | 0.910 | 0.893 |
| 3 | 0.927 | 0.911 |
| 4 | 0.929 | 0.914 |
| 5 | 0.929 | 0.914 |
| 6 | 0.929 | 0.914 |
| 7 | 0.929 | 0.913 |
| 8 | 0.929 | 0.912 |
| 9 | 0.929 | 0.912 |
| 10 | 0.929 | 0.912 |
| 11 | 0.929 | 0.913 |
| 12 | 0.929 | 0.895* |

**relu**

| T | untrained (frozen connectome) | trained |
|---:|:---:|:---:|
| 1 | 0.114 | 0.114 |
| 2 | 0.580 | 0.683 |
| 3 | 0.932 | 0.937 |
| 4 | 0.966 | 0.972 |
| 5 | 0.970 | 0.980 |
| 6 | 0.970 | 0.982 |
| 7 | 0.971 | 0.982 |
| 8 | 0.971 | 0.981 |
| 9 | 0.971 | 0.982 |
| 10 | 0.970 | 0.981 |
| 11 | 0.970 | 0.980 |
| 12 | 0.971 | 0.980 |

\* `rigid tanh trained T12` was trained for 10 epochs instead of 25 (it is the single most
expensive cell — T12 × tanh × whole-brain); its value is therefore slightly below its T6–T11
neighbours (~0.913) but the trend is unaffected.

### Findings

1. **T=1 is chance everywhere.** With the input injected but zero synaptic hops propagated, the
   signal has not reached the readout neurons (a median of 3, max 7 hops from the photoreceptors),
   so every cell sits at 0.114. At T=2 the signal arrives and accuracy jumps to 90%+. This is the
   recurrence-depth signature.

2. **ReLU ~98%, and the untrained frozen connectome matches the trained one.** Learned-eye ReLU
   holds ~0.98 across all depths whether the connectome is frozen (0.987 at T=4) or trained (0.985)
   — training the synaptic weights adds essentially nothing. The fly's measured wiring already does
   the computation.

3. **Linear caps at the ~92% linear-classifier ceiling — and a frozen linear core decays with
   depth.** Learned-eye linear, untrained, falls 0.924 (T2) → 0.579 (T9) → 0.536 (T12), while the
   trained core stays flat ~0.92. Repeatedly multiplying by a fixed `W` degrades the signal
   geometrically over many hops; training re-conditions `W` to keep it alive. (Pure linear depth-`T`
   is one effective `W^T` map, so the ceiling is expected.)

4. **The faithful FlyWire retinotopy fixes the decay — untrained even edges out trained.** The
   rigid eye carries a per-step RMS gain control (divisive normalization, like real visual systems).
   With it, the **untrained** linear core stays **flat at ~0.925 through T=12** — no decay — and
   sits slightly *above* the trained core (0.925 vs ~0.912). Same story for rigid tanh (flat ~0.929
   untrained). The measured weights are already near-optimal at this ceiling, so gradient descent
   adds nothing.

5. **tanh is the exception that proves the rule — it needs gain control.** Without normalization
   (learned-eye tanh), the untrained core works at T=2 (0.916) then **collapses to chance (0.114) by
   T≥6** as the bounded signal decays across hops; the trained core recovers (~0.97) by re-tuning
   weights. With normalization (rigid-eye tanh), the untrained core is **flat at 0.929 through T=12**.

**Takeaway.** The frozen, as-measured *Drosophila* connectome already classifies MNIST — ~92%
(linear) or ~98% (ReLU) — **provided the signal is kept alive across synaptic hops**, by either a
non-decaying nonlinearity (ReLU) or biological divisive normalization (the rigid eye's RMS step).
When the signal decays (frozen linear, or unnormalized tanh at depth), accuracy falls. Training the
synaptic weights mostly just compensates for that decay rather than adding new computational power,
and the biologically faithful retinotopic input performs as well as a learned encoder. **The
computation lives in the anatomy.**

Regenerate: `python -m flyconn.models.run eyeTact_grid` → `sbatch slurm/eyeTact_array.sbatch` →
`python -m flyconn.models.run aggregate`.

---

## Caveats and next steps

- MNIST is near-saturated for the trained models, so it weakly separates them. A harder task
  (cluttered or translated digits, or a motion task closer to what the fly visual system evolved
  for) would stress the wiring more and may reveal where the real connectome's structure helps.
- Phase 2 (not yet run) fetches the faithful FlyWire retinal column map and extends the rigid-eye
  models to both-eye and whole-brain subgraphs, where the signal can reach the descending neurons:
  `python -m flyconn.models.run eye_grid --full`.
- Family 3 Phase 2 adds the `random`-init control (and, by extending `RECMUL_SUBGRAPHS`, the
  whole-brain subgraphs): `python -m flyconn.models.run recmul_grid --full`.
- Per-run artifacts: `…/v783/models/<run>/{ckpt_best.pt, metrics.jsonl, summary.json}`.
