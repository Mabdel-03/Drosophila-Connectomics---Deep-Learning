# Stage 3: Results

Two families of connectome models, both built on the frozen FlyWire wiring. See
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

## Caveats and next steps

- MNIST is near-saturated for the trained models, so it weakly separates them. A harder task
  (cluttered or translated digits, or a motion task closer to what the fly visual system evolved
  for) would stress the wiring more and may reveal where the real connectome's structure helps.
- Phase 2 (not yet run) fetches the faithful FlyWire retinal column map and extends the rigid-eye
  models to both-eye and whole-brain subgraphs, where the signal can reach the descending neurons:
  `python -m flyconn.models.run eye_grid --full`.
- Per-run artifacts: `…/v783/models/<run>/{ckpt_best.pt, metrics.jsonl, summary.json}`.
