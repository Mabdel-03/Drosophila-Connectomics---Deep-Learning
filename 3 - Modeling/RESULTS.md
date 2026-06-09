# Stage 3 — Results (24-run MNIST grid)

All 24 connectome-constrained models (6 subgraphs × {`ff_unroll`,`rnn`} × {`from_data`,
`random`}) trained to completion. See [`MODELS.md`](MODELS.md) for the architectures and the
training/testing protocol. Regenerate with `python -m flyconn.models.run aggregate`.

## Test accuracy (every run)

| subgraph | ff · from_data | ff · random | rnn · from_data | rnn · random |
|---|---:|---:|---:|---:|
| `optic` | 0.9741 | 0.9743 | 0.9710 | **0.9761** |
| `optic_left` | 0.9692 | 0.9724 | 0.9700 | **0.9746** |
| `optic_right` | 0.9600 | 0.9761 | 0.9734 | **0.9775** |
| `whole` | 0.9718 | 0.9708 | 0.9731 | **0.9751** |
| `whole_left` | 0.9690 | 0.9706 | 0.9720 | **0.9747** |
| `whole_right` | 0.9671 | 0.9745 | 0.9724 | **0.9769** |

Overall: **min 0.9600, max 0.9775, mean 0.9724.** Best run: **`optic_right / rnn / random` =
97.75%**.

## Findings

1. **Connectome-constrained networks solve MNIST.** Freezing the real connectivity and the
   excitatory/inhibitory signs — learning only edge magnitudes plus a small input encoder and
   linear head — still reaches ~97% on MNIST across all six subgraphs.

2. **Amplitude-matched random init slightly beats data init (11 of 12 subgraph×flavor pairs).**

   | init | mean test acc |
   |---|---:|
   | `random` (log-normal matched) | **0.9745** |
   | `from_data` (real synapse counts) | 0.9703 |

   The gap is small (~0.4 pts mean, up to 1.6) but consistent. Interpretation: the **graph
   structure + signs** carry the useful inductive bias; the *specific* real synapse magnitudes
   are not a helpful starting point for MNIST (they're a mild optimization handicap vs. random
   amplitudes from the same distribution).

3. **RNN ≥ feedforward.** The recurrent flavor (persistent input + leaky feedback, full BPTT)
   edged out the unrolled-feedforward flavor in most pairs.

4. **Size barely matters for this task.** Whole-brain (139k nodes) ≈ optic (~97k) ≈ single
   hemisphere (~48k), all ~97% — MNIST is easy enough that one hemisphere's worth of fly
   visual wiring suffices.

## Caveats / next steps

- MNIST is near-saturated for all variants, so it weakly separates models; a harder task
  (e.g. CIFAR, cluttered/translated digits, or a motion task) would stress the wiring more and
  may reveal where the real connectome's structure helps.
- The `from_data` vs `random` gap could be probed further (longer training, LR sweep, or
  checking whether it's an optimization vs. generalization effect).
- Per-run artifacts: `…/v783/models/<run>/{ckpt_best.pt, metrics.jsonl, summary.json}`.
