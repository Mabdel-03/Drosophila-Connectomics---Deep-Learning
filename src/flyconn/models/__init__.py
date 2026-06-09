"""Stage 3 — connectome-constrained trainable FF/RNN models on MNIST.

One ``ConnectomeNet`` whose connectivity (edge set) and synaptic signs are FIXED from
the FlyWire connectome, but whose edge magnitudes are learned. Six subgraphs
(whole / hemispheres / optic) x {ff_unroll, rnn} x {init-from-data, init-random} are
trained to classify MNIST.

Run via:  python -m flyconn.models.run train --config configs/experiments/<name>.yaml
"""
