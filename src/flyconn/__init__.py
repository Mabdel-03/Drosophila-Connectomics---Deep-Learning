"""flyconn — FlyWire Drosophila connectome -> model-ready artifacts and models.

Stage 1 (``flyconn.data_prep``) downloads the FlyWire FAFB v783 connectome and
turns it into a canonical neuron table plus signed sparse adjacency matrices that
downstream stages (exploration, modeling, motif search) consume.
"""

__version__ = "0.1.0"
