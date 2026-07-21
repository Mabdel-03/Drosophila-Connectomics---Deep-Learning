"""Stage 5 - Paper Verification: end-to-end confirmation of *every* claim in
``Figure_Ground_Circuit.pdf`` (the LLPC1 figure-ground course-control manuscript).

Unlike Stage 4 (``flyconn.motif``), which verified the *predecessor* VCH-T4/T5 report
from the offline dump alone, this stage:

  * verifies the **current** paper (Figs 1-6, Tables 1, S1-S23), 10 claim families A-J;
  * uses the **live FlyWire CAVE API** (materialization v783, ``synapses_nt_v1``, no
    cleft threshold) as the PRIMARY track, which reproduces the paper's absolute counts
    to the digit (offline is a frozen ~65% snapshot); and
  * adds the **male CNS (MaleCNS v1.0)** connectome, from its public bulk feather files,
    for the motor-mapping half (family J).

Layout::

    paper/
      fw_access.py     FlyWire source: LiveCaveFlyWire (primary) | OfflineFlyWire
      geometry.py      synapse-cloud centroids, patch radius, syn-syn distance, scaling
      oracle/          the paper's claims as data, one module per family (A-J + interpretive)
      derive/          derivations, one module per family
      malecns/         MaleCNS access + DN->motor classification (family J)
      ledger.py        assemble all families' ClaimResults -> one JSON + Markdown

The verdict engine and emitters are reused unchanged from ``flyconn.motif.compare``.
"""

from __future__ import annotations
