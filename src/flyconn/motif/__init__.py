"""Stage 4 - Motif Search: independent verification of the VCH-T4/T5 report.

Re-derives every quantitative claim in vch_t4t5_report.pdf from the offline FlyWire
v783 data (raw synapse table = API analog; derived proofread graph = secondary track)
and renders pass/fail verdicts. See `4 - Motif Search/VERIFICATION.md` for output and
`scripts/vch_extract.py` / `scripts/vch_verify.py` for the runnable pipeline.
"""

from __future__ import annotations

from . import compare, downstream, figures, synapse_extract, vch_config, vch_verify  # noqa: F401
