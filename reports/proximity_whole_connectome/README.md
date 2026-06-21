# Whole-connectome proximity report

This directory is a lightweight GitHub snapshot of the generated report.

Canonical output directory: `/net/bmc-lab4/data/kellis/users/mabdel03/files/Connectomics/data/v783/experiments/proximity/whole_connectome_lod1_sp250_r500_t2_any/report`

Included files:

- `report.pdf` and `report.tex`
- Figure PDFs under `figures/`
- Lightweight CSV support tables under `tables/`
- `report_manifest.json` with run paths and headline metrics

Excluded files:

- `near_pairs_enriched.parquet`
- exhaustive Parquet appendices
- mesh cache, tile files, reduce buckets, and Slurm logs

Regenerate from the repository root with:

```bash
source slurm/proximity_common.sh
python -m flyconn.experiments.proximity_report \
  --run-name whole_connectome_lod1_sp250_r500_t2_any \
  --compile \
  --snapshot-repo-report
```
