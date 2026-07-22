"""Stage 4 extraction: pull VCH's synapses (and its reciprocal T4/T5's downstream
synapses) out of the 9.5 GB raw synapse table.

Two memory-mapped streaming passes over flywire_synapses_783.feather:
  Pass 1  -> vch_synapses.parquet            (rows where pre==VCH OR post==VCH)
  (compute the reciprocal T4/T5 set from pass 1 + neurons.parquet)
  Pass 2  -> recip_downstream_synapses.parquet (rows where pre in the reciprocal set)

Both passes are fast (memory-mapped Arrow; a few seconds each). Outputs land on
scratch under data_root()/v783/motif/. Run directly or via slurm/vch_extract.sbatch:

    python -u scripts/vch_extract.py
"""

from __future__ import annotations

import time

from flyconn.io import read_parquet, write_json
from flyconn.motif import vch_config as C
from flyconn.motif import vch_verify as V
from flyconn.motif.synapse_extract import extract_synapses_for_roots
from flyconn.paths import DataPaths


def main() -> None:
    t0 = time.time()
    paths = DataPaths.for_version(C.VERSION)
    feather = paths.raw_file("flywire_synapses_783.feather")
    out = C.motif_dir(C.VERSION)
    print(f"[vch_extract] feather={feather}")
    print(f"[vch_extract] out dir={out}")

    # --- Pass 1: VCH's own synapses ---
    vch_syn_path = out / "vch_synapses.parquet"
    s1 = extract_synapses_for_roots(
        feather, vch_syn_path, pre_roots={C.VCH_ROOT}, post_roots={C.VCH_ROOT}
    )
    print(f"[vch_extract] pass1 kept={s1['n_kept']} rows in {time.time()-t0:.1f}s")

    # --- Compute the reciprocal T4/T5 set (cheap, pandas) ---
    neurons = read_parquet(paths.neurons)
    lookup = V.load_lookup(neurons)
    vch_syn = read_parquet(vch_syn_path)
    ins, outs = V.split_in_out(vch_syn)
    _, _, in_roots = V.derive_t4t5_inputs(ins, lookup, total_input_syn=len(ins))
    _, _, out_roots = V.derive_t4t5_outputs(outs, lookup, total_output_syn=len(outs))
    recip_summary, recip_roots = V.derive_reciprocal(in_roots, out_roots)
    print(f"[vch_extract] reciprocal T4/T5 set size = {recip_summary['n']}")

    recip_path = out / "reciprocal_t4t5_roots.parquet"
    import pandas as pd
    pd.DataFrame({"root_id": recip_roots}).to_parquet(recip_path, index=False)

    # --- Pass 2: downstream synapses of the reciprocal set ---
    down_path = out / "recip_downstream_synapses.parquet"
    s2 = extract_synapses_for_roots(
        feather, down_path, pre_roots=set(recip_roots.tolist()), post_roots=None
    )
    print(f"[vch_extract] pass2 kept={s2['n_kept']} rows in {time.time()-t0:.1f}s")

    write_json(out / "_extract_stats.json", {
        "feather": str(feather),
        "pass1_vch": s1,
        "pass2_downstream": s2,
        "reciprocal": recip_summary,
        "elapsed_s": round(time.time() - t0, 1),
    })
    print(f"[vch_extract] DONE in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
