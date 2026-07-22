import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIR = (
    ROOT
    / "data"
    / "benchmarks"
    / "scenarios"
    / "canonical-online-native-pair.v1.json"
)


def test_canonical_online_native_pair_binds_exact_causal_intervention_fixture():
    contract = json.loads(PAIR.read_text(encoding="utf-8"))
    common = contract["common_configuration"]
    assert contract["schema_version"] == "1.0.0"
    assert contract["status"] == "exploratory"
    assert common == {
        "bridge_dt_s": 0.0005,
        "checkpoint_time_s": 0.05,
        "circuit_dt_s": 0.005,
        "duration_s": 0.1,
        "effector_laterality_hypothesis": "raw_l_to_physical_left",
        "figure_initial_world_azimuth_rad": 0.0,
        "figure_velocity_rad_s": math.pi / 6.0,
        "flybody_spawn_height_m": 0.1,
        "ground_velocity_rad_s": 0.0,
        "include_full_cell_state": False,
        "include_retinal_input": True,
        "physics_dt_s": 0.0001,
        "seed": 73,
    }
    assert len(contract["arms"]) == 2
    assert contract["arms"][0]["interventions"] == []
    intervention_arm = contract["arms"][1]
    intervention_path = ROOT / intervention_arm["intervention_fixture_uri"]
    assert hashlib.sha256(intervention_path.read_bytes()).hexdigest() == (
        intervention_arm["intervention_fixture_sha256"]
    )
    intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
    assert intervention == [
        {
            "end_s": 0.1,
            "intervention_id": "raw-l-iv2-silence-60-100ms",
            "mode": "silence",
            "muscle": "iv2",
            "output_scale": 0.0,
            "raw_app_side": "L",
            "start_s": 0.06,
        }
    ]
