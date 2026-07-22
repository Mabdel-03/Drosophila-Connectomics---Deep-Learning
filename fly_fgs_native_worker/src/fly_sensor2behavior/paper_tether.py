"""Calibrated rigid-tether torque meters for the paper assay.

``PaperFixedLoadCellFlyBodyAdapter`` is authoritative: its root has no free
joint and MuJoCo force/torque sensors measure the interaction at the welded
root origin. ``PaperTetheredFlyBodyAdapter`` retains the older free-root weld
as an equality-row-only validator.  Both retain dynamic wings and FlyBody's
ellipsoid fluid model; visual stimulus geometry never enters MuJoCo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .flight.types import AerodynamicWrench, RigidBodyState, WingKinematics
from .flybody_adapter import (
    FlyBodyDependencyError,
    FlyBodyPhysicsAdapter,
    FlyBodyWorkerConfig,
    WingAxisTorqueMap,
    WingTorqueMapper,
    _FLYGYM_FORCE_TO_NEWTON,
    _FLYGYM_TORQUE_TO_NEWTON_METRE,
    build_flybody_simulation,
)


PAPER_TETHER_SCHEMA_VERSION = "3.0.0"
PAPER_TORQUE_METER_MODES = (
    "fixed-load-cell",
    "equality-reaction",
    "comparison",
)
PAPER_TETHER_SOLVER_SUBSTEPS = 4
PAPER_ENGINE_AXES = {
    "x": "body_forward",
    "y": "body_left",
    "z": "up",
    "positive_engine_z_yaw": "attempted_left_turn",
    "positive_paper_yaw": "attempted_right_turn",
}
_PAPER_YAW_AXIS_ENGINE = np.asarray((0.0, 0.0, 1.0), dtype=float)
PAPER_WING_SIDE_ORDER = ("left", "right")


def torque_nm_to_dyne_cm(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("torque must be finite")
    return value * 1.0e7


def torque_dyne_cm_to_nm(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("torque must be finite")
    return value * 1.0e-7


def _nan3() -> np.ndarray:
    return np.full(3, np.nan, dtype=float)


def _nan6() -> np.ndarray:
    return np.full(6, np.nan, dtype=float)


def _nan2() -> np.ndarray:
    return np.full(2, np.nan, dtype=float)


def _nan23() -> np.ndarray:
    return np.full((2, 3), np.nan, dtype=float)


def _nan233() -> np.ndarray:
    return np.full((2, 3, 3), np.nan, dtype=float)


def _identity3() -> np.ndarray:
    return np.eye(3, dtype=float)


@dataclass(frozen=True)
class PaperTetherTorqueSample:
    """One versioned native torque-meter sample in public SI units.

    The six original fields remain intact.  ``fly_generated_moment`` is kept
    solely for compatibility and is explicitly the negative support reaction,
    not an independent estimator.
    """

    support_force_engine_n: np.ndarray
    support_moment_engine_n_m: np.ndarray
    fly_generated_moment_engine_n_m: np.ndarray
    root_fluid_moment_engine_n_m: np.ndarray
    reported_yaw_torque_n_m: float
    reported_yaw_torque_dyne_cm: float
    meter_mode: str = "equality-reaction"
    timestamp_s: float = 0.0
    support_load_cell_force_engine_n: np.ndarray = field(default_factory=_nan3)
    support_load_cell_moment_engine_n_m: np.ndarray = field(default_factory=_nan3)
    support_on_fly_yaw_reaction_engine_n_m: float = np.nan
    attempted_fly_yaw_moment_from_support_engine_n_m: float = np.nan
    equality_only_yaw_reaction_engine_n_m: float = np.nan
    yaw_balance_residual_n_m: float = np.nan
    load_cell_position_engine_m: np.ndarray = field(default_factory=_nan3)
    load_cell_orientation_site_to_engine: np.ndarray = field(default_factory=_identity3)
    load_cell_axis_engine: np.ndarray = field(
        default_factory=lambda: _PAPER_YAW_AXIS_ENGINE.copy()
    )
    actuator_torque_n_m: np.ndarray = field(default_factory=_nan6)
    actuator_clipped: bool = False
    wing_load_cell_position_engine_m: np.ndarray = field(default_factory=_nan23)
    wing_load_cell_orientation_site_to_engine: np.ndarray = field(
        default_factory=_nan233
    )
    wing_parent_on_child_force_engine_n: np.ndarray = field(default_factory=_nan23)
    wing_parent_on_child_moment_at_hinge_engine_n_m: np.ndarray = field(
        default_factory=_nan23
    )
    wing_on_thorax_force_engine_n: np.ndarray = field(default_factory=_nan23)
    wing_on_thorax_moment_at_hinge_engine_n_m: np.ndarray = field(
        default_factory=_nan23
    )
    wing_on_thorax_moment_at_tether_engine_n_m: np.ndarray = field(
        default_factory=_nan23
    )
    wing_reported_yaw_torque_n_m: np.ndarray = field(default_factory=_nan2)
    wing_sum_reported_yaw_torque_n_m: float = np.nan
    nonwing_reported_yaw_torque_residual_n_m: float = np.nan
    wing_aerodynamic_moment_at_tether_engine_n_m: np.ndarray = field(
        default_factory=_nan23
    )
    wing_aerodynamic_reported_yaw_torque_n_m: np.ndarray = field(
        default_factory=_nan2
    )
    full_aerodynamic_reported_yaw_torque_n_m: float = np.nan
    background_aerodynamic_reported_yaw_torque_n_m: float = np.nan
    aerodynamic_closure_residual_n_m: float = np.nan

    def __post_init__(self) -> None:
        if self.meter_mode not in PAPER_TORQUE_METER_MODES:
            raise ValueError("unsupported torque meter mode")
        finite_vectors = (
            "support_force_engine_n",
            "support_moment_engine_n_m",
            "fly_generated_moment_engine_n_m",
        )
        optional_vectors = (
            "root_fluid_moment_engine_n_m",
            "support_load_cell_force_engine_n",
            "support_load_cell_moment_engine_n_m",
            "load_cell_position_engine_m",
            "actuator_torque_n_m",
        )
        for name in finite_vectors + optional_vectors:
            value = np.asarray(getattr(self, name), dtype=float)
            expected_shape = (6,) if name == "actuator_torque_n_m" else (3,)
            if value.shape != expected_shape:
                raise ValueError("{} has the wrong shape".format(name))
            if name in finite_vectors and not np.all(np.isfinite(value)):
                raise ValueError("{} must be finite".format(name))
            if np.any(np.isinf(value)):
                raise ValueError("{} must not contain infinity".format(name))
            object.__setattr__(self, name, value.copy())
        shaped_optional_vectors = {
            "wing_load_cell_position_engine_m": (2, 3),
            "wing_load_cell_orientation_site_to_engine": (2, 3, 3),
            "wing_parent_on_child_force_engine_n": (2, 3),
            "wing_parent_on_child_moment_at_hinge_engine_n_m": (2, 3),
            "wing_on_thorax_force_engine_n": (2, 3),
            "wing_on_thorax_moment_at_hinge_engine_n_m": (2, 3),
            "wing_on_thorax_moment_at_tether_engine_n_m": (2, 3),
            "wing_reported_yaw_torque_n_m": (2,),
            "wing_aerodynamic_moment_at_tether_engine_n_m": (2, 3),
            "wing_aerodynamic_reported_yaw_torque_n_m": (2,),
        }
        for name, expected_shape in shaped_optional_vectors.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != expected_shape:
                raise ValueError("{} has the wrong shape".format(name))
            if np.any(np.isinf(value)):
                raise ValueError("{} must not contain infinity".format(name))
            object.__setattr__(self, name, value.copy())
        orientation = np.asarray(
            self.load_cell_orientation_site_to_engine, dtype=float
        )
        if orientation.shape != (3, 3) or not np.all(np.isfinite(orientation)):
            raise ValueError("load-cell orientation must be a finite 3x3 matrix")
        object.__setattr__(
            self, "load_cell_orientation_site_to_engine", orientation.copy()
        )
        axis = np.asarray(self.load_cell_axis_engine, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("load-cell axis must be a finite vector")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 0.0:
            raise ValueError("load-cell axis must be nonzero")
        object.__setattr__(self, "load_cell_axis_engine", axis / axis_norm)
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(self.reported_yaw_torque_n_m):
            raise ValueError("reported yaw torque must be finite")
        expected = torque_nm_to_dyne_cm(self.reported_yaw_torque_n_m)
        if not np.isclose(
            expected, self.reported_yaw_torque_dyne_cm, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError("dyne cm channel does not match the SI torque")

    def to_dict(self) -> Mapping[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return result


def paper_torque_from_support_reaction(
    support_force_engine_n: np.ndarray,
    support_moment_engine_n_m: np.ndarray,
    root_fluid_moment_engine_n_m: Optional[np.ndarray] = None,
    *,
    meter_mode: str = "equality-reaction",
    timestamp_s: float = 0.0,
    equality_only_yaw_reaction_engine_n_m: float = np.nan,
    yaw_balance_residual_n_m: float = np.nan,
    actuator_torque_n_m: Optional[np.ndarray] = None,
) -> PaperTetherTorqueSample:
    """Apply the declared engine-to-paper sign convention without fitting.

    Attempted fly moment is ``-support_on_fly``. Engine +z is a left turn and
    paper-positive is a right turn, so the paper channel equals support +z.
    """

    support_force = np.asarray(support_force_engine_n, dtype=float)
    support_moment = np.asarray(support_moment_engine_n_m, dtype=float)
    fluid = (
        np.zeros(3, dtype=float)
        if root_fluid_moment_engine_n_m is None
        else np.asarray(root_fluid_moment_engine_n_m, dtype=float)
    )
    fly_generated = -support_moment
    support_yaw = float(np.dot(support_moment, _PAPER_YAW_AXIS_ENGINE))
    reported = support_yaw
    return PaperTetherTorqueSample(
        support_force_engine_n=support_force,
        support_moment_engine_n_m=support_moment,
        fly_generated_moment_engine_n_m=fly_generated,
        root_fluid_moment_engine_n_m=fluid,
        reported_yaw_torque_n_m=reported,
        reported_yaw_torque_dyne_cm=torque_nm_to_dyne_cm(reported),
        meter_mode=meter_mode,
        timestamp_s=timestamp_s,
        support_on_fly_yaw_reaction_engine_n_m=support_yaw,
        attempted_fly_yaw_moment_from_support_engine_n_m=-support_yaw,
        equality_only_yaw_reaction_engine_n_m=(
            equality_only_yaw_reaction_engine_n_m
        ),
        yaw_balance_residual_n_m=yaw_balance_residual_n_m,
        actuator_torque_n_m=(
            _nan6()
            if actuator_torque_n_m is None
            else np.asarray(actuator_torque_n_m, dtype=float)
        ),
    )


class _PaperMeterCommon(FlyBodyPhysicsAdapter):
    """Shared published-clock, transform, and calibration behavior."""

    meter_mode = ""

    def _finish_initialization(self, published_config: FlyBodyWorkerConfig) -> None:
        import mujoco as mj

        self._physics_sample_timestep_s = float(published_config.timestep_s)
        self._solver_substeps = PAPER_TETHER_SOLVER_SUBSTEPS
        self.bundle["paper_physics_sample_timestep_s"] = self._physics_sample_timestep_s
        simulation = self.bundle["simulation"]
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        self._default_initial_state = self._state()
        self._tether_reference_state = self._default_initial_state.copy()
        self._last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._last_wrench = AerodynamicWrench(
            force_body_n=np.zeros(3, dtype=float),
            torque_body_n_m=np.zeros(3, dtype=float),
            left_force_body_n=np.zeros(3, dtype=float),
            right_force_body_n=np.zeros(3, dtype=float),
            mechanical_power_w=0.0,
        )
        self._sign_calibration: Optional[Mapping[str, Any]] = None
        self._last_internal_torque_samples: Tuple[PaperTetherTorqueSample, ...] = ()

    def ground_contact_count(self) -> int:
        return 0

    def body_state(self) -> RigidBodyState:
        return self._state()

    def head_transform_body(self) -> np.ndarray:
        data = self.bundle["simulation"].mj_data
        root_id = int(self.bundle["root_body_id"])
        head_id = int(self.bundle["head_body_id"])
        root_rotation = np.asarray(data.xmat[root_id], dtype=float).reshape(3, 3)
        head_rotation = np.asarray(data.xmat[head_id], dtype=float).reshape(3, 3)
        root_position = np.asarray(data.xpos[root_id], dtype=float)
        head_position = np.asarray(data.xpos[head_id], dtype=float)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = root_rotation.T @ head_rotation
        transform[:3, 3] = root_rotation.T @ (head_position - root_position) / 1000.0
        return transform

    def body_constraint_diagnostics(self) -> Mapping[str, float]:
        current = self._state()
        reference = self._tether_reference_state
        position_drift = float(
            np.linalg.norm(current.position_world_m - reference.position_world_m)
        )
        dot = abs(
            float(
                np.dot(
                    current.quaternion_body_to_world,
                    reference.quaternion_body_to_world,
                )
            )
        )
        rotation_drift = 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))
        return {
            "body_translation_drift_m": position_drift,
            "body_rotation_drift_rad": rotation_drift,
            "body_yaw_rate_rad_s": float(current.angular_velocity_body_rad_s[2]),
        }

    def solver_diagnostics(self) -> Mapping[str, Any]:
        """Expose constraints, warnings, and actuator headroom explicitly."""

        import mujoco as mj

        simulation = self.bundle["simulation"]
        model = simulation.mj_model
        data = simulation.mj_data
        nefc = int(data.nefc)
        counts: Dict[str, int] = {}
        for value in np.asarray(data.efc_type[:nefc], dtype=int):
            key = str(int(value))
            counts[key] = counts.get(key, 0) + 1
        joint_limit_rows = counts.get(
            str(int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)), 0
        )
        warning_counts = {
            str(index): int(data.warning[index].number)
            for index in range(len(data.warning))
            if int(data.warning[index].number) != 0
        }
        actuator = np.asarray(self.last_actuator_torque_n_m, dtype=float)
        limit = float(self.bundle["config"].max_abs_wing_torque_n_m)
        return {
            "constraint_row_count": nefc,
            "constraint_type_counts": counts,
            "joint_limit_constraint_row_count": joint_limit_rows,
            "numerical_warning_counts": warning_counts,
            "maximum_abs_actuator_torque_n_m": float(
                np.max(np.abs(actuator), initial=0.0)
            ),
            "actuator_limit_n_m": limit,
            "actuator_saturation_fraction": float(
                np.max(np.abs(actuator), initial=0.0) / limit
            ),
            "actuator_clipping_observed": False,
        }

    def step(
        self,
        wings: WingKinematics,
        force_body_n: np.ndarray,
        torque_body_n_m: np.ndarray,
        dt_s: float,
    ) -> RigidBodyState:
        if not np.isclose(
            dt_s, self._physics_sample_timestep_s, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError("paper meter step must equal its published timestep")
        internal_samples = []
        state = self._state()
        internal_dt_s = float(self.bundle["config"].timestep_s)
        for _ in range(self._solver_substeps):
            state = super().step(
                wings, force_body_n, torque_body_n_m, internal_dt_s
            )
            internal_samples.append(self.tether_torque_sample())
        self._last_internal_torque_samples = tuple(internal_samples)
        return state

    def consume_last_internal_torque_samples(
        self,
    ) -> Tuple[PaperTetherTorqueSample, ...]:
        result = self._last_internal_torque_samples
        self._last_internal_torque_samples = ()
        return result

    def _apply_world_wrench(
        self, force_engine_n: Sequence[float], moment_engine_n_m: Sequence[float]
    ) -> None:
        """Apply a world-frame root wrench through MuJoCo's Jacobian mapping."""

        import mujoco as mj

        simulation = self.bundle["simulation"]
        data = simulation.mj_data
        data.qfrc_applied[:] = 0.0
        force_internal = np.asarray(force_engine_n, dtype=float) / _FLYGYM_FORCE_TO_NEWTON
        moment_internal = (
            np.asarray(moment_engine_n_m, dtype=float)
            / _FLYGYM_TORQUE_TO_NEWTON_METRE
        )
        mj.mj_applyFT(
            simulation.mj_model,
            data,
            force_internal,
            moment_internal,
            data.xpos[int(self.bundle["root_body_id"])],
            int(self.bundle["root_body_id"]),
            data.qfrc_applied,
        )

    def _clear_calibration_wrench(self) -> None:
        simulation = self.bundle["simulation"]
        simulation.mj_data.qfrc_applied[:] = 0.0
        simulation.mj_data.xfrc_applied[:] = 0.0

    def calibrate_paper_yaw_sign(
        self, applied_right_turn_magnitude_n_m: float = 1.0e-7
    ) -> Mapping[str, Any]:
        """Inject both yaw signs and verify the paper convention and units."""

        import mujoco as mj

        magnitude = float(applied_right_turn_magnitude_n_m)
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("calibration magnitude must be positive")
        self.reset(self._default_initial_state.copy())
        simulation = self.bundle["simulation"]
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        baseline = self.tether_torque_sample().reported_yaw_torque_n_m
        measured: Dict[str, float] = {}
        for label, engine_sign in (("right_turn", -1.0), ("left_turn", 1.0)):
            self._set_calibration_wrench(
                np.zeros(3, dtype=float),
                np.asarray((0.0, 0.0, engine_sign * magnitude), dtype=float),
            )
            mj.mj_forward(simulation.mj_model, simulation.mj_data)
            measured[label] = (
                self.tether_torque_sample().reported_yaw_torque_n_m - baseline
            )
        expected = {"right_turn": magnitude, "left_turn": -magnitude}
        errors = {key: abs(measured[key] - expected[key]) for key in expected}
        relative_tolerance = 5.0e-4 if self.meter_mode == "fixed-load-cell" else 1.0e-3
        tolerance = max(relative_tolerance * magnitude, 1.0e-12)
        passed = all(value <= tolerance for value in errors.values())
        receipt = {
            "schema_version": PAPER_TETHER_SCHEMA_VERSION,
            "meter_mode": self.meter_mode,
            "baseline_paper_yaw_torque_n_m": baseline,
            "applied_engine_yaw_moment_n_m": {
                "right_turn": -magnitude,
                "left_turn": magnitude,
            },
            "expected_paper_yaw_torque_n_m": expected,
            "measured_paper_yaw_torque_n_m": measured,
            "measured_paper_yaw_torque_dyne_cm": {
                key: torque_nm_to_dyne_cm(value) for key, value in measured.items()
            },
            "absolute_error_n_m": errors,
            "tolerance_n_m": tolerance,
            "passed": passed,
        }
        self._clear_calibration_wrench()
        self.reset(self._default_initial_state.copy())
        self._sign_calibration = receipt
        if not passed:
            raise FlyBodyDependencyError("paper yaw sign calibration failed")
        return dict(receipt)


class PaperTetheredFlyBodyAdapter(_PaperMeterCommon):
    """Equality-reaction validator using only tether weld rows in ``J^T f``."""

    backend_name = "flybody_paper_equality_validator"
    body_state_reference = "weld-constrained FlyBody root/thorax free joint"
    meter_mode = "equality-reaction"

    def __init__(
        self,
        torque_mapper: Optional[WingTorqueMapper] = None,
        config: Optional[FlyBodyWorkerConfig] = None,
    ) -> None:
        self.torque_mapper = torque_mapper or WingAxisTorqueMap()
        published_config = config or FlyBodyWorkerConfig()
        internal_config = replace(
            published_config,
            timestep_s=published_config.timestep_s / PAPER_TETHER_SOLVER_SUBSTEPS,
        )
        self.bundle = build_flybody_simulation(
            internal_config, paper_tether=True
        )
        if self.bundle.get("paper_tether_mode") != "equality-reaction":
            raise FlyBodyDependencyError("worker did not construct equality validator")
        self._finish_initialization(published_config)

    def _root_wrench_from_generalized(self, generalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recover the root-origin world wrench without assuming a yaw DoF."""

        import mujoco as mj

        simulation = self.bundle["simulation"]
        model = simulation.mj_model
        data = simulation.mj_data
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mj.mj_jacBody(model, data, jacp, jacr, int(self.bundle["root_body_id"]))
        root = int(self.bundle["root_dof_adr"])
        spatial_jacobian = np.vstack((jacp[:, root : root + 6], jacr[:, root : root + 6]))
        root_generalized = np.asarray(generalized, dtype=float)[root : root + 6]
        try:
            wrench_internal = np.linalg.solve(spatial_jacobian.T, root_generalized)
        except np.linalg.LinAlgError as error:
            raise FlyBodyDependencyError("root wrench calibration matrix is singular") from error
        return (
            wrench_internal[:3] * _FLYGYM_FORCE_TO_NEWTON,
            wrench_internal[3:] * _FLYGYM_TORQUE_TO_NEWTON_METRE,
        )

    def _equality_only_generalized_reaction(self) -> np.ndarray:
        import mujoco as mj

        simulation = self.bundle["simulation"]
        model = simulation.mj_model
        data = simulation.mj_data
        nefc = int(data.nefc)
        isolated_force = np.zeros(nefc, dtype=float)
        constraint_type = np.asarray(data.efc_type[:nefc], dtype=int)
        constraint_id = np.asarray(data.efc_id[:nefc], dtype=int)
        rows = np.flatnonzero(
            (constraint_type == int(mj.mjtConstraint.mjCNSTR_EQUALITY))
            & (constraint_id == int(self.bundle["tether_equality_id"]))
        )
        if rows.size != 6:
            raise FlyBodyDependencyError(
                "tether weld must contribute exactly six equality rows"
            )
        isolated_force[rows] = np.asarray(data.efc_force[:nefc], dtype=float)[rows]
        generalized = np.zeros(model.nv, dtype=float)
        mj.mj_mulJacTVec(model, data, generalized, isolated_force)
        return generalized

    def yaw_equation_residual_n_m(self) -> float:
        import mujoco as mj

        simulation = self.bundle["simulation"]
        model = simulation.mj_model
        data = simulation.mj_data
        mass_acceleration = np.empty(model.nv, dtype=float)
        mj.mj_mulM(model, data, mass_acceleration, data.qacc)
        residual = (
            mass_acceleration
            + data.qfrc_bias
            - data.qfrc_passive
            - data.qfrc_actuator
            - data.qfrc_applied
            - data.qfrc_constraint
        )
        _force, moment = self._root_wrench_from_generalized(residual)
        return float(np.dot(moment, _PAPER_YAW_AXIS_ENGINE))

    def tether_torque_sample(self) -> PaperTetherTorqueSample:
        data = self.bundle["simulation"].mj_data
        equality_generalized = self._equality_only_generalized_reaction()
        support_force, support_moment = self._root_wrench_from_generalized(
            equality_generalized
        )
        fluid_force, fluid_moment = self._root_wrench_from_generalized(
            np.asarray(data.qfrc_fluid, dtype=float)
        )
        del fluid_force
        residual = self.yaw_equation_residual_n_m()
        equality_yaw = float(np.dot(support_moment, _PAPER_YAW_AXIS_ENGINE))
        return paper_torque_from_support_reaction(
            support_force,
            support_moment,
            fluid_moment,
            meter_mode=self.meter_mode,
            timestamp_s=float(data.time),
            equality_only_yaw_reaction_engine_n_m=equality_yaw,
            yaw_balance_residual_n_m=residual,
            actuator_torque_n_m=self.last_actuator_torque_n_m,
        )

    def _set_calibration_wrench(
        self, force_engine_n: np.ndarray, moment_engine_n_m: np.ndarray
    ) -> None:
        self._apply_world_wrench(force_engine_n, moment_engine_n_m)

    def provenance_metadata(self) -> Mapping[str, Any]:
        base = dict(super().provenance_metadata())
        base.update(
            {
                "schema_version": PAPER_TETHER_SCHEMA_VERSION,
                "torque_meter_mode": self.meter_mode,
                "topology": "free_root_six_dof_weld_validator",
                "authoritative": False,
                "paper_coordinate_system": dict(PAPER_ENGINE_AXES),
                "ground_contact_topology": "absent",
                "tether_equality_id": self.bundle["tether_equality_id"],
                "constraint_reaction_source": (
                    "six matching equality rows reconstructed with MuJoCo mj_mulJacTVec"
                ),
                "yaw_projection": "root Jacobian basis projected on calibrated world +z",
                "physics_sample_timestep_s": self._physics_sample_timestep_s,
                "internal_solver_timestep_s": self.bundle["config"].timestep_s,
                "solver_substeps_per_physics_sample": self._solver_substeps,
                "sign_calibration": self._sign_calibration,
            }
        )
        return base


class _PerWingAerodynamicProbe:
    """Non-integrating full/left/right fluid probes at fixed-model kinematics."""

    _QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}
    _QVEL_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}

    def __init__(self, published_config: FlyBodyWorkerConfig) -> None:
        import mujoco as mj

        internal_config = replace(
            published_config,
            timestep_s=published_config.timestep_s / PAPER_TETHER_SOLVER_SUBSTEPS,
        )
        self._bundles = {
            name: build_flybody_simulation(
                internal_config,
                paper_tether_mode="equality-reaction",
            )
            for name in ("full", "left", "right")
        }
        for probe_name, active_side in (("left", "left"), ("right", "right")):
            model = self._bundles[probe_name]["simulation"].mj_model
            for geom_name in self._bundles[probe_name]["fluid_geom_names"]:
                keep = (
                    (active_side == "left" and "/l_wing_fluid" in geom_name)
                    or (active_side == "right" and "/r_wing_fluid" in geom_name)
                )
                if not keep:
                    geom_id = int(
                        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
                    )
                    model.geom_fluid[geom_id, :] = 0.0
        self._joint_maps: Dict[str, Tuple[Tuple[int, int, int, int, int, int], ...]] = {}
        self._source_model_identity: Optional[int] = None

    def _build_joint_map(
        self, source_model: Any, destination_model: Any
    ) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
        import mujoco as mj

        result = []
        for source_joint in range(int(source_model.njnt)):
            source_type = int(source_model.jnt_type[source_joint])
            if source_type == int(mj.mjtJoint.mjJNT_FREE):
                continue
            name = mj.mj_id2name(
                source_model, mj.mjtObj.mjOBJ_JOINT, source_joint
            )
            destination_joint = int(
                mj.mj_name2id(
                    destination_model, mj.mjtObj.mjOBJ_JOINT, name
                )
            )
            if destination_joint < 0:
                raise FlyBodyDependencyError(
                    "aerodynamic probe is missing joint {!r}".format(name)
                )
            destination_type = int(destination_model.jnt_type[destination_joint])
            if destination_type != source_type:
                raise FlyBodyDependencyError(
                    "aerodynamic probe joint type changed for {!r}".format(name)
                )
            result.append(
                (
                    int(source_model.jnt_qposadr[source_joint]),
                    int(source_model.jnt_dofadr[source_joint]),
                    int(destination_model.jnt_qposadr[destination_joint]),
                    int(destination_model.jnt_dofadr[destination_joint]),
                    int(self._QPOS_WIDTH[source_type]),
                    int(self._QVEL_WIDTH[source_type]),
                )
            )
        return tuple(result)

    def _copy_fixed_kinematics(self, fixed_bundle: Mapping[str, Any], probe: str) -> None:
        import mujoco as mj

        source_simulation = fixed_bundle["simulation"]
        destination_bundle = self._bundles[probe]
        destination_simulation = destination_bundle["simulation"]
        source_model = source_simulation.mj_model
        source_data = source_simulation.mj_data
        destination_model = destination_simulation.mj_model
        destination_data = destination_simulation.mj_data
        cache_key = "{}:{}".format(id(source_model), probe)
        mapping = self._joint_maps.get(cache_key)
        if mapping is None:
            mapping = self._build_joint_map(source_model, destination_model)
            self._joint_maps[cache_key] = mapping
        destination_data.qpos[:] = 0.0
        destination_data.qvel[:] = 0.0
        free_joint_ids = np.flatnonzero(
            destination_model.jnt_type == mj.mjtJoint.mjJNT_FREE
        )
        if free_joint_ids.size != 1:
            raise FlyBodyDependencyError(
                "aerodynamic probe requires exactly one free root"
            )
        free_joint = int(free_joint_ids[0])
        root_qpos = int(destination_model.jnt_qposadr[free_joint])
        source_root = int(fixed_bundle["root_body_id"])
        destination_data.qpos[root_qpos : root_qpos + 3] = np.asarray(
            source_data.xpos[source_root], dtype=float
        )
        destination_data.qpos[root_qpos + 3 : root_qpos + 7] = np.asarray(
            source_data.xquat[source_root], dtype=float
        )
        for source_qpos, source_dof, target_qpos, target_dof, nq, nv in mapping:
            destination_data.qpos[target_qpos : target_qpos + nq] = (
                source_data.qpos[source_qpos : source_qpos + nq]
            )
            destination_data.qvel[target_dof : target_dof + nv] = (
                source_data.qvel[source_dof : source_dof + nv]
            )
        if destination_model.na:
            destination_data.act[:] = 0.0
        destination_data.ctrl[:] = 0.0
        destination_data.time = source_data.time
        mj.mj_forward(destination_model, destination_data)

    @staticmethod
    def _root_wrench_from_generalized(
        bundle: Mapping[str, Any], generalized: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        import mujoco as mj

        simulation = bundle["simulation"]
        model = simulation.mj_model
        data = simulation.mj_data
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mj.mj_jacBody(model, data, jacp, jacr, int(bundle["root_body_id"]))
        root = int(bundle["root_dof_adr"])
        spatial_jacobian = np.vstack(
            (jacp[:, root : root + 6], jacr[:, root : root + 6])
        )
        root_generalized = np.asarray(generalized, dtype=float)[root : root + 6]
        try:
            wrench_internal = np.linalg.solve(
                spatial_jacobian.T, root_generalized
            )
        except np.linalg.LinAlgError as error:
            raise FlyBodyDependencyError(
                "aerodynamic probe root basis is singular"
            ) from error
        return (
            wrench_internal[:3] * _FLYGYM_FORCE_TO_NEWTON,
            wrench_internal[3:] * _FLYGYM_TORQUE_TO_NEWTON_METRE,
        )

    def measure(self, fixed_bundle: Mapping[str, Any]) -> Mapping[str, Any]:
        moments = {}
        reported = {}
        for probe in ("full", "left", "right"):
            self._copy_fixed_kinematics(fixed_bundle, probe)
            bundle = self._bundles[probe]
            data = bundle["simulation"].mj_data
            _force, moment = self._root_wrench_from_generalized(
                bundle, np.asarray(data.qfrc_fluid, dtype=float)
            )
            moments[probe] = moment
            reported[probe] = -float(
                np.dot(moment, _PAPER_YAW_AXIS_ENGINE)
            )
        # Global density/viscosity also produces a non-wing body-fluid
        # background.  Each one-wing probe contains that same background, so
        # subtract it algebraically rather than counting it twice.
        background_moment = moments["left"] + moments["right"] - moments["full"]
        left_moment = moments["full"] - moments["right"]
        right_moment = moments["full"] - moments["left"]
        wing_moments = np.stack((left_moment, right_moment), axis=0)
        wing_reported = -wing_moments.dot(_PAPER_YAW_AXIS_ENGINE)
        full_wing_reported = float(np.sum(wing_reported))
        background_reported = -float(
            np.dot(background_moment, _PAPER_YAW_AXIS_ENGINE)
        )
        closure = float(
            full_wing_reported
            - (reported["full"] - background_reported)
        )
        return {
            "wing_moment_at_tether_engine_n_m": wing_moments,
            "wing_reported_yaw_torque_n_m": wing_reported,
            "full_reported_yaw_torque_n_m": full_wing_reported,
            "background_reported_yaw_torque_n_m": background_reported,
            "closure_residual_n_m": closure,
        }

    def provenance_metadata(self) -> Mapping[str, Any]:
        return {
            "method": (
                "three non-integrating free-root probes at copied fixed-model "
                "named-joint qpos/qvel: full, left-fluid-only, right-fluid-only"
            ),
            "side_order": list(PAPER_WING_SIDE_ORDER),
            "state_feedback_to_authoritative_model": False,
            "compiled_model_fingerprints": {
                key: value["compiled_model_fingerprint"]
                for key, value in self._bundles.items()
            },
        }


class PaperFixedLoadCellFlyBodyAdapter(_PaperMeterCommon):
    """Authoritative structurally fixed FlyBody root with a six-axis load cell."""

    backend_name = "flybody_paper_fixed_load_cell"
    body_state_reference = "structurally fixed FlyBody root/thorax"
    meter_mode = "fixed-load-cell"

    def __init__(
        self,
        torque_mapper: Optional[WingTorqueMapper] = None,
        config: Optional[FlyBodyWorkerConfig] = None,
        *,
        enable_aerodynamic_probe: bool = False,
    ) -> None:
        self.torque_mapper = torque_mapper or WingAxisTorqueMap()
        published_config = config or FlyBodyWorkerConfig()
        internal_config = replace(
            published_config,
            timestep_s=published_config.timestep_s / PAPER_TETHER_SOLVER_SUBSTEPS,
        )
        self.bundle = build_flybody_simulation(
            internal_config, paper_tether_mode="fixed-load-cell"
        )
        if self.bundle.get("paper_tether_mode") != "fixed-load-cell":
            raise FlyBodyDependencyError("worker did not construct fixed load cell")
        self._aerodynamic_probe = (
            _PerWingAerodynamicProbe(published_config)
            if enable_aerodynamic_probe
            else None
        )
        self._finish_initialization(published_config)

    def _state(self) -> RigidBodyState:
        data = self.bundle["simulation"].mj_data
        root = int(self.bundle["root_body_id"])
        return RigidBodyState(
            position_world_m=np.asarray(data.xpos[root], dtype=float).copy() / 1000.0,
            velocity_world_m_s=np.zeros(3, dtype=float),
            quaternion_body_to_world=np.asarray(data.xquat[root], dtype=float).copy(),
            angular_velocity_body_rad_s=np.zeros(3, dtype=float),
        )

    def reset(self, initial_state: RigidBodyState) -> None:
        import mujoco as mj

        initial_state.validate()
        if hasattr(self, "_tether_reference_state"):
            reference = self._tether_reference_state
            if not np.allclose(
                initial_state.position_world_m,
                reference.position_world_m,
                rtol=0.0,
                atol=1.0e-15,
            ) or not np.allclose(
                np.abs(
                    np.dot(
                        initial_state.quaternion_body_to_world,
                        reference.quaternion_body_to_world,
                    )
                ),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("fixed load-cell root pose cannot be reset elsewhere")
        simulation = self.bundle["simulation"]
        simulation.reset()
        self._clear_calibration_wrench()
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        self._last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._last_wrench = AerodynamicWrench(
            force_body_n=np.zeros(3, dtype=float),
            torque_body_n_m=np.zeros(3, dtype=float),
            left_force_body_n=np.zeros(3, dtype=float),
            right_force_body_n=np.zeros(3, dtype=float),
            mechanical_power_w=0.0,
        )
        self._last_internal_torque_samples = ()

    def _update_telemetry(
        self,
        rotation_body_to_world_before_step: np.ndarray,
        wing_velocity_before_step_rad_s: np.ndarray,
    ) -> None:
        del rotation_body_to_world_before_step
        simulation = self.bundle["simulation"]
        self._last_actuator_torque_n_m = (
            simulation.get_actuator_forces(
                self.bundle["fly"].name, self.bundle["actuator_type"]
            )
            * _FLYGYM_TORQUE_TO_NEWTON_METRE
        )
        all_velocities = simulation.get_joint_velocities(self.bundle["fly"].name)
        current = all_velocities[self.bundle["wing_joint_indices"]]
        mean_velocity = 0.5 * (
            np.asarray(wing_velocity_before_step_rad_s, dtype=float) + current
        )
        self._last_wrench = AerodynamicWrench(
            force_body_n=np.zeros(3, dtype=float),
            torque_body_n_m=np.zeros(3, dtype=float),
            left_force_body_n=np.zeros(3, dtype=float),
            right_force_body_n=np.zeros(3, dtype=float),
            mechanical_power_w=float(
                np.sum(np.abs(self._last_actuator_torque_n_m * mean_velocity))
            ),
        )

    def _sensor_vector(self, sensor_id: int) -> np.ndarray:
        model = self.bundle["simulation"].mj_model
        data = self.bundle["simulation"].mj_data
        address = int(model.sensor_adr[int(sensor_id)])
        dimension = int(model.sensor_dim[int(sensor_id)])
        return np.asarray(data.sensordata[address : address + dimension], dtype=float)

    def _wing_load_cell_measurement(self) -> Mapping[str, np.ndarray]:
        """Return bilateral hinge reactions in the registered engine frame.

        MuJoCo's sensor is the parent-on-child interaction.  The physical
        contribution applied by the wing to the thorax is its negative.  Each
        moment is translated from the hinge site to the canonical tether
        origin before projecting onto paper-positive yaw.
        """

        data = self.bundle["simulation"].mj_data
        tether_site = int(self.bundle["load_cell_site_id"])
        tether_position_m = (
            np.asarray(data.site_xpos[tether_site], dtype=float) / 1000.0
        )
        positions = []
        orientations = []
        parent_forces = []
        parent_moments = []
        wing_forces = []
        wing_hinge_moments = []
        wing_tether_moments = []
        reported = []
        for physical_side in PAPER_WING_SIDE_ORDER:
            site = int(self.bundle["wing_load_cell_site_ids"][physical_side])
            orientation = np.asarray(
                data.site_xmat[site], dtype=float
            ).reshape(3, 3)
            position_m = np.asarray(data.site_xpos[site], dtype=float) / 1000.0
            parent_force = orientation.dot(
                self._sensor_vector(
                    int(
                        self.bundle["wing_load_cell_force_sensor_ids"][
                            physical_side
                        ]
                    )
                )
            ) * _FLYGYM_FORCE_TO_NEWTON
            parent_moment = orientation.dot(
                self._sensor_vector(
                    int(
                        self.bundle["wing_load_cell_torque_sensor_ids"][
                            physical_side
                        ]
                    )
                )
            ) * _FLYGYM_TORQUE_TO_NEWTON_METRE
            wing_force = -parent_force
            wing_hinge_moment = -parent_moment
            wing_tether_moment = wing_hinge_moment + np.cross(
                position_m - tether_position_m, wing_force
            )
            positions.append(position_m)
            orientations.append(orientation)
            parent_forces.append(parent_force)
            parent_moments.append(parent_moment)
            wing_forces.append(wing_force)
            wing_hinge_moments.append(wing_hinge_moment)
            wing_tether_moments.append(wing_tether_moment)
            reported.append(
                -float(np.dot(wing_tether_moment, _PAPER_YAW_AXIS_ENGINE))
            )
        return {
            "position_engine_m": np.asarray(positions, dtype=float),
            "orientation_site_to_engine": np.asarray(orientations, dtype=float),
            "parent_on_child_force_engine_n": np.asarray(
                parent_forces, dtype=float
            ),
            "parent_on_child_moment_at_hinge_engine_n_m": np.asarray(
                parent_moments, dtype=float
            ),
            "wing_on_thorax_force_engine_n": np.asarray(wing_forces, dtype=float),
            "wing_on_thorax_moment_at_hinge_engine_n_m": np.asarray(
                wing_hinge_moments, dtype=float
            ),
            "wing_on_thorax_moment_at_tether_engine_n_m": np.asarray(
                wing_tether_moments, dtype=float
            ),
            "reported_yaw_torque_n_m": np.asarray(reported, dtype=float),
        }

    def tether_torque_sample(self) -> PaperTetherTorqueSample:
        simulation = self.bundle["simulation"]
        data = simulation.mj_data
        site = int(self.bundle["load_cell_site_id"])
        site_to_engine = np.asarray(data.site_xmat[site], dtype=float).reshape(3, 3)
        # MuJoCo force/torque sensors report the parent-on-child interaction at
        # the site's body.  The injected-axis calibration below independently
        # gates this sign before any paper-positive result can be released.
        sensor_force_site = self._sensor_vector(
            int(self.bundle["load_cell_force_sensor_id"])
        )
        sensor_moment_site = self._sensor_vector(
            int(self.bundle["load_cell_torque_sensor_id"])
        )
        support_force = site_to_engine.dot(sensor_force_site)
        support_moment = site_to_engine.dot(sensor_moment_site)
        support_force *= _FLYGYM_FORCE_TO_NEWTON
        support_moment *= _FLYGYM_TORQUE_TO_NEWTON_METRE
        base = paper_torque_from_support_reaction(
            support_force,
            support_moment,
            _nan3(),
            meter_mode=self.meter_mode,
            timestamp_s=float(data.time),
            actuator_torque_n_m=self.last_actuator_torque_n_m,
        )
        wing = self._wing_load_cell_measurement()
        wing_reported = wing["reported_yaw_torque_n_m"]
        wing_sum = float(np.sum(wing_reported))
        aerodynamic = (
            None
            if self._aerodynamic_probe is None
            else self._aerodynamic_probe.measure(self.bundle)
        )
        return replace(
            base,
            support_load_cell_force_engine_n=support_force,
            support_load_cell_moment_engine_n_m=support_moment,
            load_cell_position_engine_m=(
                np.asarray(data.site_xpos[site], dtype=float) / 1000.0
            ),
            load_cell_orientation_site_to_engine=site_to_engine,
            wing_load_cell_position_engine_m=wing["position_engine_m"],
            wing_load_cell_orientation_site_to_engine=wing[
                "orientation_site_to_engine"
            ],
            wing_parent_on_child_force_engine_n=wing[
                "parent_on_child_force_engine_n"
            ],
            wing_parent_on_child_moment_at_hinge_engine_n_m=wing[
                "parent_on_child_moment_at_hinge_engine_n_m"
            ],
            wing_on_thorax_force_engine_n=wing[
                "wing_on_thorax_force_engine_n"
            ],
            wing_on_thorax_moment_at_hinge_engine_n_m=wing[
                "wing_on_thorax_moment_at_hinge_engine_n_m"
            ],
            wing_on_thorax_moment_at_tether_engine_n_m=wing[
                "wing_on_thorax_moment_at_tether_engine_n_m"
            ],
            wing_reported_yaw_torque_n_m=wing_reported,
            wing_sum_reported_yaw_torque_n_m=wing_sum,
            nonwing_reported_yaw_torque_residual_n_m=(
                base.reported_yaw_torque_n_m - wing_sum
            ),
            wing_aerodynamic_moment_at_tether_engine_n_m=(
                _nan23()
                if aerodynamic is None
                else aerodynamic["wing_moment_at_tether_engine_n_m"]
            ),
            wing_aerodynamic_reported_yaw_torque_n_m=(
                _nan2()
                if aerodynamic is None
                else aerodynamic["wing_reported_yaw_torque_n_m"]
            ),
            full_aerodynamic_reported_yaw_torque_n_m=(
                np.nan
                if aerodynamic is None
                else aerodynamic["full_reported_yaw_torque_n_m"]
            ),
            background_aerodynamic_reported_yaw_torque_n_m=(
                np.nan
                if aerodynamic is None
                else aerodynamic["background_reported_yaw_torque_n_m"]
            ),
            aerodynamic_closure_residual_n_m=(
                np.nan
                if aerodynamic is None
                else aerodynamic["closure_residual_n_m"]
            ),
        )

    def yaw_equation_residual_n_m(self) -> float:
        # There is no root generalized coordinate in this topology.  Balance
        # is assessed by the matched equality meter and reported separately.
        return float("nan")

    def _set_calibration_wrench(
        self, force_engine_n: np.ndarray, moment_engine_n_m: np.ndarray
    ) -> None:
        data = self.bundle["simulation"].mj_data
        data.xfrc_applied[:] = 0.0
        root = int(self.bundle["root_body_id"])
        data.xfrc_applied[root, :3] = (
            np.asarray(force_engine_n, dtype=float) / _FLYGYM_FORCE_TO_NEWTON
        )
        data.xfrc_applied[root, 3:] = (
            np.asarray(moment_engine_n_m, dtype=float)
            / _FLYGYM_TORQUE_TO_NEWTON_METRE
        )

    def attachment_offset_sensitivity(
        self, offset_m: float = 0.0005
    ) -> Mapping[str, Mapping[str, float]]:
        """Analytical ``M + r x F`` yaw sensitivity at ±x/±y offsets."""

        sample = self.tether_torque_sample()
        result: Dict[str, Mapping[str, float]] = {}
        for axis, index in (("x", 0), ("y", 1)):
            for sign, label in ((-1.0, "minus"), (1.0, "plus")):
                r = np.zeros(3, dtype=float)
                r[index] = sign * float(offset_m)
                shifted = sample.support_moment_engine_n_m + np.cross(
                    r, sample.support_force_engine_n
                )
                result["{}_{}".format(axis, label)] = {
                    "offset_m": float(r[index]),
                    "reported_yaw_torque_n_m": float(shifted[2]),
                    "reported_yaw_torque_dyne_cm": torque_nm_to_dyne_cm(
                        float(shifted[2])
                    ),
                }
        return result

    def provenance_metadata(self) -> Mapping[str, Any]:
        base = dict(super().provenance_metadata())
        site = int(self.bundle["load_cell_site_id"])
        data = self.bundle["simulation"].mj_data
        base.update(
            {
                "schema_version": PAPER_TETHER_SCHEMA_VERSION,
                "torque_meter_mode": self.meter_mode,
                "topology": "structurally_fixed_root_with_site_force_torque_sensor",
                "authoritative_meter": True,
                "paper_coordinate_system": dict(PAPER_ENGINE_AXES),
                "ground_contact_topology": "absent",
                "head_constraint": "thorax-to-head DoFs omitted; rigid compiled transform",
                "load_cell": {
                    "site_id": site,
                    "position_engine_m": (
                        np.asarray(data.site_xpos[site], dtype=float) / 1000.0
                    ).tolist(),
                    "orientation_site_to_engine": np.asarray(
                        data.site_xmat[site], dtype=float
                    ).reshape(3, 3).tolist(),
                    "yaw_axis_engine": _PAPER_YAW_AXIS_ENGINE.tolist(),
                    "raw_sensor_semantics": "parent_on_child",
                    "reported_support_semantics": "parent_on_child",
                },
                "wing_load_cells": {
                    "side_order": list(PAPER_WING_SIDE_ORDER),
                    "physical_side_registration": {
                        "left": "flybody/l_wing; body +y",
                        "right": "flybody/r_wing; body -y",
                    },
                    "sensor_semantics": "parent_on_wing at physical hinge",
                    "reported_semantics": (
                        "negative wing-on-thorax engine-z moment transported "
                        "to the canonical tether origin"
                    ),
                    "paper_comparable": False,
                    "channels": {
                        physical_side: {
                            "site_id": int(
                                self.bundle["wing_load_cell_site_ids"][physical_side]
                            ),
                            "force_sensor_id": int(
                                self.bundle["wing_load_cell_force_sensor_ids"][
                                    physical_side
                                ]
                            ),
                            "torque_sensor_id": int(
                                self.bundle["wing_load_cell_torque_sensor_ids"][
                                    physical_side
                                ]
                            ),
                        }
                        for physical_side in PAPER_WING_SIDE_ORDER
                    },
                },
                "root_fluid_moment_availability": (
                    "unavailable without a root generalized coordinate; use comparison validator"
                ),
                "per_wing_aerodynamic_probe": (
                    None
                    if self._aerodynamic_probe is None
                    else self._aerodynamic_probe.provenance_metadata()
                ),
                "physics_sample_timestep_s": self._physics_sample_timestep_s,
                "internal_solver_timestep_s": self.bundle["config"].timestep_s,
                "solver_substeps_per_physics_sample": self._solver_substeps,
                "sign_calibration": self._sign_calibration,
            }
        )
        return base


class PaperComparisonFlyBodyAdapter:
    """Advance fixed and equality meters with identical commands and clocks."""

    meter_mode = "comparison"
    backend_name = "flybody_paper_dual_meter_comparison"

    def __init__(
        self,
        torque_mapper: Optional[WingTorqueMapper] = None,
        config: Optional[FlyBodyWorkerConfig] = None,
    ) -> None:
        mapper = torque_mapper or WingAxisTorqueMap()
        self.fixed = PaperFixedLoadCellFlyBodyAdapter(
            mapper, config, enable_aerodynamic_probe=True
        )
        self.validator = PaperTetheredFlyBodyAdapter(mapper, config)
        self.bundle = self.fixed.bundle
        self._last_internal_torque_samples: Tuple[PaperTetherTorqueSample, ...] = ()
        self._sign_calibration: Optional[Mapping[str, Any]] = None

    def default_initial_state(self) -> RigidBodyState:
        return self.fixed.default_initial_state()

    def reset(self, initial_state: RigidBodyState) -> None:
        self.fixed.reset(initial_state)
        self.validator.reset(self.validator.default_initial_state())
        self._last_internal_torque_samples = ()

    def step(
        self,
        wings: WingKinematics,
        force_body_n: np.ndarray,
        torque_body_n_m: np.ndarray,
        dt_s: float,
    ) -> RigidBodyState:
        state = self.fixed.step(wings, force_body_n, torque_body_n_m, dt_s)
        self.validator.step(wings, force_body_n, torque_body_n_m, dt_s)
        fixed_samples = self.fixed.consume_last_internal_torque_samples()
        validation_samples = self.validator.consume_last_internal_torque_samples()
        if len(fixed_samples) != len(validation_samples):
            raise FlyBodyDependencyError("dual-meter internal clocks diverged")
        self._last_internal_torque_samples = tuple(
            replace(
                fixed_sample,
                meter_mode="comparison",
                equality_only_yaw_reaction_engine_n_m=(
                    validation_sample.support_on_fly_yaw_reaction_engine_n_m
                ),
                root_fluid_moment_engine_n_m=(
                    validation_sample.root_fluid_moment_engine_n_m
                ),
                yaw_balance_residual_n_m=validation_sample.yaw_balance_residual_n_m,
            )
            for fixed_sample, validation_sample in zip(
                fixed_samples, validation_samples
            )
        )
        return state

    def consume_last_internal_torque_samples(
        self,
    ) -> Tuple[PaperTetherTorqueSample, ...]:
        result = self._last_internal_torque_samples
        self._last_internal_torque_samples = ()
        return result

    def tether_torque_sample(self) -> PaperTetherTorqueSample:
        fixed = self.fixed.tether_torque_sample()
        validator = self.validator.tether_torque_sample()
        return replace(
            fixed,
            meter_mode="comparison",
            equality_only_yaw_reaction_engine_n_m=(
                validator.support_on_fly_yaw_reaction_engine_n_m
            ),
            root_fluid_moment_engine_n_m=validator.root_fluid_moment_engine_n_m,
            yaw_balance_residual_n_m=validator.yaw_balance_residual_n_m,
        )

    def calibrate_paper_yaw_sign(
        self, applied_right_turn_magnitude_n_m: float = 1.0e-7
    ) -> Mapping[str, Any]:
        fixed = self.fixed.calibrate_paper_yaw_sign(
            applied_right_turn_magnitude_n_m
        )
        validator = self.validator.calibrate_paper_yaw_sign(
            applied_right_turn_magnitude_n_m
        )
        receipt = {
            "schema_version": PAPER_TETHER_SCHEMA_VERSION,
            "meter_mode": self.meter_mode,
            "fixed_load_cell": fixed,
            "equality_reaction": validator,
            "passed": fixed.get("passed") is True and validator.get("passed") is True,
        }
        self._sign_calibration = receipt
        return receipt

    def body_state(self) -> RigidBodyState:
        return self.fixed.body_state()

    def wing_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fixed.wing_joint_state()

    @property
    def last_actuator_torque_n_m(self) -> np.ndarray:
        return self.fixed.last_actuator_torque_n_m

    def head_transform_body(self) -> np.ndarray:
        return self.fixed.head_transform_body()

    def ground_contact_count(self) -> int:
        return max(self.fixed.ground_contact_count(), self.validator.ground_contact_count())

    def body_constraint_diagnostics(self) -> Mapping[str, float]:
        return self.fixed.body_constraint_diagnostics()

    def yaw_equation_residual_n_m(self) -> float:
        return self.validator.yaw_equation_residual_n_m()

    def solver_diagnostics(self) -> Mapping[str, Any]:
        return {
            "fixed_load_cell": self.fixed.solver_diagnostics(),
            "equality_reaction": self.validator.solver_diagnostics(),
        }

    def attachment_offset_sensitivity(
        self, offset_m: float = 0.0005
    ) -> Mapping[str, Mapping[str, float]]:
        return self.fixed.attachment_offset_sensitivity(offset_m)

    def provenance_metadata(self) -> Mapping[str, Any]:
        return {
            "schema_version": PAPER_TETHER_SCHEMA_VERSION,
            "torque_meter_mode": self.meter_mode,
            "authoritative_meter": self.fixed.provenance_metadata(),
            "independent_validator": self.validator.provenance_metadata(),
            "sign_calibration": self._sign_calibration,
        }


def make_paper_torque_meter(
    mode: str,
    *,
    torque_mapper: Optional[WingTorqueMapper] = None,
    config: Optional[FlyBodyWorkerConfig] = None,
) -> Any:
    """Construct a declared meter mode without changing legacy adapters."""

    if mode == "fixed-load-cell":
        return PaperFixedLoadCellFlyBodyAdapter(torque_mapper, config)
    if mode == "equality-reaction":
        return PaperTetheredFlyBodyAdapter(torque_mapper, config)
    if mode == "comparison":
        return PaperComparisonFlyBodyAdapter(torque_mapper, config)
    raise ValueError("unsupported torque meter mode")


__all__ = [
    "PAPER_ENGINE_AXES",
    "PAPER_TETHER_SCHEMA_VERSION",
    "PAPER_TETHER_SOLVER_SUBSTEPS",
    "PAPER_TORQUE_METER_MODES",
    "PAPER_WING_SIDE_ORDER",
    "PaperComparisonFlyBodyAdapter",
    "PaperFixedLoadCellFlyBodyAdapter",
    "PaperTetherTorqueSample",
    "PaperTetheredFlyBodyAdapter",
    "make_paper_torque_meter",
    "paper_torque_from_support_reaction",
    "torque_dyne_cm_to_nm",
    "torque_nm_to_dyne_cm",
]
