"""Pinned FlyGym/FlyBody worker adapter.

The scientific core deliberately does not import FlyGym: FlyGym 2.1.0 requires
Python 3.12 while the evidence and exploratory models remain usable on older
Python installations.  This module is imported only by the authoritative worker.

FlyGym 2.1.0's experimental FlyBody importer includes the articulated wings but
does not recreate the two ellipsoid aerodynamic proxy geoms from the original
FlyBody flight model.  ``build_flybody_simulation`` adds those geoms with the
published upstream dimensions and coefficients before compilation.  The adapter
then accepts generalized wing torques; it never interprets connectome synapse
counts as actuator commands.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sysconfig
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .flight.rigid_body import quaternion_to_matrix
from .flight.types import AerodynamicWrench, RigidBodyState, WingKinematics
from .schema import REVIEWED_FLYBODY_WING_AXIS_ORDER


PINNED_FLYGYM_VERSION = "2.1.0"
PINNED_MUJOCO_SERIES = "3.9"
WORKER_IMAGE_DIGEST_ENV = "FLY_S2B_WORKER_IMAGE_DIGEST"
WORKER_DEPENDENCY_LOCK_NAME = "requirements.lock"
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_FLYGYM_FORCE_TO_NEWTON = 1.0e-6  # gram * millimetre / second^2
_FLYGYM_TORQUE_TO_NEWTON_METRE = 1.0e-9  # gram * millimetre^2 / second^2

# FlyGym 2.1.0 copies these two values from FlyBody's centimetre-based XML
# without converting them when it attaches the model to a millimetre-based
# world.  MuJoCo's fluid model therefore sees air that is 1000 times too dense
# unless the worker corrects the parent world before compilation.
_AIR_DENSITY_G_MM3 = 1.28e-6
_AIR_VISCOSITY_G_MM_S = 1.85e-5
_RELEASED_FLIGHT_WING_STIFFNESS_INTERNAL = 1.0
_RELEASED_FLIGHT_WING_DAMPING_INTERNAL = 0.776923
_UPSTREAM_FLIGHT_CONTROL_TIMESTEP_S = 2.0e-4
_FLYBODY_CHECKPOINT_SCHEMA_VERSION = "1.0.0"
_FLYBODY_CHECKPOINT_STATE_SPEC = "mjSTATE_INTEGRATION"
_FLYBODY_CHECKPOINT_FLOAT_ENCODING = "float64_le_base64"

# Original FlyBody XML is expressed in centimetres.  FlyGym 2.x converts the
# articulated model to millimetres, hence the factor-of-ten coordinates below.
_WING_FLUID_GEOMS = (
    {
        "segment": "l_wing",
        "name": "l_wing_fluid",
        "size_mm": (0.005, 0.551, 1.14),
        "pos_mm": (0.263, -1.48, -0.289),
        "quat_wxyz": (-0.685, -0.634, 0.265, -0.243),
    },
    {
        "segment": "r_wing",
        "name": "r_wing_fluid",
        "size_mm": (0.005, 0.551, 1.14),
        "pos_mm": (-0.263, 1.48, 0.289),
        "quat_wxyz": (0.243, 0.265, 0.634, -0.685),
    },
)
_FLUID_COEFFICIENTS = (1.0, 0.5, 1.5, 1.7, 1.0)


class FlyBodyDependencyError(RuntimeError):
    """Raised when the authoritative worker dependencies are unavailable."""


def declared_worker_image_digest() -> Optional[str]:
    """Return a validated externally supplied OCI image identity, if present."""

    value = os.environ.get(WORKER_IMAGE_DIGEST_ENV)
    if value is None:
        return None
    if _IMAGE_DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "%s must be sha256:<64 lowercase hexadecimal characters>"
            % WORKER_IMAGE_DIGEST_ENV
        )
    return value


def worker_dependency_lock_path() -> Path:
    """Resolve the exact hash-locked worker dependency inventory.

    Source checkouts use the repository root. Installed wheels place the same
    bytes under the interpreter data prefix. No current-working-directory
    fallback is accepted, so an unrelated file cannot become a provenance
    receipt merely by changing where the command is invoked.
    """

    candidates = (
        Path(__file__).resolve().parents[2] / WORKER_DEPENDENCY_LOCK_NAME,
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "worker"
        / WORKER_DEPENDENCY_LOCK_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "the packaged worker dependency lock is unavailable: %s"
        % WORKER_DEPENDENCY_LOCK_NAME
    )


def worker_dependency_lock_sha256() -> str:
    """Return the dependency-lock digest in OCI-style ``sha256:`` form."""

    return "sha256:" + hashlib.sha256(worker_dependency_lock_path().read_bytes()).hexdigest()


class WingTorqueMapper(Protocol):
    """Explicit boundary from a two-wing hinge output to FlyBody's six axes."""

    def map_torque(self, wings: WingKinematics) -> np.ndarray:
        ...


@dataclass(frozen=True)
class WingAxisTorqueMap:
    """Exploratory distribution of each wing torque over yaw/roll/pitch.

    The three coefficients for each side are fit parameters, not anatomical
    facts. The default assigns the reduced hinge's stroke-axis torque to the
    FlyBody yaw (stroke) DoF and leaves the other axes unforced. A calibrated
    inverse-dynamics or moment-arm map can replace this object without changing
    neural, muscle, or MuJoCo code.
    """

    left_yaw_roll_pitch: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    right_yaw_roll_pitch: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    validation_status: str = "exploratory"
    provenance: str = "virtual stroke-axis assumption; not empirically calibrated"

    def __post_init__(self) -> None:
        for name, values in (
            ("left_yaw_roll_pitch", self.left_yaw_roll_pitch),
            ("right_yaw_roll_pitch", self.right_yaw_roll_pitch),
        ):
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError(name + " must contain three finite coefficients")

    def map_torque(self, wings: WingKinematics) -> np.ndarray:
        if wings.wing_axis_torque_n_m is not None:
            axis_torque = np.asarray(wings.wing_axis_torque_n_m, dtype=float)
            if axis_torque.shape != (6,) or not np.all(np.isfinite(axis_torque)):
                raise ValueError("WingKinematics wing-axis torque must have shape (6,)")
            return axis_torque.copy()
        torque = np.asarray(wings.generalized_torque_n_m, dtype=float)
        if torque.shape != (2,) or not np.all(np.isfinite(torque)):
            raise ValueError("WingKinematics generalized torque must have shape (2,)")
        return np.concatenate(
            (
                torque[0] * np.asarray(self.left_yaw_roll_pitch, dtype=float),
                torque[1] * np.asarray(self.right_yaw_roll_pitch, dtype=float),
            )
        )


@dataclass(frozen=True)
class FlyBodyWorkerConfig:
    """Configuration for a free-flight FlyBody rollout.

    FlyGym uses millimetres internally. Public results are converted to SI.
    """

    timestep_s: float = 1.0e-4
    spawn_height_m: float = 0.02
    max_abs_wing_torque_n_m: float = 3.0e-6
    add_aerodynamic_geoms: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestep_s) or self.timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if not np.isfinite(self.spawn_height_m) or self.spawn_height_m <= 0:
            raise ValueError("spawn_height_m must be positive")
        if (
            not np.isfinite(self.max_abs_wing_torque_n_m)
            or self.max_abs_wing_torque_n_m <= 0
        ):
            raise ValueError("max_abs_wing_torque_n_m must be positive")


@dataclass
class FlyBodyTrace:
    """SI-unit output from an authoritative FlyGym rollout."""

    time_s: np.ndarray
    body_position_m: np.ndarray
    body_quaternion_wxyz: np.ndarray
    body_linear_velocity_m_s: np.ndarray
    body_angular_velocity_rad_s: np.ndarray
    wing_angles_rad: np.ndarray
    wing_angular_velocity_rad_s: np.ndarray
    actuator_torque_n_m: np.ndarray
    root_fluid_force_n: np.ndarray
    root_fluid_torque_n_m: np.ndarray
    metadata: Dict[str, Any]

    def to_serializable(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return result


@dataclass(frozen=True)
class FlyBodyPhysicsCheckpoint:
    """Hash-protected MuJoCo integration state for fresh-adapter re-entry.

    The payload is intentionally opaque to the neural and muscle layers.  It
    binds MuJoCo's complete ``mjSTATE_INTEGRATION`` vector to the compiled
    FlyBody model fingerprint, reviewed wing-axis order, worker versions, and
    physics timestep.  Adapter telemetry is included because it is observable
    state at the process boundary even though MuJoCo does not need it to
    advance the next transition.

    Constructing this value verifies content integrity only.  Compatibility
    with a particular compiled model is checked by
    :meth:`FlyBodyPhysicsAdapter.restore_checkpoint` before ``mjData`` is
    mutated.
    """

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "state_spec",
            "float_encoding",
            "state_size",
            "simulation_state_b64",
            "compiled_model_sha256",
            "worker_versions",
            "physics_timestep_s",
            "wing_dof_order",
            "last_actuator_torque_n_m",
            "last_aerodynamic_wrench",
            "payload_sha256",
        }
        if not isinstance(self.payload, Mapping) or set(self.payload) != expected:
            raise ValueError("FlyBody checkpoint fields do not match schema")
        without_digest = dict(self.payload)
        supplied_digest = without_digest.pop("payload_sha256")
        if not isinstance(supplied_digest, str) or _IMAGE_DIGEST_PATTERN.fullmatch(
            supplied_digest
        ) is None:
            raise ValueError("FlyBody checkpoint payload_sha256 is invalid")
        observed_digest = "sha256:" + hashlib.sha256(
            json.dumps(
                without_digest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if supplied_digest != observed_digest:
            raise ValueError("FlyBody checkpoint payload SHA-256 mismatch")

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-compatible payload."""

        return json.loads(
            json.dumps(
                self.payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


def dependency_versions() -> Dict[str, Optional[str]]:
    """Return installed worker versions without importing their native modules."""

    versions: Dict[str, Optional[str]] = {}
    for package in ("flygym", "mujoco"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def dependency_license_metadata() -> Dict[str, str]:
    """Return declared package-license metadata without guessing asset terms."""

    licenses: Dict[str, str] = {}
    for package in ("flygym", "mujoco"):
        distribution = metadata.distribution(package)
        declared = (
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License")
            or "not declared in installed package metadata"
        )
        licenses[package] = str(declared).strip()
    licenses["flybody_model_assets"] = (
        "not independently declared by this adapter; verify FlyGym/FlyBody "
        "source-asset redistribution terms before repackaging"
    )
    return licenses


def dependency_record_fingerprints() -> Dict[str, str]:
    """Hash installed wheel RECORD manifests for the reviewed native stack.

    The hash-locked worker installation already verifies wheel payloads against
    ``requirements.lock``. Recording the installed distributions' RECORD bytes
    gives each immutable regression an exact, platform-specific dependency
    receipt without importing either native package.
    """

    fingerprints: Dict[str, str] = {}
    for package in ("flygym", "mujoco"):
        distribution = metadata.distribution(package)
        record = distribution.read_text("RECORD")
        if not record:
            raise FlyBodyDependencyError(
                "installed {} distribution has no RECORD manifest".format(package)
            )
        fingerprints[package] = "sha256:" + hashlib.sha256(
            record.encode("utf-8")
        ).hexdigest()
    return fingerprints


def _compiled_model_fingerprint(model: Any) -> Dict[str, Any]:
    """Return an exact structural fingerprint of the compiled MuJoCo model."""

    digest = hashlib.sha256()
    counts = {
        name: int(getattr(model, name))
        for name in (
            "nq",
            "nv",
            "nu",
            "nbody",
            "njnt",
            "ngeom",
            "ntendon",
            "nmesh",
        )
    }
    options = {
        "integrator": int(model.opt.integrator),
        "timestep_s": float(model.opt.timestep),
        "gravity_internal": np.asarray(model.opt.gravity, dtype=float).tolist(),
        "density_internal": float(model.opt.density),
        "viscosity_internal": float(model.opt.viscosity),
    }
    digest.update(
        json.dumps(
            {"counts": counts, "options": options},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for field in (
        "names",
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_bodyid",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "jnt_stiffness",
        "dof_damping",
        "geom_type",
        "geom_bodyid",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_fluid",
        "geom_contype",
        "geom_conaffinity",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "tendon_adr",
        "tendon_num",
        "tendon_range",
        "tendon_stiffness",
        "tendon_damping",
        "mesh_vert",
        "mesh_face",
    ):
        value = getattr(model, field, None)
        if value is None:
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(field.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "counts": counts,
        "options": options,
        "total_body_mass_internal": float(np.sum(model.body_mass)),
    }


def assert_worker_compatible() -> Dict[str, str]:
    """Fail closed if the worker is not running the reviewed upstream versions."""

    versions = dependency_versions()
    if versions["flygym"] != PINNED_FLYGYM_VERSION:
        raise FlyBodyDependencyError(
            "Authoritative FlyBody runs require flygym=={}; found {!r}.".format(
                PINNED_FLYGYM_VERSION, versions["flygym"]
            )
        )
    mujoco_version = versions["mujoco"]
    if mujoco_version is None or not mujoco_version.startswith(
        PINNED_MUJOCO_SERIES + "."
    ):
        raise FlyBodyDependencyError(
            "Authoritative FlyBody runs require MuJoCo {}.x; found {!r}.".format(
                PINNED_MUJOCO_SERIES, mujoco_version
            )
        )
    return {"flygym": versions["flygym"], "mujoco": mujoco_version}


def _add_wing_fluid_geoms(fly: Any, mj: Any) -> None:
    """Restore the original FlyBody wing-fluid ellipsoids to FlyGym's MjSpec."""

    bodies = {segment.name: body for segment, body in fly.bodyseg_to_mjcfbody.items()}
    for definition in _WING_FLUID_GEOMS:
        body = bodies[definition["segment"]]
        geom = body.add_geom(
            name=definition["name"],
            type=mj.mjtGeom.mjGEOM_ELLIPSOID,
            size=definition["size_mm"],
            pos=definition["pos_mm"],
            quat=definition["quat_wxyz"],
            mass=0,
            group=3,
            # These are aerodynamic proxy surfaces, not collision shapes.
            # Leaving MjSpec's default masks enabled creates an unintended
            # wing-wing contact pair that is absent from the flight model.
            contype=0,
            conaffinity=0,
        )
        # Native MjSpec names corresponding to MJCF
        # fluidshape="ellipsoid" and fluidcoef="...".
        geom.fluid_ellipsoid = 1.0
        geom.fluid_coefs = _FLUID_COEFFICIENTS


def build_flybody_simulation(
    config: Optional[FlyBodyWorkerConfig] = None,
    *,
    paper_tether: bool = False,
    paper_tether_mode: Optional[str] = None,
    paper_wing_load_cells: bool = True,
) -> Dict[str, Any]:
    """Build a FlyBody with six torque-controlled wing DoFs.

    Returns an internal bundle consumed by :func:`run_torque_trace`. Keeping this
    boundary private prevents neural code from depending on MuJoCo indices.
    ``paper_tether`` retains the released equality-tether contract.  The new
    ``paper_tether_mode="fixed-load-cell"`` topology removes the root free
    joint and installs MuJoCo force and torque sensors at the root origin and,
    by default, at both physical wing hinges.  ``paper_wing_load_cells=False``
    exists only for topology-invariance tests.  The default free-flight
    topology and its existing fingerprints are unchanged.
    """

    if paper_tether and paper_tether_mode is not None:
        raise ValueError("paper_tether and paper_tether_mode are mutually exclusive")
    resolved_tether_mode = (
        "equality-reaction" if paper_tether else paper_tether_mode
    )
    if resolved_tether_mode not in (None, "equality-reaction", "fixed-load-cell"):
        raise ValueError("unsupported paper tether mode")
    is_paper_tether = resolved_tether_mode is not None

    versions = assert_worker_compatible()
    config = config or FlyBodyWorkerConfig()

    import mujoco as mj
    from flygym import assets_dir
    from flygym.compose.fly import ActuatorType, FlyBody
    from flygym.compose.pose import KinematicPose
    from flygym.compose.world import FlatGroundWorld
    from flygym.compose.world.base_world import BaseWorld
    from flygym.flybody import (
        FlyBodyAxisOrder,
        FlyBodyContactBodiesPreset,
        FlyBodyJointPreset,
        FlyBodySkeleton,
    )
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D

    fly = FlyBody(name="flybody")
    skeleton = FlyBodySkeleton(
        axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
        joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL,
    )
    if is_paper_tether:
        # The historical preparation fixed the head to the thorax.  FlyBody's
        # ordinary skeleton has three passive c_thorax-c_head hinges; remove
        # those three DoFs before joints are added so this is a truly rigid
        # transform rather than a stiff approximation or a per-step reset.
        matching_head_joints = 0
        for anatomical_joint in skeleton.anatomical_joints:
            if anatomical_joint.child.name == "c_head":
                matching_head_joints += 1
                for axis in tuple(anatomical_joint.axes):
                    anatomical_joint.axes.remove(axis)
        if matching_head_joints != 1:
            raise FlyBodyDependencyError(
                "Paper tether expected one thorax-to-head anatomical joint"
            )
    flight_pose = KinematicPose(
        path=assets_dir / "model/flybody/pose/flight/yaw_roll_pitch.yaml",
        mirror_left2right=False,
    )
    fly.add_joints(skeleton, neutral_pose=flight_pose)

    wing_dofs = [dof for dof in skeleton.iter_jointdofs() if dof.child.is_wing()]
    if len(wing_dofs) != 6:
        raise FlyBodyDependencyError(
            "Reviewed FlyBody model has six wing DoFs; found {}.".format(
                len(wing_dofs)
            )
        )
    wing_dof_order = tuple(dof.name for dof in wing_dofs)
    if wing_dof_order != REVIEWED_FLYBODY_WING_AXIS_ORDER:
        raise FlyBodyDependencyError(
            "Reviewed FlyBody wing DoF order changed: expected {!r}, found {!r}.".format(
                REVIEWED_FLYBODY_WING_AXIS_ORDER, wing_dof_order
            )
        )
    # The generic FlyBody joint table contains damping=0.05 in millimetre
    # units. The released flight task explicitly overrides the source model to
    # 0.00776923 in centimetre units; rotational quantities scale by 100 in the
    # FlyGym conversion. Apply that flight-specific parameter here.
    for dof in wing_dofs:
        joint = fly.jointdof_to_mjcfjoint[dof]
        joint.stiffness = (_RELEASED_FLIGHT_WING_STIFFNESS_INTERNAL, 0.0, 0.0)
        joint.damping = (_RELEASED_FLIGHT_WING_DAMPING_INTERNAL, 0.0, 0.0)

    # Retain FlyBody's abdomen and tarsus coupling. These are passive here;
    # only the six wing DoFs receive external motor commands.
    fly.add_tendons()
    fly.add_actuators(
        wing_dofs,
        ActuatorType.MOTOR,
        forcelimited=True,
        forcerange=(
            -config.max_abs_wing_torque_n_m / _FLYGYM_TORQUE_TO_NEWTON_METRE,
            config.max_abs_wing_torque_n_m / _FLYGYM_TORQUE_TO_NEWTON_METRE,
        ),
    )
    if config.add_aerodynamic_geoms:
        _add_wing_fluid_geoms(fly, mj)

    load_cell_site_name = None
    load_cell_force_sensor_name = None
    load_cell_torque_sensor_name = None
    wing_load_cell_site_names: Dict[str, str] = {}
    wing_load_cell_force_sensor_names: Dict[str, str] = {}
    wing_load_cell_torque_sensor_names: Dict[str, str] = {}
    if is_paper_tether:
        class _PaperTetherWorld(BaseWorld):
            """Contact-free paper world with a selectable root topology."""

            def _attach_fly_mjcf(
                self, attached_fly, spawn_position, spawn_rotation
            ):
                spawn_site = self.mjcf_root.worldbody.add_site(
                    name=attached_fly.name,
                    pos=spawn_position,
                    **spawn_rotation.as_kwargs(),
                )
                self.mjcf_root.attach(
                    attached_fly.mjcf_root,
                    prefix="{}/".format(attached_fly.name),
                    site=spawn_site,
                )
                if resolved_tether_mode == "equality-reaction":
                    freejoint = attached_fly.bodyseg_to_mjcfbody[
                        attached_fly.root_segment
                    ].add_freejoint(name=attached_fly.name)
                    return {freejoint.name}
                return set()

        world = _PaperTetherWorld(name="paper_tether_world")
        world.add_fly(
            fly,
            spawn_position=(0.0, 0.0, config.spawn_height_m * 1000.0),
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        )
        root_body = fly.bodyseg_to_mjcfbody[fly.root_segment]
        if resolved_tether_mode == "equality-reaction":
            world.mjcf_root.add_equality(
                type=mj.mjtEq.mjEQ_WELD,
                objtype=mj.mjtObj.mjOBJ_BODY,
                name1=root_body.name,
                name2="",
                active=1,
                # This topology is a validator, not the authoritative meter.
                solref=(2.0 * config.timestep_s, 1.0),
                solimp=(0.999999, 0.9999999, 0.0001, 0.5, 2.0),
            )
        else:
            # Force/torque sensors report the welded child-parent interaction
            # in this site's frame.  The root body is structurally fixed: no
            # free joint or equality approximation exists in this topology.
            load_cell_site = root_body.add_site(
                name="paper_load_cell_site",
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
                size=(0.001, 0.001, 0.001),
            )
            load_cell_site_name = load_cell_site.name
            force_sensor = world.mjcf_root.add_sensor(
                name="paper_load_cell_force",
                type=mj.mjtSensor.mjSENS_FORCE,
                objtype=mj.mjtObj.mjOBJ_SITE,
                objname=load_cell_site.name,
            )
            torque_sensor = world.mjcf_root.add_sensor(
                name="paper_load_cell_torque",
                type=mj.mjtSensor.mjSENS_TORQUE,
                objtype=mj.mjtObj.mjOBJ_SITE,
                objname=load_cell_site.name,
            )
            load_cell_force_sensor_name = force_sensor.name
            load_cell_torque_sensor_name = torque_sensor.name
            if paper_wing_load_cells:
                segment_bodies = {
                    segment.name: body
                    for segment, body in fly.bodyseg_to_mjcfbody.items()
                }
                for physical_side, segment_name in (
                    ("left", "l_wing"),
                    ("right", "r_wing"),
                ):
                    wing_body = segment_bodies.get(segment_name)
                    if wing_body is None:
                        raise FlyBodyDependencyError(
                            "Paper wing load cell is missing {}".format(segment_name)
                        )
                    site = wing_body.add_site(
                        name="paper_{}_wing_load_cell_site".format(physical_side),
                        pos=(0.0, 0.0, 0.0),
                        quat=(1.0, 0.0, 0.0, 0.0),
                        size=(0.001, 0.001, 0.001),
                    )
                    force = world.mjcf_root.add_sensor(
                        name="paper_{}_wing_load_cell_force".format(physical_side),
                        type=mj.mjtSensor.mjSENS_FORCE,
                        objtype=mj.mjtObj.mjOBJ_SITE,
                        objname=site.name,
                    )
                    torque = world.mjcf_root.add_sensor(
                        name="paper_{}_wing_load_cell_torque".format(physical_side),
                        type=mj.mjtSensor.mjSENS_TORQUE,
                        objtype=mj.mjtObj.mjOBJ_SITE,
                        objname=site.name,
                    )
                    wing_load_cell_site_names[physical_side] = site.name
                    wing_load_cell_force_sensor_names[physical_side] = force.name
                    wing_load_cell_torque_sensor_names[physical_side] = torque.name
    else:
        world = FlatGroundWorld(name="flight_world")
        world.add_fly(
            fly,
            spawn_position=(0.0, 0.0, config.spawn_height_m * 1000.0),
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
            bodysegs_with_ground_contact=FlyBodyContactBodiesPreset.LEGS_ONLY,
            add_ground_contact_sensors=False,
        )
    world.mjcf_root.option.density = _AIR_DENSITY_G_MM3
    world.mjcf_root.option.viscosity = _AIR_VISCOSITY_G_MM_S
    simulation = Simulation(world, timestep=config.timestep_s)
    simulation.reset()

    if not np.isclose(simulation.mj_model.opt.density, _AIR_DENSITY_G_MM3):
        raise FlyBodyDependencyError("FlyBody air-density unit correction was not applied")
    if not np.isclose(simulation.mj_model.opt.viscosity, _AIR_VISCOSITY_G_MM_S):
        raise FlyBodyDependencyError("FlyBody air-viscosity unit correction was not applied")

    joint_order = fly.get_jointdofs_order()
    wing_joint_indices = np.asarray(
        [joint_order.index(dof) for dof in wing_dofs], dtype=np.int64
    )
    wing_qpos_adrs = []
    wing_dof_adrs = []
    for dof in wing_dofs:
        compiled_joint = simulation.mj_model.joint("flybody/" + dof.name)
        wing_qpos_adrs.append(int(simulation.mj_model.jnt_qposadr[compiled_joint.id]))
        wing_dof_adrs.append(int(simulation.mj_model.jnt_dofadr[compiled_joint.id]))
    free_joint_ids = np.flatnonzero(simulation.mj_model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if resolved_tether_mode == "fixed-load-cell":
        if len(free_joint_ids) != 0:
            raise FlyBodyDependencyError(
                "Fixed load-cell topology must not contain a free root joint"
            )
        free_body_id = int(
            mj.mj_name2id(
                simulation.mj_model,
                mj.mjtObj.mjOBJ_BODY,
                root_body.name,
            )
        )
        if free_body_id < 1:
            raise FlyBodyDependencyError("fixed paper root body is unavailable")
        qpos_adr = None
        dof_adr = None
    else:
        if len(free_joint_ids) != 1:
            raise FlyBodyDependencyError(
                "Expected exactly one free body joint; found {}.".format(
                    len(free_joint_ids)
                )
            )
        free_joint_id = int(free_joint_ids[0])
        free_body_id = int(simulation.mj_model.jnt_bodyid[free_joint_id])
        qpos_adr = int(simulation.mj_model.jnt_qposadr[free_joint_id])
        dof_adr = int(simulation.mj_model.jnt_dofadr[free_joint_id])
    head_body_id = int(
        mj.mj_name2id(
            simulation.mj_model,
            mj.mjtObj.mjOBJ_BODY,
            "flybody/c_head",
        )
    )
    if head_body_id < 0:
        raise FlyBodyDependencyError("compiled FlyBody is missing flybody/c_head")

    fluid_geom_names = [
        simulation.mj_model.geom(i).name
        for i in range(simulation.mj_model.ngeom)
        if simulation.mj_model.geom_fluid[i, 0] > 0
    ]
    if config.add_aerodynamic_geoms and len(fluid_geom_names) != 2:
        raise FlyBodyDependencyError(
            "Expected two active wing-fluid geoms; found {!r}.".format(
                fluid_geom_names
            )
        )
    for geom_name in fluid_geom_names:
        geom_id = simulation.mj_model.geom(geom_name).id
        if (
            simulation.mj_model.geom_contype[geom_id] != 0
            or simulation.mj_model.geom_conaffinity[geom_id] != 0
        ):
            raise FlyBodyDependencyError(
                "wing-fluid aerodynamic proxy unexpectedly participates in contact"
            )
    load_cell_site_id = None
    load_cell_force_sensor_id = None
    load_cell_torque_sensor_id = None
    wing_load_cell_site_ids: Dict[str, int] = {}
    wing_load_cell_force_sensor_ids: Dict[str, int] = {}
    wing_load_cell_torque_sensor_ids: Dict[str, int] = {}
    wing_body_ids: Dict[str, int] = {}
    if is_paper_tether:
        ground_geom_id = -1
        ground_geom_name = None
        ground_contact_pair_count = 0
        if resolved_tether_mode == "equality-reaction":
            if int(simulation.mj_model.neq) != 1:
                raise FlyBodyDependencyError(
                    "Paper equality meter requires exactly one equality constraint"
                )
            if int(simulation.mj_model.eq_type[0]) != int(mj.mjtEq.mjEQ_WELD):
                raise FlyBodyDependencyError("Paper tether equality is not a weld")
            tether_equality_id = 0
        else:
            if int(simulation.mj_model.neq) != 0:
                raise FlyBodyDependencyError(
                    "Fixed load-cell topology must not contain equality constraints"
                )
            tether_equality_id = None
            load_cell_site_id = int(
                mj.mj_name2id(
                    simulation.mj_model,
                    mj.mjtObj.mjOBJ_SITE,
                    load_cell_site_name,
                )
            )
            load_cell_force_sensor_id = int(
                mj.mj_name2id(
                    simulation.mj_model,
                    mj.mjtObj.mjOBJ_SENSOR,
                    load_cell_force_sensor_name,
                )
            )
            load_cell_torque_sensor_id = int(
                mj.mj_name2id(
                    simulation.mj_model,
                    mj.mjtObj.mjOBJ_SENSOR,
                    load_cell_torque_sensor_name,
                )
            )
            if min(
                load_cell_site_id,
                load_cell_force_sensor_id,
                load_cell_torque_sensor_id,
            ) < 0:
                raise FlyBodyDependencyError("compiled load-cell sensors are unavailable")
            if (
                int(simulation.mj_model.sensor_dim[load_cell_force_sensor_id]) != 3
                or int(simulation.mj_model.sensor_dim[load_cell_torque_sensor_id]) != 3
            ):
                raise FlyBodyDependencyError("load-cell sensor dimensions changed")
            for physical_side, body_name in (
                ("left", "flybody/l_wing"),
                ("right", "flybody/r_wing"),
            ):
                wing_body_ids[physical_side] = int(
                    mj.mj_name2id(
                        simulation.mj_model,
                        mj.mjtObj.mjOBJ_BODY,
                        body_name,
                    )
                )
                if paper_wing_load_cells:
                    wing_load_cell_site_ids[physical_side] = int(
                        mj.mj_name2id(
                            simulation.mj_model,
                            mj.mjtObj.mjOBJ_SITE,
                            wing_load_cell_site_names[physical_side],
                        )
                    )
                    wing_load_cell_force_sensor_ids[physical_side] = int(
                        mj.mj_name2id(
                            simulation.mj_model,
                            mj.mjtObj.mjOBJ_SENSOR,
                            wing_load_cell_force_sensor_names[physical_side],
                        )
                    )
                    wing_load_cell_torque_sensor_ids[physical_side] = int(
                        mj.mj_name2id(
                            simulation.mj_model,
                            mj.mjtObj.mjOBJ_SENSOR,
                            wing_load_cell_torque_sensor_names[physical_side],
                        )
                    )
                    ids = (
                        wing_body_ids[physical_side],
                        wing_load_cell_site_ids[physical_side],
                        wing_load_cell_force_sensor_ids[physical_side],
                        wing_load_cell_torque_sensor_ids[physical_side],
                    )
                    if min(ids) < 0:
                        raise FlyBodyDependencyError(
                            "compiled {} wing load cell is unavailable".format(
                                physical_side
                            )
                        )
                    if (
                        int(
                            simulation.mj_model.sensor_dim[
                                wing_load_cell_force_sensor_ids[physical_side]
                            ]
                        )
                        != 3
                        or int(
                            simulation.mj_model.sensor_dim[
                                wing_load_cell_torque_sensor_ids[physical_side]
                            ]
                        )
                        != 3
                    ):
                        raise FlyBodyDependencyError(
                            "{} wing load-cell sensor dimensions changed".format(
                                physical_side
                            )
                        )
    else:
        ground_geom_id = int(
            mj.mj_name2id(
                simulation.mj_model,
                mj.mjtObj.mjOBJ_GEOM,
                "ground_plane",
            )
        )
        if ground_geom_id < 0:
            raise FlyBodyDependencyError(
                "Reviewed FlyBody world is missing the named ground_plane geom"
            )
        if int(simulation.mj_model.geom_bodyid[ground_geom_id]) != 0:
            raise FlyBodyDependencyError("ground_plane must belong to the MuJoCo world body")
        ground_contact_pair_count = sum(
            int(
                int(simulation.mj_model.pair_geom1[index]) == ground_geom_id
                or int(simulation.mj_model.pair_geom2[index]) == ground_geom_id
            )
            for index in range(int(simulation.mj_model.npair))
        )
        if ground_contact_pair_count != 48:
            raise FlyBodyDependencyError(
                "Reviewed legs-only topology requires 48 explicit ground pairs; found {}".format(
                    ground_contact_pair_count
                )
            )
        ground_geom_name = "ground_plane"
        tether_equality_id = None
    for dof in wing_dofs:
        joint = simulation.mj_model.joint("flybody/" + dof.name)
        dof_index = int(simulation.mj_model.jnt_dofadr[joint.id])
        if not np.isclose(
            simulation.mj_model.jnt_stiffness[joint.id],
            _RELEASED_FLIGHT_WING_STIFFNESS_INTERNAL,
        ):
            raise FlyBodyDependencyError("released flight wing stiffness was not applied")
        if not np.isclose(
            simulation.mj_model.dof_damping[dof_index],
            _RELEASED_FLIGHT_WING_DAMPING_INTERNAL,
        ):
            raise FlyBodyDependencyError("released flight wing damping was not applied")

    return {
        "config": config,
        "versions": versions,
        "fly": fly,
        "simulation": simulation,
        "actuator_type": ActuatorType.MOTOR,
        "wing_dofs": wing_dofs,
        "wing_joint_indices": wing_joint_indices,
        "wing_qpos_adrs": np.asarray(wing_qpos_adrs, dtype=np.int64),
        "wing_dof_adrs": np.asarray(wing_dof_adrs, dtype=np.int64),
        "root_qpos_adr": qpos_adr,
        "root_dof_adr": dof_adr,
        "root_body_id": free_body_id,
        "head_body_id": head_body_id,
        "fluid_geom_names": fluid_geom_names,
        "ground_geom_id": ground_geom_id,
        "ground_geom_name": ground_geom_name,
        "ground_contact_pair_count": ground_contact_pair_count,
        "paper_tether": bool(is_paper_tether),
        "paper_tether_mode": resolved_tether_mode,
        "head_to_thorax_rigid": bool(is_paper_tether),
        "tether_equality_id": tether_equality_id,
        "load_cell_site_id": load_cell_site_id,
        "load_cell_force_sensor_id": load_cell_force_sensor_id,
        "load_cell_torque_sensor_id": load_cell_torque_sensor_id,
        "wing_load_cell_side_order": ("left", "right"),
        "wing_load_cell_site_ids": wing_load_cell_site_ids,
        "wing_load_cell_force_sensor_ids": wing_load_cell_force_sensor_ids,
        "wing_load_cell_torque_sensor_ids": wing_load_cell_torque_sensor_ids,
        "wing_body_ids": wing_body_ids,
        "paper_wing_load_cells": bool(
            resolved_tether_mode == "fixed-load-cell" and paper_wing_load_cells
        ),
        "tendon_count": int(simulation.mj_model.ntendon),
        "compiled_model_fingerprint": _compiled_model_fingerprint(
            simulation.mj_model
        ),
        "dependency_record_sha256": dependency_record_fingerprints(),
    }


def _run_built_trace(
    bundle: Dict[str, Any],
    n_steps: int,
    torque_controller: Any,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> FlyBodyTrace:
    """Run a controller against an already compiled worker model."""

    config = bundle["config"]
    simulation = bundle["simulation"]
    fly = bundle["fly"]
    actuator_type = bundle["actuator_type"]
    time_s = np.arange(n_steps + 1, dtype=float) * config.timestep_s
    body_position_m = np.empty((n_steps + 1, 3), dtype=float)
    body_quaternion = np.empty((n_steps + 1, 4), dtype=float)
    body_linear_velocity = np.empty((n_steps + 1, 3), dtype=float)
    body_angular_velocity = np.empty((n_steps + 1, 3), dtype=float)
    wing_angles = np.empty((n_steps + 1, 6), dtype=float)
    wing_velocity = np.empty((n_steps + 1, 6), dtype=float)
    actuator_torque_n_m = np.empty((n_steps + 1, 6), dtype=float)
    root_fluid_force_n = np.empty((n_steps + 1, 3), dtype=float)
    root_fluid_torque_n_m = np.empty((n_steps + 1, 3), dtype=float)

    def sample(index: int) -> None:
        qpos_adr = bundle["root_qpos_adr"]
        dof_adr = bundle["root_dof_adr"]
        data = simulation.mj_data
        body_position_m[index] = data.qpos[qpos_adr : qpos_adr + 3] / 1000.0
        body_quaternion[index] = data.qpos[qpos_adr + 3 : qpos_adr + 7]
        body_linear_velocity[index] = data.qvel[dof_adr : dof_adr + 3] / 1000.0
        body_angular_velocity[index] = data.qvel[dof_adr + 3 : dof_adr + 6]
        all_angles = simulation.get_joint_angles(fly.name)
        all_velocities = simulation.get_joint_velocities(fly.name)
        wing_angles[index] = all_angles[bundle["wing_joint_indices"]]
        wing_velocity[index] = all_velocities[bundle["wing_joint_indices"]]
        actuator_torque_n_m[index] = (
            simulation.get_actuator_forces(fly.name, actuator_type)
            * _FLYGYM_TORQUE_TO_NEWTON_METRE
        )
        root_fluid_force_n[index] = (
            data.qfrc_fluid[dof_adr : dof_adr + 3] * _FLYGYM_FORCE_TO_NEWTON
        )
        root_fluid_torque_n_m[index] = (
            data.qfrc_fluid[dof_adr + 3 : dof_adr + 6]
            * _FLYGYM_TORQUE_TO_NEWTON_METRE
        )

    sample(0)
    for step in range(n_steps):
        torque_n_m = np.asarray(torque_controller(step, bundle), dtype=float)
        if torque_n_m.shape != (6,) or not np.all(np.isfinite(torque_n_m)):
            raise ValueError("wing torque controller must produce six finite values")
        if np.max(np.abs(torque_n_m), initial=0.0) > config.max_abs_wing_torque_n_m:
            raise ValueError("wing torque controller exceeded configured actuator limit")
        simulation.set_actuator_inputs(
            fly.name,
            actuator_type,
            torque_n_m / _FLYGYM_TORQUE_TO_NEWTON_METRE,
        )
        simulation.step()
        sample(step + 1)

    metadata: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": "FlyGym/FlyBody",
        "status": "exploratory",
        "worker_versions": bundle["versions"],
        "config": asdict(config),
        "wing_dof_order": [dof.name for dof in bundle["wing_dofs"]],
        "fluid_geom_names": bundle["fluid_geom_names"],
        "fluid_geoms_contact_disabled": True,
        "tendon_count": bundle["tendon_count"],
        "compiled_model_fingerprint": bundle["compiled_model_fingerprint"],
        "dependency_record_sha256": bundle["dependency_record_sha256"],
        "ground_contact_topology": "legs_only",
        "released_policy_topology_equivalent": False,
        "released_flight_wing_stiffness_n_m_per_rad": 1.0e-9,
        "released_flight_wing_damping_n_m_s_per_rad": 7.76923e-10,
        "air_density_kg_m3": 1.28,
        "air_viscosity_pa_s": 1.85e-5,
        "flygym_2_1_air_unit_correction_applied": True,
        "flygym_internal_length_unit": "millimetre",
        "public_length_unit": "metre",
        "body_state_reference": (
            "FlyBody free-joint root/thorax frame; not whole-fly centre of mass"
        ),
        "root_fluid_force_sample_semantics": (
            "sample i>=1 is the MuJoCo force applied over interval "
            "[time_s[i-1], time_s[i]]; integrate with dt*sum(samples[1:])"
        ),
        "source": {
            "flygym_tag": "v2.1.0",
            "flygym_commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
            "flybody_aerodynamics_commit_reviewed": (
                "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
            ),
        },
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return FlyBodyTrace(
        time_s=time_s,
        body_position_m=body_position_m,
        body_quaternion_wxyz=body_quaternion,
        body_linear_velocity_m_s=body_linear_velocity,
        body_angular_velocity_rad_s=body_angular_velocity,
        wing_angles_rad=wing_angles,
        wing_angular_velocity_rad_s=wing_velocity,
        actuator_torque_n_m=actuator_torque_n_m,
        root_fluid_force_n=root_fluid_force_n,
        root_fluid_torque_n_m=root_fluid_torque_n_m,
        metadata=metadata,
    )


def run_torque_trace(
    torque_trace: Sequence[Sequence[float]],
    config: Optional[FlyBodyWorkerConfig] = None,
) -> FlyBodyTrace:
    """Run a six-column generalized wing-torque trace through FlyGym/FlyBody.

    Column order is derived from FlyBody itself and recorded in metadata. Inputs
    are SI newton-metres and must already be the output of a calibrated
    muscle-to-hinge model.
    """

    resolved_config = config or FlyBodyWorkerConfig()
    torques = np.asarray(torque_trace, dtype=float)
    if torques.ndim != 2 or torques.shape[1] != 6:
        raise ValueError("torque_trace must have shape (steps, 6)")
    if not np.all(np.isfinite(torques)):
        raise ValueError("torque_trace must contain finite values")
    if np.max(np.abs(torques), initial=0.0) > resolved_config.max_abs_wing_torque_n_m:
        raise ValueError("torque_trace exceeds configured actuator force limit")
    bundle = build_flybody_simulation(resolved_config)
    return _run_built_trace(
        bundle,
        torques.shape[0],
        lambda step, _bundle: torques[step],
        {"actuation_source": "external_generalized_torque_trace"},
    )


def _analytic_wing_pattern(
    phase_rad: float,
    left_scale: float,
    right_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the beat-local analytic fallback and its phase derivative.

    Upstream FlyBody creates one 0..2*pi lookup-table cycle and repeats it.
    In particular, its 1.5-harmonic roll term restarts on every wingbeat.
    Evaluating that term at an unbounded phase silently alternates the roll
    waveform between beats, so this continuous approximation explicitly wraps
    phase at the cycle boundary. It remains a compatibility waveform rather
    than the measured wing pattern used by the released flight policy.
    """

    phase_rad = float(np.mod(phase_rad, 2.0 * np.pi))

    oscillation = np.array(
        [
            1.1 * np.sin(phase_rad - np.pi / 2.0),
            0.25 * np.sin(1.5 * phase_rad),
            1.35 * np.sin(phase_rad),
        ],
        dtype=float,
    )
    offset = np.array((0.3, -0.1, 0.8), dtype=float)
    derivative = np.array(
        [
            1.1 * np.cos(phase_rad - np.pi / 2.0),
            0.375 * np.cos(1.5 * phase_rad),
            1.35 * np.cos(phase_rad),
        ],
        dtype=float,
    )
    angles = np.concatenate(
        (offset + left_scale * oscillation, offset + right_scale * oscillation)
    )
    derivatives = np.concatenate((left_scale * derivative, right_scale * derivative))
    return angles, derivatives


def run_analytic_wingbeat_smoke(
    duration_s: float = 0.020,
    wingbeat_frequency_hz: float = 218.0,
    left_stroke_scale: float = 1.0,
    right_stroke_scale: float = 1.0,
    initial_phase_rad: float = 0.0,
    control_timestep_s: float = _UPSTREAM_FLIGHT_CONTROL_TIMESTEP_S,
    config: Optional[FlyBodyWorkerConfig] = None,
) -> FlyBodyTrace:
    """Exercise actual FlyBody aerodynamics with its analytic fallback wingbeat.

    This ports the formula used when upstream FlyBody has no measured base
    pattern. It is a compatibility smoke test, not the released learned
    straight-flight or saccade policy and not a neuromuscular calibration.
    """

    if duration_s <= 0.0 or wingbeat_frequency_hz <= 0.0:
        raise ValueError("duration and wingbeat frequency must be positive")
    if control_timestep_s <= 0.0:
        raise ValueError("control timestep must be positive")
    if left_stroke_scale <= 0.0 or right_stroke_scale <= 0.0:
        raise ValueError("stroke scales must be positive")
    resolved_config = config or FlyBodyWorkerConfig(timestep_s=5.0e-5)
    steps_float = duration_s / resolved_config.timestep_s
    n_steps = int(round(steps_float))
    if not np.isclose(steps_float, n_steps, rtol=0.0, atol=1e-9):
        raise ValueError("duration must be an integer multiple of physics timestep")
    control_ratio = control_timestep_s / resolved_config.timestep_s
    steps_per_control = int(round(control_ratio))
    if steps_per_control < 1 or not np.isclose(
        control_ratio, steps_per_control, rtol=0.0, atol=1e-9
    ):
        raise ValueError("control timestep must be an integer multiple of physics timestep")
    bundle = build_flybody_simulation(resolved_config)
    simulation = bundle["simulation"]

    initial_angles, derivatives_per_phase = _analytic_wing_pattern(
        initial_phase_rad, left_stroke_scale, right_stroke_scale
    )
    omega = 2.0 * np.pi * wingbeat_frequency_hz
    simulation.mj_data.qpos[bundle["wing_qpos_adrs"]] = initial_angles
    simulation.mj_data.qvel[bundle["wing_dof_adrs"]] = derivatives_per_phase * omega
    import mujoco as mj

    mj.mj_forward(simulation.mj_model, simulation.mj_data)
    proportional_gain_n_m_per_rad = 1.8e-6

    # FlyBody's released flight task uses a 0.2 ms controller clock and a
    # 0.05 ms physics clock.  Hold each feedback command between control ticks
    # so timestep-refinement experiments vary physics alone rather than also
    # changing the feedback bandwidth.
    held_torque = np.zeros(6, dtype=float)
    last_control_index = -1

    def controller(step: int, active_bundle: Dict[str, Any]) -> np.ndarray:
        nonlocal held_torque, last_control_index
        control_index = step // steps_per_control
        if control_index != last_control_index:
            # Upstream WPG.step() advances its lookup-table index before the
            # command is applied to the following physics interval.
            phase = (
                initial_phase_rad
                + omega * (control_index + 1) * control_timestep_s
            )
            target, _ = _analytic_wing_pattern(
                phase, left_stroke_scale, right_stroke_scale
            )
            measured = active_bundle["simulation"].mj_data.qpos[
                active_bundle["wing_qpos_adrs"]
            ]
            held_torque = proportional_gain_n_m_per_rad * (target - measured)
            last_control_index = control_index
        return held_torque

    return _run_built_trace(
        bundle,
        n_steps,
        controller,
        {
            "actuation_source": (
                "beat_wrapped_analytic_fallback_formula_proportional_tracking"
            ),
            "baseline_class": "compatibility_smoke_only",
            "waveform_semantics": (
                "continuous beat-local approximation of upstream fallback lookup table; "
                "not measured WPG"
            ),
            "released_policy_regression_status": "not_run",
            "wingbeat_frequency_hz": wingbeat_frequency_hz,
            "left_stroke_scale": left_stroke_scale,
            "right_stroke_scale": right_stroke_scale,
            "initial_phase_rad": initial_phase_rad,
            "control_timestep_s": control_timestep_s,
            "actuation_hold": "zero_order_hold",
            "control_phase_semantics": "advance_then_hold",
            "proportional_gain_n_m_per_rad": proportional_gain_n_m_per_rad,
        },
    )


class FlyBodyPhysicsAdapter:
    """Stateful implementation of the flight core's external-physics protocol.

    FlyBody computes its own aerodynamic and contact forces. The reduced
    engine's force and torque arguments are therefore accepted only to satisfy
    the common protocol and are deliberately not applied a second time.
    """

    backend_name = "flybody"
    aerodynamic_owner = "flybody"
    body_state_reference = (
        "FlyBody free-joint root/thorax frame; not whole-fly centre of mass"
    )

    def __init__(
        self,
        torque_mapper: WingTorqueMapper,
        config: Optional[FlyBodyWorkerConfig] = None,
    ) -> None:
        self.torque_mapper = torque_mapper
        self.bundle = build_flybody_simulation(config)
        self._default_initial_state = self._state()
        self._last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._last_wrench = AerodynamicWrench(
            force_body_n=np.zeros(3, dtype=float),
            torque_body_n_m=np.zeros(3, dtype=float),
            left_force_body_n=np.zeros(3, dtype=float),
            right_force_body_n=np.zeros(3, dtype=float),
            mechanical_power_w=0.0,
        )

    def default_initial_state(self) -> RigidBodyState:
        """Return the exact compiled airborne root state in public SI units."""

        return self._default_initial_state.copy()

    def provenance_metadata(self) -> Mapping[str, Any]:
        """Return the exact compiled worker receipt for immutable run manifests."""

        worker_image_digest = declared_worker_image_digest()
        return {
            "schema_version": "1.0.0",
            "engine": "FlyGym/FlyBody with native MuJoCo",
            "worker_versions": dict(self.bundle["versions"]),
            "worker_image_digest": worker_image_digest,
            "worker_image_digest_status": (
                "declared_oci_digest"
                if worker_image_digest is not None
                else "not_declared"
            ),
            "worker_dependency_lock": {
                "name": WORKER_DEPENDENCY_LOCK_NAME,
                "sha256": worker_dependency_lock_sha256(),
            },
            "dependency_record_sha256": dict(
                self.bundle["dependency_record_sha256"]
            ),
            "compiled_model_fingerprint": dict(
                self.bundle["compiled_model_fingerprint"]
            ),
            "worker_config": asdict(self.bundle["config"]),
            "wing_dof_order": [dof.name for dof in self.bundle["wing_dofs"]],
            "fluid_geom_names": list(self.bundle["fluid_geom_names"]),
            "fluid_geoms_contact_disabled": True,
            "ground_contact_topology": "legs_only",
            "ground_contact_telemetry": {
                "kind": (
                    "exact MuJoCo pre-integration transition contact-point "
                    "count involving ground_plane"
                ),
                "sample_semantics": (
                    "after reset: mj_forward state; after mj_step: contact list "
                    "computed from the pre-integration state and used for the "
                    "completed transition, not the post-step pose"
                ),
                "ground_geom_name": self.bundle["ground_geom_name"],
                "configured_pair_count": self.bundle[
                    "ground_contact_pair_count"
                ],
            },
            "released_policy_topology_equivalent": False,
            "tendon_count": int(self.bundle["tendon_count"]),
            "source": {
                "flygym_tag": "v2.1.0",
                "flygym_commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
                "flybody_aerodynamics_commit_reviewed": (
                    "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
                ),
            },
            "licenses": dependency_license_metadata(),
            "body_state_reference": self.body_state_reference,
            "root_fluid_wrench_scope": "root-total; no reviewed per-wing decomposition",
            "checkpoint_contract": {
                "schema_version": _FLYBODY_CHECKPOINT_SCHEMA_VERSION,
                "state_spec": _FLYBODY_CHECKPOINT_STATE_SPEC,
                "float_encoding": _FLYBODY_CHECKPOINT_FLOAT_ENCODING,
                "compatibility_bound_to_compiled_model": True,
                "fresh_adapter_exact_reentry_required": True,
            },
            "public_units": "SI",
        }

    @property
    def wing_joint_order(self) -> Tuple[str, ...]:
        return tuple(dof.name for dof in self.bundle["wing_dofs"])

    def wing_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return measured six-axis FlyBody wing qpos/qvel in reviewed order."""

        simulation = self.bundle["simulation"]
        angles = simulation.mj_data.qpos[self.bundle["wing_qpos_adrs"]].copy()
        all_velocities = simulation.get_joint_velocities(self.bundle["fly"].name)
        velocities = all_velocities[self.bundle["wing_joint_indices"]].copy()
        return angles, velocities

    def whole_fly_com_position_m(self) -> np.ndarray:
        """Return MuJoCo's articulated subtree center of mass in world SI units."""

        return (
            self.bundle["simulation"].mj_data.subtree_com[
                self.bundle["root_body_id"]
            ].copy()
            / 1000.0
        )

    def ground_contact_count(self) -> int:
        """Return ground points used for reset/preceding-transition dynamics.

        This deliberately counts every ground-involving entry in ``data.ncon``.
        Immediately after reset these were computed by ``mj_forward`` at the
        reset state. Immediately after ``mj_step`` they correspond to the
        pre-integration configuration used for that completed transition, not
        a fresh collision query at the advanced qpos. This is a conservative
        constraint-influence detector and does not claim positive force.
        """

        data = self.bundle["simulation"].mj_data
        ground_geom_id = int(self.bundle["ground_geom_id"])
        return sum(
            int(
                int(data.contact[index].geom[0]) == ground_geom_id
                or int(data.contact[index].geom[1]) == ground_geom_id
            )
            for index in range(int(data.ncon))
        )

    def _state(self) -> RigidBodyState:
        simulation = self.bundle["simulation"]
        qpos_adr = self.bundle["root_qpos_adr"]
        dof_adr = self.bundle["root_dof_adr"]
        return RigidBodyState(
            position_world_m=(
                simulation.mj_data.qpos[qpos_adr : qpos_adr + 3].copy() / 1000.0
            ),
            velocity_world_m_s=(
                simulation.mj_data.qvel[dof_adr : dof_adr + 3].copy() / 1000.0
            ),
            quaternion_body_to_world=(
                simulation.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
            ),
            angular_velocity_body_rad_s=(
                simulation.mj_data.qvel[dof_adr + 3 : dof_adr + 6].copy()
            ),
        )

    def reset(self, initial_state: RigidBodyState) -> None:
        import mujoco as mj

        initial_state.validate()
        simulation = self.bundle["simulation"]
        simulation.reset()
        qpos_adr = self.bundle["root_qpos_adr"]
        dof_adr = self.bundle["root_dof_adr"]
        simulation.mj_data.qpos[qpos_adr : qpos_adr + 3] = (
            np.asarray(initial_state.position_world_m, dtype=float) * 1000.0
        )
        simulation.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
            initial_state.quaternion_body_to_world, dtype=float
        )
        simulation.mj_data.qvel[dof_adr : dof_adr + 3] = (
            np.asarray(initial_state.velocity_world_m_s, dtype=float) * 1000.0
        )
        simulation.mj_data.qvel[dof_adr + 3 : dof_adr + 6] = np.asarray(
            initial_state.angular_velocity_body_rad_s, dtype=float
        )
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        self._last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._last_wrench = AerodynamicWrench(
            force_body_n=np.zeros(3, dtype=float),
            torque_body_n_m=np.zeros(3, dtype=float),
            left_force_body_n=np.zeros(3, dtype=float),
            right_force_body_n=np.zeros(3, dtype=float),
            mechanical_power_w=0.0,
        )

    @property
    def last_actuator_torque_n_m(self) -> np.ndarray:
        return self._last_actuator_torque_n_m.copy()

    def aerodynamic_wrench(self) -> AerodynamicWrench:
        """Return root-total FlyBody fluid telemetry from the preceding step.

        MuJoCo exposes the total generalized fluid wrench on the free joint.
        It does not expose a reviewed per-wing decomposition at this adapter
        boundary, so the two per-wing fields remain zero and must not be used
        for a bilateral aerodynamic claim.
        """

        return AerodynamicWrench(
            force_body_n=self._last_wrench.force_body_n.copy(),
            torque_body_n_m=self._last_wrench.torque_body_n_m.copy(),
            left_force_body_n=self._last_wrench.left_force_body_n.copy(),
            right_force_body_n=self._last_wrench.right_force_body_n.copy(),
            mechanical_power_w=float(self._last_wrench.mechanical_power_w),
        )

    def checkpoint(self) -> FlyBodyPhysicsCheckpoint:
        """Capture all integration state needed for deterministic re-entry.

        ``mjSTATE_INTEGRATION`` is MuJoCo's supported complete integration
        boundary.  In addition to time, generalized positions/velocities and
        actuator state, it includes warm-start, controls, applied forces,
        equality state, mocap/user/plugin state where present.  Encoding is
        fixed to little-endian float64 so checkpoint bytes do not depend on the
        worker host's native byte order.
        """

        import mujoco as mj

        simulation = self.bundle["simulation"]
        state_spec = mj.mjtState.mjSTATE_INTEGRATION
        state_size = int(mj.mj_stateSize(simulation.mj_model, state_spec))
        state = np.empty(state_size, dtype=np.float64)
        mj.mj_getState(
            simulation.mj_model,
            simulation.mj_data,
            state,
            state_spec,
        )
        if not np.all(np.isfinite(state)):
            raise FlyBodyDependencyError(
                "MuJoCo integration state contains a non-finite value"
            )
        wrench = self.aerodynamic_wrench()
        payload: Dict[str, Any] = {
            "schema_version": _FLYBODY_CHECKPOINT_SCHEMA_VERSION,
            "state_spec": _FLYBODY_CHECKPOINT_STATE_SPEC,
            "float_encoding": _FLYBODY_CHECKPOINT_FLOAT_ENCODING,
            "state_size": state_size,
            "simulation_state_b64": base64.b64encode(
                np.asarray(state, dtype="<f8").tobytes(order="C")
            ).decode("ascii"),
            "compiled_model_sha256": self.bundle[
                "compiled_model_fingerprint"
            ]["sha256"],
            "worker_versions": dict(self.bundle["versions"]),
            "physics_timestep_s": float(self.bundle["config"].timestep_s),
            "wing_dof_order": list(self.wing_joint_order),
            "last_actuator_torque_n_m": self._last_actuator_torque_n_m.tolist(),
            "last_aerodynamic_wrench": {
                "force_body_n": wrench.force_body_n.tolist(),
                "torque_body_n_m": wrench.torque_body_n_m.tolist(),
                "left_force_body_n": wrench.left_force_body_n.tolist(),
                "right_force_body_n": wrench.right_force_body_n.tolist(),
                "mechanical_power_w": float(wrench.mechanical_power_w),
            },
        }
        payload["payload_sha256"] = "sha256:" + hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return FlyBodyPhysicsCheckpoint(payload)

    def restore_checkpoint(
        self, checkpoint: FlyBodyPhysicsCheckpoint
    ) -> RigidBodyState:
        """Restore a checkpoint after validating every compatibility receipt.

        All validation and decoding completes before ``mjData`` is mutated.
        This fail-closed ordering is important for queued workers: a corrupt or
        cross-model checkpoint cannot leave a reusable adapter half-restored.
        """

        import mujoco as mj

        if not isinstance(checkpoint, FlyBodyPhysicsCheckpoint):
            raise TypeError("checkpoint must be FlyBodyPhysicsCheckpoint")
        # Re-validate a detached copy here as the public payload contains
        # ordinary JSON containers that a caller could have mutated after the
        # frozen wrapper itself was constructed.
        checkpoint = FlyBodyPhysicsCheckpoint(checkpoint.to_dict())
        payload = checkpoint.payload
        if payload["schema_version"] != _FLYBODY_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("FlyBody checkpoint schema version mismatch")
        if payload["state_spec"] != _FLYBODY_CHECKPOINT_STATE_SPEC:
            raise ValueError("FlyBody checkpoint state specification mismatch")
        if payload["float_encoding"] != _FLYBODY_CHECKPOINT_FLOAT_ENCODING:
            raise ValueError("FlyBody checkpoint float encoding mismatch")
        if payload["compiled_model_sha256"] != self.bundle[
            "compiled_model_fingerprint"
        ]["sha256"]:
            raise ValueError("FlyBody checkpoint compiled model mismatch")
        if payload["worker_versions"] != dict(self.bundle["versions"]):
            raise ValueError("FlyBody checkpoint worker version mismatch")
        if payload["wing_dof_order"] != list(self.wing_joint_order):
            raise ValueError("FlyBody checkpoint wing DoF order mismatch")
        if not np.isclose(
            float(payload["physics_timestep_s"]),
            float(self.bundle["config"].timestep_s),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("FlyBody checkpoint physics timestep mismatch")

        simulation = self.bundle["simulation"]
        state_spec = mj.mjtState.mjSTATE_INTEGRATION
        expected_state_size = int(
            mj.mj_stateSize(simulation.mj_model, state_spec)
        )
        state_size = payload["state_size"]
        if (
            isinstance(state_size, bool)
            or not isinstance(state_size, int)
            or state_size != expected_state_size
        ):
            raise ValueError("FlyBody checkpoint state size mismatch")
        try:
            state_bytes = base64.b64decode(
                payload["simulation_state_b64"], validate=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("FlyBody checkpoint state encoding is invalid") from exc
        if len(state_bytes) != expected_state_size * np.dtype("<f8").itemsize:
            raise ValueError("FlyBody checkpoint state byte length mismatch")
        state = np.frombuffer(state_bytes, dtype="<f8").astype(
            np.float64, copy=True
        )
        if not np.all(np.isfinite(state)):
            raise ValueError("FlyBody checkpoint state contains a non-finite value")

        actuator = np.asarray(
            payload["last_actuator_torque_n_m"], dtype=float
        )
        wrench_payload = payload["last_aerodynamic_wrench"]
        expected_wrench_fields = {
            "force_body_n",
            "torque_body_n_m",
            "left_force_body_n",
            "right_force_body_n",
            "mechanical_power_w",
        }
        if (
            actuator.shape != (6,)
            or not np.all(np.isfinite(actuator))
            or not isinstance(wrench_payload, Mapping)
            or set(wrench_payload) != expected_wrench_fields
        ):
            raise ValueError("FlyBody checkpoint telemetry shape mismatch")
        wrench_vectors = {
            name: np.asarray(wrench_payload[name], dtype=float)
            for name in (
                "force_body_n",
                "torque_body_n_m",
                "left_force_body_n",
                "right_force_body_n",
            )
        }
        if any(
            vector.shape != (3,) or not np.all(np.isfinite(vector))
            for vector in wrench_vectors.values()
        ):
            raise ValueError("FlyBody checkpoint wrench vectors are invalid")
        mechanical_power_w = float(wrench_payload["mechanical_power_w"])
        if not np.isfinite(mechanical_power_w):
            raise ValueError("FlyBody checkpoint mechanical power is invalid")

        # Mutation begins only after the entire payload has been validated.
        mj.mj_setState(
            simulation.mj_model,
            simulation.mj_data,
            state,
            state_spec,
        )
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        self._last_actuator_torque_n_m = actuator.copy()
        self._last_wrench = AerodynamicWrench(
            force_body_n=wrench_vectors["force_body_n"].copy(),
            torque_body_n_m=wrench_vectors["torque_body_n_m"].copy(),
            left_force_body_n=wrench_vectors["left_force_body_n"].copy(),
            right_force_body_n=wrench_vectors["right_force_body_n"].copy(),
            mechanical_power_w=mechanical_power_w,
        )
        return self._state()

    def _update_telemetry(
        self,
        rotation_body_to_world_before_step: np.ndarray,
        wing_velocity_before_step_rad_s: np.ndarray,
    ) -> None:
        simulation = self.bundle["simulation"]
        data = simulation.mj_data
        dof_adr = self.bundle["root_dof_adr"]
        force_world_n = (
            data.qfrc_fluid[dof_adr : dof_adr + 3] * _FLYGYM_FORCE_TO_NEWTON
        )
        force_body_n = np.asarray(rotation_body_to_world_before_step, dtype=float).T.dot(
            force_world_n
        )
        torque_body_n_m = (
            data.qfrc_fluid[dof_adr + 3 : dof_adr + 6]
            * _FLYGYM_TORQUE_TO_NEWTON_METRE
        )
        self._last_actuator_torque_n_m = (
            simulation.get_actuator_forces(
                self.bundle["fly"].name, self.bundle["actuator_type"]
            )
            * _FLYGYM_TORQUE_TO_NEWTON_METRE
        )
        all_velocities = simulation.get_joint_velocities(self.bundle["fly"].name)
        wing_velocity_after_step = all_velocities[self.bundle["wing_joint_indices"]]
        mean_wing_velocity = 0.5 * (
            np.asarray(wing_velocity_before_step_rad_s, dtype=float)
            + wing_velocity_after_step
        )
        self._last_wrench = AerodynamicWrench(
            force_body_n=np.asarray(force_body_n, dtype=float),
            torque_body_n_m=np.asarray(torque_body_n_m, dtype=float).copy(),
            left_force_body_n=np.zeros(3, dtype=float),
            right_force_body_n=np.zeros(3, dtype=float),
            mechanical_power_w=float(
                np.sum(
                    np.abs(self._last_actuator_torque_n_m * mean_wing_velocity)
                )
            ),
        )

    def step(
        self,
        wings: WingKinematics,
        force_body_n: np.ndarray,
        torque_body_n_m: np.ndarray,
        dt_s: float,
    ) -> RigidBodyState:
        del force_body_n, torque_body_n_m
        config = self.bundle["config"]
        if not np.isclose(dt_s, config.timestep_s, rtol=0.0, atol=1e-15):
            raise ValueError(
                "FlyBody adapter step must equal its compiled physics timestep"
            )
        torque_n_m = np.asarray(self.torque_mapper.map_torque(wings), dtype=float)
        if torque_n_m.shape != (6,) or not np.all(np.isfinite(torque_n_m)):
            raise ValueError("wing torque mapper must produce six finite values")
        if np.max(np.abs(torque_n_m), initial=0.0) > config.max_abs_wing_torque_n_m:
            raise ValueError("mapped wing torque exceeds configured actuator limit")
        simulation = self.bundle["simulation"]
        state_before_step = self._state()
        rotation_before_step = quaternion_to_matrix(
            state_before_step.quaternion_body_to_world
        )
        all_velocities_before_step = simulation.get_joint_velocities(
            self.bundle["fly"].name
        )
        wing_velocity_before_step = all_velocities_before_step[
            self.bundle["wing_joint_indices"]
        ].copy()
        simulation.set_actuator_inputs(
            self.bundle["fly"].name,
            self.bundle["actuator_type"],
            torque_n_m / _FLYGYM_TORQUE_TO_NEWTON_METRE,
        )
        simulation.step()
        self._update_telemetry(rotation_before_step, wing_velocity_before_step)
        return self._state()
