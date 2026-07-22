"""Dependency-light six-degree-of-freedom rigid-body flight integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .types import AerodynamicWrench, RigidBodyState, WingKinematics


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quaternion_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    wx, wy, wz = omega_body
    return 0.5 * np.array(
        [
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class RigidBodyParameters:
    mass_kg: float = 1.0e-6
    inertia_body_kg_m2: tuple = (1.0e-12, 1.0e-12, 1.8e-12)
    gravity_m_s2: float = 9.80665


class ExternalFlightPhysicsAdapter(Protocol):
    """Protocol for a FlyBody/MuJoCo worker adapter.

    The dependency-light host does not import FlyBody.  A Python 3.12 worker
    implements this boundary and translates ``WingKinematics`` into the exact
    pinned model's generalized actuation while retaining physics ownership.
    """

    backend_name: str
    aerodynamic_owner: str

    def default_initial_state(self) -> RigidBodyState:
        ...

    def reset(self, initial_state: RigidBodyState) -> None:
        ...

    def step(
        self,
        wings: WingKinematics,
        force_body_n: np.ndarray,
        torque_body_n_m: np.ndarray,
        dt_s: float,
    ) -> RigidBodyState:
        ...

    def aerodynamic_wrench(self) -> AerodynamicWrench:
        ...


class RigidBody6DOF:
    def __init__(
        self,
        parameters: RigidBodyParameters = RigidBodyParameters(),
        initial_state: Optional[RigidBodyState] = None,
    ) -> None:
        self.parameters = parameters
        self.inertia = np.diag(np.asarray(parameters.inertia_body_kg_m2, dtype=float))
        self.inverse_inertia = np.diag(1.0 / np.diag(self.inertia))
        self.state = initial_state.copy() if initial_state is not None else RigidBodyState()

    def reset(self, initial_state: RigidBodyState) -> None:
        self.state = initial_state.copy()

    def step(
        self,
        force_body_n: np.ndarray,
        torque_body_n_m: np.ndarray,
        dt_s: float,
        external_force_world_n: Optional[np.ndarray] = None,
    ) -> RigidBodyState:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        state = self.state
        rotation = quaternion_to_matrix(state.quaternion_body_to_world)
        external = (
            np.zeros(3, dtype=float)
            if external_force_world_n is None
            else np.asarray(external_force_world_n, dtype=float)
        )
        gravity = np.array([0.0, 0.0, -self.parameters.mass_kg * self.parameters.gravity_m_s2])
        force_world = rotation.dot(np.asarray(force_body_n, dtype=float)) + gravity + external
        acceleration = force_world / self.parameters.mass_kg
        # Constant-acceleration update over this aerodynamic substep. This is
        # second-order for the held wrench and avoids a timestep-dependent hover
        # bias when the cycle-averaged lift nearly balances gravity.
        state.position_world_m += (
            state.velocity_world_m_s * dt_s
            + 0.5 * acceleration * dt_s * dt_s
        )
        state.velocity_world_m_s += acceleration * dt_s

        omega = state.angular_velocity_body_rad_s
        angular_momentum = self.inertia.dot(omega)
        angular_acceleration = self.inverse_inertia.dot(
            np.asarray(torque_body_n_m, dtype=float) - np.cross(omega, angular_momentum)
        )
        omega_mid = omega + 0.5 * angular_acceleration * dt_s
        state.angular_velocity_body_rad_s += angular_acceleration * dt_s
        state.quaternion_body_to_world += _quaternion_derivative(
            state.quaternion_body_to_world, omega_mid
        ) * dt_s
        state.quaternion_body_to_world /= np.linalg.norm(
            state.quaternion_body_to_world
        )
        return state.copy()
