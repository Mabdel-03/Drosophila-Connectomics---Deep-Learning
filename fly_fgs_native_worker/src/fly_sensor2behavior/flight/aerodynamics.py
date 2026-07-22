"""Fast quasi-steady aerodynamic approximation for prescribed wing motion.

This is suitable for deterministic regression and interactive preview, not CFD
or fluid-structure validation.  It follows the standard quasi-steady structure
used in insect-flight models: force scales with dynamic pressure, wing area,
and angle-dependent lift/drag coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import AerodynamicWrench, WingKinematics


@dataclass(frozen=True)
class QuasiSteadyParameters:
    air_density_kg_m3: float = 1.225
    wing_area_m2: float = 1.05e-6
    representative_radius_m: float = 1.18e-3
    # The reduced model places the cycle-averaged pressure center over the body
    # COM. A calibrated body/hinge model may supply a non-zero fore-aft offset.
    center_of_pressure_x_m: float = 0.0
    center_of_pressure_y_m: float = 1.05e-3
    # Chosen so the declared 1 mg baseline approximately balances weight. This
    # is a regression calibration target, not a held-out biological validation.
    lift_coefficient_scale: float = 1.65
    drag_coefficient_offset: float = 0.12
    drag_coefficient_scale: float = 1.15
    body_linear_drag_n_per_m_s: float = 1.8e-7
    body_angular_drag_n_m_per_rad_s: float = 1.0e-14


class QuasiSteadyAerodynamics:
    validation_status = "exploratory"

    def __init__(
        self, parameters: QuasiSteadyParameters = QuasiSteadyParameters()
    ) -> None:
        self.parameters = parameters

    def evaluate(
        self,
        wings: WingKinematics,
        air_velocity_body_m_s: np.ndarray,
        angular_velocity_body_rad_s: np.ndarray,
    ) -> AerodynamicWrench:
        p = self.parameters
        air_velocity_body_m_s = np.asarray(air_velocity_body_m_s, dtype=float)
        angular_velocity_body_rad_s = np.asarray(
            angular_velocity_body_rad_s, dtype=float
        )
        wing_speed = p.representative_radius_m * np.abs(
            wings.stroke_velocity_rad_s
        )
        alpha = np.clip(wings.angle_of_attack_rad, 0.0, np.pi / 2.0)
        cl = p.lift_coefficient_scale * np.sin(2.0 * alpha)
        cd = p.drag_coefficient_offset + p.drag_coefficient_scale * (
            1.0 - np.cos(2.0 * alpha)
        )
        dynamic = 0.5 * p.air_density_kg_m3 * p.wing_area_m2 * wing_speed ** 2
        lift = dynamic * cl
        drag = dynamic * cd

        forces = np.zeros((2, 3), dtype=float)
        for side_index, mirror in enumerate((1.0, -1.0)):
            deviation = wings.deviation_rad[side_index]
            # z is dorsal/up; deviation tilts lift fore/aft. Lateral profile
            # drag reverses each half stroke and is mirrored by side.
            forces[side_index, 0] = lift[side_index] * np.sin(deviation)
            forces[side_index, 1] = (
                -mirror
                * np.sign(wings.stroke_velocity_rad_s[side_index])
                * drag[side_index]
            )
            forces[side_index, 2] = lift[side_index] * np.cos(deviation)

        positions = np.array(
            [
                [p.center_of_pressure_x_m, p.center_of_pressure_y_m, 0.0],
                [p.center_of_pressure_x_m, -p.center_of_pressure_y_m, 0.0],
            ],
            dtype=float,
        )
        torques = np.cross(positions, forces)
        total_force = forces.sum(axis=0)
        total_force -= p.body_linear_drag_n_per_m_s * air_velocity_body_m_s
        total_torque = torques.sum(axis=0)
        total_torque -= p.body_angular_drag_n_m_per_rad_s * angular_velocity_body_rad_s
        power = float(
            np.sum(np.abs(wings.generalized_torque_n_m * wings.stroke_velocity_rad_s))
        )
        return AerodynamicWrench(
            force_body_n=total_force,
            torque_body_n_m=total_torque,
            left_force_body_n=forces[0].copy(),
            right_force_body_n=forces[1].copy(),
            mechanical_power_w=power,
        )
