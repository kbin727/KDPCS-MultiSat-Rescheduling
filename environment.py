# -*- coding: utf-8 -*-
"""Minimal data environment for KDPCS benchmark and teacher records.

This module is intentionally limited to data loading and schedule replay.
It provides the class/function names required to unpickle released instances
and to reconstruct oracle trajectories from Gurobi teacher schedules. It does
not contain the KDPCS policy network, training code, or checkpoint logic.
"""

MEMORY_CAPACITY = 200.0
ENERGY_CAPACITY = 50000.0
OBS_POWER = 50.0
TRANS_POWER = 30.0
ATTITUDE_RATE = 2.0


class MultiSatSchedulingInstance:
    """Container class used by released pickle records."""

    pass


def compute_attitude_transition(roll1, pitch1, roll2, pitch2, rate=ATTITUDE_RATE):
    delta_roll = abs(float(roll2) - float(roll1))
    delta_pitch = abs(float(pitch2) - float(pitch1))
    return max(delta_roll, delta_pitch) / max(float(rate), 1e-9)


def get_sat_memory_capacity(instance, satellite_id):
    sat = instance.satellites[satellite_id] if hasattr(instance, "satellites") else {}
    return float(sat.get("memory_capacity", getattr(instance, "memory_capacity", MEMORY_CAPACITY)))


def get_sat_energy_capacity(instance, satellite_id):
    sat = instance.satellites[satellite_id] if hasattr(instance, "satellites") else {}
    return float(sat.get("energy_capacity", getattr(instance, "energy_capacity", ENERGY_CAPACITY)))


def get_sat_obs_power(instance, satellite_id):
    sat = instance.satellites[satellite_id] if hasattr(instance, "satellites") else {}
    return float(sat.get("obs_power", sat.get("sensor_power", OBS_POWER)))


def get_sat_trans_power(instance, satellite_id):
    sat = instance.satellites[satellite_id] if hasattr(instance, "satellites") else {}
    return float(sat.get("trans_power", sat.get("reaction_wheel_power", TRANS_POWER)))


def get_sat_attitude_rate(instance, satellite_id):
    sat = instance.satellites[satellite_id] if hasattr(instance, "satellites") else {}
    return float(sat.get("attitude_rate", ATTITUDE_RATE))


def compute_sat_attitude_transition(instance, satellite_id, roll1, pitch1, roll2, pitch2):
    return compute_attitude_transition(
        roll1,
        pitch1,
        roll2,
        pitch2,
        rate=get_sat_attitude_rate(instance, satellite_id),
    )


def compute_task_energy(instance, satellite_id, transition_time, duration):
    return (
        float(transition_time) * get_sat_trans_power(instance, satellite_id)
        + float(duration) * get_sat_obs_power(instance, satellite_id)
    )

