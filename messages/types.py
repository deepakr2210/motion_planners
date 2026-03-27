"""
Message dataclasses for all ZMQ topics.

This file is the single source of truth for message structure.
It has no ZMQ, no encoding, no topic names — just data shapes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


# ── Control mode constants ───────────────────────────────────────────────────

MODE_TORQUE    = "torque"     # raw joint torques [Nm] → data.ctrl
MODE_POSITION  = "position"   # desired q [rad]   → internal sim PD servo
MODE_KINEMATIC = "kinematic"  # desired q [rad]   → direct qpos write, no physics


# ── Message types ────────────────────────────────────────────────────────────

@dataclass
class StateMsg:
    """
    Published by the sim on the STATE topic.

    q          joint positions      [rad]    length = ndof
    qd         joint velocities     [rad/s]  length = ndof
    qfrc_bias  gravity + Coriolis   [Nm]     length = ndof  (MuJoCo qfrc_bias)
    sim_time   simulation clock     [s]
    wall_time  publisher wall-clock [s]
    """
    q:         List[float]
    qd:        List[float]
    qfrc_bias: List[float]
    sim_time:  float
    wall_time: float = field(default_factory=time.time)


@dataclass
class Waypoint:
    """Single time-stamped point on a joint-space trajectory."""
    t:   float        # time from trajectory start [s]
    q:   List[float]  # joint positions   [rad]
    qd:  List[float]  # joint velocities  [rad/s]


@dataclass
class TrajectoryMsg:
    """
    Published by the planner on the TRAJ topic.

    waypoints   ordered list of (t, q, qd) waypoints
    start_time  wall-clock time corresponding to waypoint t=0
    mode        control mode the controller should use for this trajectory
                one of: MODE_TORQUE | MODE_POSITION | MODE_KINEMATIC
    wall_time   publisher wall-clock [s]
    """
    waypoints:  List[Waypoint]
    start_time: float
    mode:       str   = MODE_TORQUE
    wall_time:  float = field(default_factory=time.time)


@dataclass
class CommandMsg:
    """
    Sent by the controller to the sim on the CMD topic.

    values   interpretation depends on mode:
               torque    → joint torques [Nm]
               position  → desired joint positions [rad]
               kinematic → desired joint positions [rad]
    mode     one of: MODE_TORQUE | MODE_POSITION | MODE_KINEMATIC
    wall_time publisher wall-clock [s]
    """
    values:    List[float]
    mode:      str   = MODE_TORQUE
    wall_time: float = field(default_factory=time.time)
