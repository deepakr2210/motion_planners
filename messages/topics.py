"""
Topic registry — maps every ZMQ topic to its wire bytes, message type, and metadata.

This is the single place to add or rename a topic. Nothing else needs to change.

Usage
-----
  from messages.topics import TOPICS, STATE, TRAJ, CMD

  # Bind a SUB socket to the STATE topic
  sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)

  # Decode an incoming frame using the registry
  topic_bytes, raw = sock.recv_multipart()
  spec = TOPICS.by_bytes[topic_bytes]
  msg  = spec.decode(raw)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

from .types import StateMsg, TrajectoryMsg, CommandMsg, Waypoint, Twist


# ── TopicSpec ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TopicSpec:
    """Descriptor for a single ZMQ pub/sub topic."""
    name:        str         # human-readable name, also used as the JSON "type" tag
    bytes:       bytes       # ZMQ topic filter bytes sent on the wire
    msg_type:    Type        # message dataclass carried by this topic
    publisher:   str         # which process publishes this topic
    subscribers: tuple       # which processes subscribe
    description: str         # one-line description


# ── Topic definitions ────────────────────────────────────────────────────────

STATE = TopicSpec(
    name        = "STATE",
    bytes       = b"STATE",
    msg_type    = StateMsg,
    publisher   = "sim",
    subscribers = ("planner", "controller"),
    description = "Joint state (q, qd, qfrc_bias) published by the sim at state_hz",
)

TRAJ = TopicSpec(
    name        = "TRAJ",
    bytes       = b"TRAJ",
    msg_type    = TrajectoryMsg,
    publisher   = "planner",
    subscribers = ("controller",),
    description = "Planned joint-space trajectory (waypoints + control mode)",
)

CMD = TopicSpec(
    name        = "CMD",
    bytes       = b"CMD",
    msg_type    = CommandMsg,
    publisher   = "controller",
    subscribers = ("sim",),
    description = "Control command (torques | desired q | kinematic q) sent to the sim",
)

TWIST = TopicSpec(
    name        = "TWIST",
    bytes       = b"TWIST",
    msg_type    = Twist,
    publisher   = "twist_publisher",
    subscribers = ("diff_ik_control",),
    description = "End-effector velocity twist [vx, vy, vz, wx, wy, wz] in world frame",
)


# ── Registry ─────────────────────────────────────────────────────────────────

class _TopicRegistry:
    """Lookup table for topics by name or by wire bytes."""

    def __init__(self, *specs: TopicSpec) -> None:
        self._by_name:  Dict[str,   TopicSpec] = {s.name:  s for s in specs}
        self._by_bytes: Dict[bytes, TopicSpec] = {s.bytes: s for s in specs}

    # dict-style access by name
    def __getitem__(self, name: str) -> TopicSpec:
        return self._by_name[name]

    def __iter__(self):
        return iter(self._by_name.values())

    @property
    def by_bytes(self) -> Dict[bytes, TopicSpec]:
        """Look up a TopicSpec from the raw ZMQ topic frame."""
        return self._by_bytes

    @property
    def by_name(self) -> Dict[str, TopicSpec]:
        return self._by_name

    def __repr__(self) -> str:
        names = ", ".join(self._by_name)
        return f"TopicRegistry({names})"


TOPICS = _TopicRegistry(STATE, TRAJ, CMD, TWIST)
