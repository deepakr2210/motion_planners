"""
ZMQ encode / decode helpers.

Wire format (multipart)
-----------------------
  frame 0 : topic bytes   e.g. b"STATE"
  frame 1 : JSON payload  UTF-8

Message types and topic configuration live in their own modules:
  messages/types.py   — dataclasses (StateMsg, TrajectoryMsg, CommandMsg, Waypoint)
  messages/topics.py  — TopicSpec registry (STATE, TRAJ, CMD)

This file only handles serialisation.

Usage
-----
  from messages.protocol import encode, decode
  from messages.topics   import STATE, TRAJ, CMD

  # Publish
  pub.send_multipart(encode(STATE, state_msg))

  # Receive
  topic_bytes, raw = sub.recv_multipart()
  spec = TOPICS.by_bytes[topic_bytes]
  msg  = decode(spec, raw)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .types  import StateMsg, TrajectoryMsg, CommandMsg, Waypoint
from .topics import TopicSpec, TOPICS, STATE, TRAJ, CMD


# ── Encode ───────────────────────────────────────────────────────────────────

def encode(spec: TopicSpec, msg: Any) -> list[bytes]:
    """Serialise *msg* into a two-frame ZMQ multipart message for *spec*'s topic."""
    return [spec.bytes, json.dumps(asdict(msg)).encode()]


# ── Decode ───────────────────────────────────────────────────────────────────

def decode(spec: TopicSpec, raw: bytes) -> Any:
    """Deserialise *raw* JSON bytes into the message type registered for *spec*."""
    d = json.loads(raw)

    if spec.msg_type is StateMsg:
        return StateMsg(**d)

    if spec.msg_type is TrajectoryMsg:
        waypoints = [Waypoint(**w) for w in d.pop("waypoints")]
        return TrajectoryMsg(waypoints=waypoints, **d)

    if spec.msg_type is CommandMsg:
        return CommandMsg(**d)

    raise ValueError(f"No decoder registered for topic '{spec.name}'")


# ── Convenience one-liners (kept for backward compatibility) ─────────────────

def encode_state(msg: StateMsg)       -> list[bytes]: return encode(STATE, msg)
def encode_traj(msg: TrajectoryMsg)   -> list[bytes]: return encode(TRAJ,  msg)
def encode_cmd(msg: CommandMsg)       -> list[bytes]: return encode(CMD,   msg)

def decode_state(raw: bytes)          -> StateMsg:       return decode(STATE, raw)
def decode_traj(raw: bytes)           -> TrajectoryMsg:  return decode(TRAJ,  raw)
def decode_cmd(raw: bytes)            -> CommandMsg:     return decode(CMD,   raw)
