"""Peer registry for the training project.

Declaring ``PEERS`` on the app package is what ``AppConfig.import_models`` looks
for, which is how MACTF discovers peers without a manual registry call.
"""

from __future__ import annotations

from . import DeepPeer, MLPPeer, WidePeer

__all__ = ["MLPPeer", "DeepPeer", "WidePeer"]
