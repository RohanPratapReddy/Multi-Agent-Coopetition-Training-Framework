"""The MACTF client: one API, two transports (in-process and HTTP).

Importing this package pulls in only ``mactf.http`` and the standard library; the framework
classes it returns (``PeerScorecard``, ``AggregationReport``, ...) load on first use.
"""

from __future__ import annotations

from . import specs
from .base import Checkpoint, Episode, Event, Page, Report, Run, RuleEvent, RunHandle, Transport
from .client import MACTFClient, PointsEntry
from .errors import (
    CapabilityUnavailable,
    Conflict,
    Forbidden,
    MACTFClientError,
    NotFound,
    RateLimited,
    ServerError,
    TransportError,
    Unauthorized,
    ValidationError,
)
from .transport import HTTPTransport, InProcessTransport

__all__ = [
    "CapabilityUnavailable", "Checkpoint", "Conflict", "Episode", "Event", "Forbidden",
    "HTTPTransport", "InProcessTransport", "MACTFClient", "MACTFClientError", "NotFound",
    "Page", "PointsEntry", "RateLimited", "Report", "Run", "RunHandle", "RuleEvent",
    "ServerError", "Transport", "TransportError", "Unauthorized", "ValidationError", "specs",
]
