"""Server-sent events for a run (document 02 section 3.4)."""

from __future__ import annotations

from collections.abc import Iterator

from ....http.exceptions import Http404, ValidationError
from ....http.request import Request
from ....http.response import StreamingResponse, sse_event
from ....http.response import sse_headers
from ....orchestration.live import RunNotFound, Subscription
from .. import deps
from .base import ApiView, get_run_or_404


def _stream(subscription: Subscription) -> Iterator[bytes]:
    try:
        for event in subscription:
            yield sse_event(event["event"], event["data"], event_id=event.get("id"))
    finally:
        # Runs on normal completion and when the server drops the connection.
        subscription.cancel()


class RunEventsView(ApiView):
    summary = "Live run events (SSE)"
    description = (
        "text/event-stream. Events: run, phase, epoch, warning, incident, checkpoint, rollback, "
        "terminal, heartbeat. Reconnect with Last-Event-ID; an id older than the buffer yields "
        "event: stream_lost. The stream closes after terminal."
    )

    def get(self, request: Request, run_id: str) -> StreamingResponse:
        get_run_or_404(request, run_id)
        raw = request.headers.get("Last-Event-ID") or request.GET.get("last_event_id")
        last_id = None
        if raw not in (None, ""):
            try:
                last_id = int(raw)
            except ValueError:
                raise ValidationError({"Last-Event-ID": ["must be an integer."]}) from None
        try:
            subscription = deps.get_run_manager_dep(request).subscribe(run_id, last_id)
        except RunNotFound:
            raise Http404(f"no run {run_id!r}") from None
        return StreamingResponse(_stream(subscription), headers=sse_headers())
