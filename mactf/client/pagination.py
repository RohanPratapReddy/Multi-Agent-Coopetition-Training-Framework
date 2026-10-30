"""Exhaust a collection (document 03 section 3).

Kept apart from :class:`~mactf.client.base.Page` on purpose: iterating a ``Page`` yields what
it holds and never issues a request; walking every page is an explicit act.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

from .base import Page, Transport
from .errors import raise_for_response

T = TypeVar("T")


def fetch_json(transport: Transport, path: str, params: Mapping[str, Any] | None = None) -> Any:
    """GET ``path`` and decode it, raising the mapped client error for a non-2xx answer."""
    response = raise_for_response(transport.request("GET", path, params=params))
    return response.json()


def iter_pages(
    transport: Transport,
    path: str,
    *,
    limit: int = 50,
    offset: int = 0,
    parse: Callable[[Any], T] = lambda item: item,  # type: ignore[assignment,return-value]
    **filters: Any,
) -> Iterator[Page[T]]:
    """Yield successive pages of ``path`` until ``has_more`` is false."""
    while True:
        payload = fetch_json(transport, path, {**filters, "limit": limit, "offset": offset})
        page = Page.from_payload(payload, parse)
        yield page
        if not page.has_more or not page.items:
            return
        offset = page.offset + len(page.items)


def iter_items(transport: Transport, path: str, **kwargs: Any) -> Iterator[Any]:
    """Every item of every page, in order."""
    for page in iter_pages(transport, path, **kwargs):
        yield from page


__all__ = ["fetch_json", "iter_items", "iter_pages"]
