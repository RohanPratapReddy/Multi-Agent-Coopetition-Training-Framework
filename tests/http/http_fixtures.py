"""A tiny URL conf and application shared by the http tests."""

from __future__ import annotations

import time
import types

from mactf.http import (
    CharField,
    Conflict,
    IntField,
    JsonResponse,
    Response,
    Serializer,
    StreamingResponse,
    View,
    include,
    path,
    sse_event,
    sse_heartbeat,
)
from mactf.http.handler import WSGIHandler


class ItemIn(Serializer):
    name = CharField(max_length=20)
    count = IntField(min_value=0, default=1)


class ItemList(View):
    summary = "Items"
    tags = ["items"]
    request_serializer = ItemIn
    response_serializer = ItemIn

    def get(self, request):
        return JsonResponse({"items": [], "q": request.GET.get("q")})

    def post(self, request):
        ser = ItemIn.from_dict(request.json())
        ser.is_valid(raise_exception=True)
        return JsonResponse(ser.validated_data, status=201)


class ItemDetail(View):
    def get(self, request, item_id):
        if item_id == 0:
            raise Conflict("zero")
        return JsonResponse({"id": item_id})


def boom(request):
    raise RuntimeError("kaput")


def plain(request):
    return Response(b"hello", content_type="text/plain")


def stream(request):
    def gen():
        yield sse_event("tick", {"n": 1}, event_id=1)
        yield sse_heartbeat()
        yield sse_event("tick", {"n": 2}, event_id=2)

    return StreamingResponse(gen(), content_type="text/event-stream")


def slow_stream(request):
    marks = []

    def gen():
        for n in range(3):
            marks.append(time.monotonic())
            yield sse_event("tick", {"n": n})

    resp = StreamingResponse(gen(), content_type="text/event-stream")
    resp.marks = marks
    return resp


def make_urlconf() -> types.ModuleType:
    module = types.ModuleType("http_fixture_urls")
    sub = types.ModuleType("http_fixture_sub")
    sub.app_name = "items"
    sub.urlpatterns = [
        path("", ItemList.as_view(), name="item-list"),
        path("{item_id:int}/", ItemDetail.as_view(), name="item-detail"),
    ]
    module.urlpatterns = [
        path("api/items/", include((sub, "items"))),
        path("boom/", boom, name="boom"),
        path("plain/", plain, name="plain"),
        path("stream/", stream, name="stream"),
    ]
    return module


def make_app(settings=None) -> WSGIHandler:
    return WSGIHandler(urlconf=make_urlconf())
