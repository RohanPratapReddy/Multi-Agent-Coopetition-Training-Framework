"""Worker routes. Mounted by the *project's* URL conf (or by ``ROOT_URLCONF = None`` when the
app is installed); the api app never includes them, so removing this app removes the routes."""

from __future__ import annotations

from ..http.urls import path
from . import views

app_name = "worker"

urlpatterns = [
    path("register", views.register, name="register"),
    path("heartbeat", views.heartbeat, name="heartbeat"),
    path("tasks", views.tasks, name="tasks"),
    path("results", views.results, name="results"),
    path("exchange", views.exchange, name="exchange"),
    path("deregister", views.deregister, name="deregister"),
]
