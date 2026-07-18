"""Runtime boundaries shared by protected inference and evaluation."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CI_ENVIRONMENT_MARKERS = (
    "BUILDKITE",
    "CIRCLECI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TF_BUILD",
)
_OFFLINE_ENVIRONMENT = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
}


def reject_ci_environment(operation: str) -> None:
    """Reject hosted or generic CI before protected content is opened."""
    active = [name for name in _CI_ENVIRONMENT_MARKERS if _environment_truthy(name)]
    if _environment_truthy("CI"):
        active.append("CI")
    if active:
        raise RuntimeError(
            f"{operation} refuses CI environments; active markers=" + ",".join(active)
        )


@contextmanager
def offline_inference_environment() -> Iterator[None]:
    """Set library offline flags and block new Python socket connections during inference."""
    prior_environment = {key: os.environ.get(key) for key in _OFFLINE_ENVIRONMENT}
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class OfflineSocket(original_socket):  # type: ignore[misc, valid-type]
        def connect(self, address: Any) -> None:
            raise RuntimeError("Offline inference blocked an outbound socket connection")

        def connect_ex(self, address: Any) -> int:
            raise RuntimeError("Offline inference blocked an outbound socket connection")

    def blocked_create_connection(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Offline inference blocked an outbound socket connection")

    try:
        os.environ.update(_OFFLINE_ENVIRONMENT)
        setattr(socket, "socket", OfflineSocket)
        setattr(socket, "create_connection", blocked_create_connection)
        yield
    finally:
        setattr(socket, "socket", original_socket)
        setattr(socket, "create_connection", original_create_connection)
        for key, prior in prior_environment.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _environment_truthy(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().casefold() not in {"", "0", "false", "no"}
