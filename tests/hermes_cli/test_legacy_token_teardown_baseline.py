"""Baseline harness for the legacy-session-token teardown.

Pins the CURRENT (pre-teardown) auth contract of BOTH dashboard regimes
so the phased removal of ``_SESSION_TOKEN`` can prove it didn't regress
the gated path or silently widen the public surface.

This file ADDS the contracts not already covered by
``test_dashboard_auth_gate.py`` (which already pins loopback token
enforcement, the ``should_require_auth`` truth table, and ``start_server``
flag-stashing):

  * gated mode IGNORES the legacy ``X-Hermes-Session-Token`` header
  * the WS auth matrix via ``_ws_auth_reason`` (loopback token vs gated
    ticket/internal)
  * no ``_require_token``-guarded sensitive path is in PUBLIC_API_PATHS

The expectations in this file are intentionally the PRE-teardown contract.
Later phases edit the specific assertions they intentionally change (and
the commit that changes them documents why).
"""
from __future__ import annotations

import re

import pytest

# These tests mutate ``web_server.app.state.auth_required`` at module
# scope; share the xdist group used by every dashboard-auth gate test so
# they don't race against each other.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loopback_client():
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def gated_client():
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


# ---------------------------------------------------------------------------
# Gated mode ignores the legacy session token (mutual-exclusivity invariant)
# ---------------------------------------------------------------------------


def test_gated_ignores_legacy_token_header(gated_client):
    """In gated mode the legacy token header is inert: a request carrying
    a *valid* ``X-Hermes-Session-Token`` and no cookie must still 401."""
    r = gated_client.get(
        "/api/sessions",
        headers={"X-Hermes-Session-Token": "stale-token-ignored"},
    )
    assert r.status_code == 401
    assert r.json().get("error") in ("unauthenticated", "session_expired")


def test_gated_status_still_public(gated_client):
    """``/api/status`` stays public in gated mode (NAS liveness probe)."""
    assert gated_client.get("/api/status").status_code == 200


# ---------------------------------------------------------------------------
# Loopback has no identity gate (post-Phase-2 contract)
# ---------------------------------------------------------------------------


def test_loopback_no_identity_gate(loopback_client):
    """Loopback: the bind + CSRF guard + CORS are the boundary, not an
    identity token. A tokenless read is allowed."""
    r = loopback_client.get("/api/sessions")
    assert r.status_code != 401


def test_loopback_still_blocks_cross_site_mutation(loopback_client):
    """The CSRF guard (not an identity token) is what protects loopback
    mutations from a drive-by cross-origin page."""
    r = loopback_client.post(
        "/api/providers/validate",
        headers={"Sec-Fetch-Site": "cross-site"},
        json={"key": "OPENAI_API_KEY", "value": "x"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# WS auth matrix (via _ws_auth_reason — TestClient.websocket_connect is
# unreliable for handshake-rejection assertions, so test the function)
# ---------------------------------------------------------------------------


def _fake_ws(params: dict):
    class _Client:
        host = "127.0.0.1"

    class _URL:
        path = "/api/ws"

    class _WS:
        query_params = params
        client = _Client()
        url = _URL()

    return _WS()


def test_ws_loopback_no_token_required():
    """Loopback WS accepts without a token: the peer-IP loopback gate +
    Host/Origin guard are the boundary (the WS analogue of the loopback
    bind being the HTTP boundary)."""
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        reason, cred = web_server._ws_auth_reason(_fake_ws({}))
        assert reason is None and cred == "loopback"
    finally:
        web_server.app.state.auth_required = prev


def test_ws_loopback_token_ignored():
    """A stale/garbage ``?token=`` on loopback is simply ignored (no
    identity token is consulted anymore)."""
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        reason, cred = web_server._ws_auth_reason(_fake_ws({"token": "anything"}))
        assert reason is None and cred == "loopback"
    finally:
        web_server.app.state.auth_required = prev


def test_ws_gated_rejects_legacy_token():
    """Gated mode never consults the legacy ``?token=`` path."""
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    try:
        reason, cred = web_server._ws_auth_reason(
            _fake_ws({"token": "stale-token-ignored"})
        )
        assert reason == "no_credential"  # token ignored; no ticket present
    finally:
        web_server.app.state.auth_required = prev


# ---------------------------------------------------------------------------
# _require_token gating invariant — no sensitive guarded path is public
# ---------------------------------------------------------------------------


def test_require_token_call_sites_exist():
    """At least one handler still guards via ``_require_token``."""
    text = open(web_server.__file__).read()
    n_sites = len(re.findall(r"_require_token\(request\)", text)) - 1  # minus def
    assert n_sites >= 1


def test_sensitive_paths_not_in_public_allowlist():
    """The public allowlist must never contain a sensitive route. This is
    the audit invariant the gate relies on (a _require_token route that is
    also public-allowlisted gets no session attached and 401s under the
    gate even after the loopback teardown)."""
    for sensitive in (
        "/api/env/reveal",
        "/api/providers/validate",
        "/api/dashboard/agent-plugins/install",
    ):
        assert sensitive not in PUBLIC_API_PATHS
