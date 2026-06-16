import base64
from pathlib import Path

import pytest

from hermes_cli import web_server

# These tests mutate ``web_server.app.state`` (auth_required / bound_host);
# share the xdist group used by every dashboard-auth gate test so they
# don't race against each other across workers.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    # Loopback bind has no identity gate; no session header needed.
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def test_fs_list_sorts_and_hides_noise(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "b.txt").write_text("b")
    (root / "a_dir").mkdir()
    (root / "a.txt").write_text("a")
    (root / "node_modules").mkdir()
    (root / ".git").mkdir()

    response = client.get("/api/fs/list", params={"path": str(root)})

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [entry["name"] for entry in entries] == ["a_dir", "a.txt", "b.txt"]
    assert entries[0] == {"name": "a_dir", "path": str(root / "a_dir"), "isDirectory": True}
    assert all(entry["name"] not in {".git", "node_modules"} for entry in entries)


def test_fs_list_accepts_relative_paths(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel").mkdir()
    (tmp_path / "rel" / "file.txt").write_text("ok")

    response = client.get("/api/fs/list", params={"path": "rel"})

    assert response.status_code == 200
    assert response.json()["entries"] == [
        {"name": "file.txt", "path": str(tmp_path / "rel" / "file.txt"), "isDirectory": False}
    ]


def test_fs_list_missing_path_returns_structured_error(client, tmp_path):
    response = client.get("/api/fs/list", params={"path": str(tmp_path / "missing")})

    assert response.status_code == 200
    assert response.json() == {"entries": [], "error": "ENOENT"}


def test_fs_read_text_matches_preview_shape_and_truncates(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_TEXT_SOURCE_MAX_BYTES", 32)
    monkeypatch.setattr(web_server, "_FS_TEXT_PREVIEW_MAX_BYTES", 5)
    target = tmp_path / "sample.py"
    target.write_text("print('hello')")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json() == {
        "binary": False,
        "byteSize": 14,
        "language": "python",
        "mimeType": "text/x-python",
        "path": str(target),
        "text": "print",
        "truncated": True,
    }


def test_fs_read_text_rejects_source_over_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_TEXT_SOURCE_MAX_BYTES", 4)
    target = tmp_path / "large.txt"
    target.write_text("12345")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 413


def test_fs_read_text_flags_binary(client, tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello\x00world")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 200
    body = response.json()
    assert body["binary"] is True
    assert body["text"].startswith("hello")


def test_fs_read_data_url_returns_capped_data_url(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 16)
    target = tmp_path / "image.png"
    target.write_bytes(b"pngbytes")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json() == {"dataUrl": "data:image/png;base64," + base64.b64encode(b"pngbytes").decode("ascii")}


def test_fs_read_data_url_rejects_over_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 3)
    target = tmp_path / "image.png"
    target.write_bytes(b"1234")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 413


def test_fs_git_root_for_nested_file(client, tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "pkg" / "mod"
    nested.mkdir(parents=True)
    target = nested / "file.py"
    target.write_text("x")

    response = client.get("/api/fs/git-root", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json() == {"root": str(tmp_path)}


def test_fs_git_root_returns_null_outside_repo(client, tmp_path):
    response = client.get("/api/fs/git-root", params={"path": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {"root": None}


def test_fs_default_cwd_prefers_existing_terminal_cwd(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "load_config", lambda: {"terminal": {"cwd": str(tmp_path)}})
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "env"))
    monkeypatch.setattr(web_server.Path, "cwd", lambda: tmp_path / "process")
    monkeypatch.setattr(web_server, "_fs_git_branch", lambda cwd: "main")

    response = client.get("/api/fs/default-cwd")

    assert response.status_code == 200
    assert response.json() == {"cwd": str(tmp_path), "branch": "main"}


def test_fs_default_cwd_falls_back_when_terminal_cwd_is_invalid(client, tmp_path, monkeypatch):
    fallback = tmp_path / "backend"
    fallback.mkdir()
    monkeypatch.setattr(web_server, "load_config", lambda: {"terminal": {"cwd": "/client/missing"}})
    monkeypatch.setenv("TERMINAL_CWD", "/client/missing")
    monkeypatch.setattr(web_server.Path, "cwd", lambda: fallback)
    monkeypatch.setattr(web_server, "_fs_git_branch", lambda cwd: "")

    response = client.get("/api/fs/default-cwd")

    assert response.status_code == 200
    assert response.json() == {"cwd": str(fallback), "branch": ""}


def test_fs_endpoints_no_identity_gate_on_loopback(tmp_path):
    """Loopback has no identity gate after the legacy-token teardown.

    The /api/fs/* endpoints read arbitrary files, but on a loopback bind
    the OS boundary + CSRF guard are the protection, not a per-request
    token. A tokenless local request is served (reaches the handler — any
    non-401). Identity enforcement for these endpoints in GATED mode is
    pinned by test_fs_endpoints_require_auth_in_gated_mode below.
    """
    prev = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    try:
        client = TestClient(web_server.app)
        list_response = client.get("/api/fs/list", params={"path": str(tmp_path)})
        read_response = client.get("/api/fs/read-text", params={"path": str(target)})
        default_response = client.get("/api/fs/default-cwd")
        assert list_response.status_code != 401
        assert read_response.status_code != 401
        assert default_response.status_code != 401
    finally:
        web_server.app.state.auth_required = prev


def test_fs_endpoints_require_auth_in_gated_mode(tmp_path):
    """In gated (non-loopback) mode the /api/fs/* endpoints require a
    verified session cookie — a cookieless request 401s at the gate."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_req = getattr(web_server.app.state, "auth_required", None)
    clear_providers()
    register_provider(StubAuthProvider())
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.auth_required = True
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    try:
        client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        assert client.get(
            "/api/fs/list", params={"path": str(tmp_path)}
        ).status_code == 401
        assert client.get(
            "/api/fs/read-text", params={"path": str(target)}
        ).status_code == 401
    finally:
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.auth_required = prev_req
