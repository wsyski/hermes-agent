"""Tests for the dashboard-managed file browser API."""

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server

# These tests mutate web_server.app.state (auth_required / bound_host); share
# the dashboard-auth xdist group so they don't race other app.state mutators.
pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")


def _client_with_app_state():
    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    # Loopback bind has no identity gate; no session header needed.
    return client, prev_auth_required, prev_bound_host


def _restore_app_state(prev_auth_required, prev_bound_host):
    if prev_auth_required is None:
        delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth_required
    if prev_bound_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_bound_host


def _close_client(client):
    close = getattr(client, "close", None)
    if close is not None:
        close()


@pytest.fixture
def forced_files_client(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        yield client, root
    finally:
        _close_client(client)
        _restore_app_state(prev_auth_required, prev_bound_host)


@pytest.fixture
def local_files_client(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        yield client, home
    finally:
        _close_client(client)
        _restore_app_state(prev_auth_required, prev_bound_host)


def test_forced_root_file_upload_list_read_delete_roundtrip(forced_files_client):
    client, root = forced_files_client
    file_path = root / "out" / "hello.txt"

    created = client.post(
        "/api/files/upload",
        json={
            "path": str(file_path),
            "data_url": "data:text/plain;base64,aGVsbG8=",
        },
    )
    assert created.status_code == 200
    assert created.json()["entry"]["path"] == str(file_path)
    assert created.json()["locked_root"] == str(root)
    assert created.json()["can_change_path"] is False
    assert file_path.read_text() == "hello"

    listing = client.get("/api/files", params={"path": str(root / "out")})
    assert listing.status_code == 200
    assert listing.json()["path"] == str(root / "out")
    assert listing.json()["parent"] == str(root)
    assert listing.json()["entries"] == [
        {
            "name": "hello.txt",
            "path": str(file_path),
            "is_directory": False,
            "size": 5,
            "mtime": pytest.approx(file_path.stat().st_mtime),
            "mime_type": "text/plain",
        }
    ]

    read = client.get("/api/files/read", params={"path": str(file_path)})
    assert read.status_code == 200
    assert read.json()["data_url"] == "data:text/plain;base64,aGVsbG8="

    deleted = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(file_path)},
    )
    assert deleted.status_code == 200
    assert not file_path.exists()


def test_directory_management_requires_recursive_delete_for_nonempty_dirs(forced_files_client):
    client, root = forced_files_client
    runs_path = root / "runs"
    checkpoints_path = runs_path / "checkpoints"

    created = client.post("/api/files/mkdir", json={"path": str(checkpoints_path)})
    assert created.status_code == 200
    assert checkpoints_path.is_dir()

    listing = client.get("/api/files", params={"path": str(runs_path)})
    assert listing.status_code == 200
    assert listing.json()["entries"][0]["path"] == str(checkpoints_path)
    assert listing.json()["entries"][0]["is_directory"] is True

    non_recursive = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(runs_path), "recursive": False},
    )
    assert non_recursive.status_code == 409

    recursive = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(runs_path), "recursive": True},
    )
    assert recursive.status_code == 200
    assert not runs_path.exists()


def test_forced_root_paths_stay_under_root(forced_files_client, tmp_path):
    client, root = forced_files_client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not leak")

    traversal = client.get("/api/files", params={"path": "../outside"})
    assert traversal.status_code == 400

    outside_absolute = client.get("/api/files", params={"path": str(outside)})
    assert outside_absolute.status_code == 403

    root_delete = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(root), "recursive": True},
    )
    assert root_delete.status_code == 400

    root.mkdir(exist_ok=True)
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not allow directory symlinks")

    escaped = client.get("/api/files", params={"path": str(link)})
    assert escaped.status_code == 403


def test_local_mode_defaults_to_home_and_can_jump_to_absolute_path(local_files_client, tmp_path):
    client, home = local_files_client
    (home / "home.txt").write_text("home")

    default_listing = client.get("/api/files")
    assert default_listing.status_code == 200
    assert default_listing.json()["path"] == str(home)
    assert default_listing.json()["locked_root"] is None
    assert default_listing.json()["can_change_path"] is True
    assert default_listing.json()["entries"][0]["path"] == str(home / "home.txt")

    other = tmp_path / "other"
    other.mkdir()
    (other / "other.txt").write_text("other")

    other_listing = client.get("/api/files", params={"path": str(other)})
    assert other_listing.status_code == 200
    assert other_listing.json()["path"] == str(other)
    assert other_listing.json()["parent"] == str(tmp_path)
    assert other_listing.json()["entries"][0]["path"] == str(other / "other.txt")


def test_gated_local_mode_still_defaults_to_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))

    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = "0.0.0.0"
    try:
        request = SimpleNamespace(
            app=web_server.app,
            client=SimpleNamespace(host="10.0.0.2"),
            url=SimpleNamespace(hostname="example.com"),
        )
        policy = web_server._managed_files_policy(request, create_root=False)
    finally:
        _restore_app_state(prev_auth_required, prev_bound_host)

    assert policy.default_path == home.resolve()
    assert policy.locked_root is None
    assert policy.can_change_path is True


def test_local_mode_upload_read_mkdir_delete_roundtrip(local_files_client):
    client, home = local_files_client
    folder = home / "workspace"
    file_path = folder / "note.txt"

    created_folder = client.post("/api/files/mkdir", json={"path": str(folder)})
    assert created_folder.status_code == 200
    assert created_folder.json()["locked_root"] is None
    assert created_folder.json()["can_change_path"] is True
    assert folder.is_dir()

    uploaded = client.post(
        "/api/files/upload",
        json={
            "path": str(file_path),
            "data_url": "data:text/plain;base64,bG9jYWw=",
        },
    )
    assert uploaded.status_code == 200
    assert file_path.read_text() == "local"

    read = client.get("/api/files/read", params={"path": str(file_path)})
    assert read.status_code == 200
    assert read.json()["data_url"] == "data:text/plain;base64,bG9jYWw="

    deleted = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(folder), "recursive": True},
    )
    assert deleted.status_code == 200
    assert not folder.exists()


def _seed_file(client, root, name="out/hello.txt"):
    file_path = root / name
    created = client.post(
        "/api/files/upload",
        json={"path": str(file_path), "data_url": "data:text/plain;base64,aGVsbG8="},
    )
    assert created.status_code == 200
    return file_path


def test_download_returns_file_as_attachment(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root)

    resp = client.get("/api/files/download", params={"path": str(file_path)})
    assert resp.status_code == 200
    assert resp.content == b"hello"
    disposition = resp.headers["content-disposition"]
    assert "attachment" in disposition
    assert "hello.txt" in disposition


def test_download_no_identity_gate_on_loopback(forced_files_client):
    """Loopback download needs no credential after the legacy-token teardown.

    The browser/shell-opened download (which can't set a session header) just
    works on a loopback bind — the bind is the security boundary. The old
    ``?token=`` query-param escape hatch is gone with the token. Gated-mode
    enforcement is pinned by test_download_requires_auth_in_gated_mode below.
    """
    client, root = forced_files_client
    file_path = _seed_file(client, root)

    ok = client.get("/api/files/download", params={"path": str(file_path)})
    assert ok.status_code == 200
    assert ok.content == b"hello"

    # A stale/garbage ?token= is simply ignored, not rejected, on loopback.
    still_ok = client.get(
        "/api/files/download", params={"path": str(file_path), "token": "anything"}
    )
    assert still_ok.status_code == 200


def test_download_requires_auth_in_gated_mode(forced_files_client, monkeypatch):
    """In gated (non-loopback) mode the download endpoint requires a verified
    session cookie — a cookieless request 401s at the gate, and there is no
    ``?token=`` query-param bypass."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    client, root = forced_files_client
    file_path = _seed_file(client, root)

    prev_host = getattr(web_server.app.state, "bound_host", None)
    clear_providers()
    register_provider(StubAuthProvider())
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.auth_required = True
    try:
        gated = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        # Cookieless → 401 at the gate, with or without a bogus ?token=.
        assert gated.get(
            "/api/files/download", params={"path": str(file_path)}
        ).status_code == 401
        assert gated.get(
            "/api/files/download", params={"path": str(file_path), "token": "anything"}
        ).status_code == 401
    finally:
        clear_providers()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.auth_required = False


def test_hosted_policy_locks_to_opt_data(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.setenv("HERMES_HOME", "/opt/data")
    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        request = SimpleNamespace(
            app=web_server.app,
            client=SimpleNamespace(host="127.0.0.1"),
            url=SimpleNamespace(hostname="127.0.0.1"),
        )
        policy = web_server._managed_files_policy(request, create_root=False)
    finally:
        _restore_app_state(prev_auth_required, prev_bound_host)
        client.close()

    assert str(policy.locked_root) == "/opt/data"
    assert policy.can_change_path is False
