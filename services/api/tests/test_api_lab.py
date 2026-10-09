import io
import zipfile

from fastapi.testclient import TestClient
import pytest

from neuralis.lab.auth import Principal
from neuralis.main import create_app
from neuralis.storage.repository import Repository


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("NEURALIS_TOKEN", raising=False)
    monkeypatch.delenv("NEURALIS_OIDC_ISSUER", raising=False)
    monkeypatch.setenv("NEURALIS_HOST", "127.0.0.1")
    with TestClient(create_app(tmp_path)) as value:
        yield value


def test_end_to_end_run_intervention_export_and_ws(client):
    assert client.get("/health").json()["status"] == "ok"
    row = client.post("/api/v1/runs", json={"steps": 20}).json()
    run_id = row["id"]
    assert client.post(f"/api/v1/runs/{run_id}/commands", json={"command": "step"}).json()["step"] == 1
    tensors = client.get(f"/api/v1/runs/{run_id}/tensors").json()
    assert tensors["tensors"]
    with client.websocket_connect(f"/api/v1/ws/runs/{run_id}") as websocket:
        assert websocket.receive_json()["type"] == "snapshot"
        assert websocket.receive_json()["run_id"] == run_id
    inference = client.post(f"/api/v1/runs/{run_id}/inference", json={}).json()
    assert inference["output"]["shape"] == [4, 2]
    experiment = client.post("/api/v1/experiments", json={"run_id": run_id,
        "interventions": [{"module": "output", "operation": "scale", "strength": 0.5}]}).json()
    assert experiment["result"]["applied"][0]["change_norm"] > 0
    response = client.get(f"/api/v1/runs/{run_id}/export")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "events.json" in archive.namelist()
    assert client.post(f"/api/v1/runs/{run_id}/commands", json={"command": "stop"}).status_code == 200


def test_bearer_auth_and_origin_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALIS_TOKEN", "test-only-credential")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/capabilities").status_code == 401
        assert client.get("/api/v1/capabilities", headers={"Authorization": "Bearer bad"}).status_code == 401
        assert client.get("/api/v1/capabilities", headers={"Authorization": "Bearer test-only-credential"}).status_code == 200


def test_cross_project_rbac_and_permissions(client):
    project_a = client.post("/api/v1/projects", json={"name": "A"}).json()["id"]
    project_b = client.post("/api/v1/projects", json={"name": "B"}).json()["id"]
    client.post(f"/api/v1/projects/{project_a}/members", json={"subject": "alice", "role": "viewer"})
    auth = client.app.state.auth
    repository = client.app.state.repository
    auth.authorize(repository, Principal("alice"), project_a, "read")
    with pytest.raises(Exception) as error:
        auth.authorize(repository, Principal("alice"), project_a, "train")
    assert error.value.status_code == 403
    with pytest.raises(Exception) as error:
        auth.authorize(repository, Principal("alice"), project_b, "read")
    assert error.value.status_code == 403
    assert any(record["outcome"] == "denied" for record in repository.audit_records())


def test_worker_lease_heartbeat_cancel_and_completion(client):
    registered = client.post("/api/v1/workers/register", json={"worker_id": "worker-1",
        "capabilities": {"tasks": ["xor"], "cpu_count": 2, "memory_mb": 4096, "max_parallel": 1}})
    assert registered.status_code == 200
    job = client.post("/api/v1/queue", json={"task": "xor", "steps": 5}).json()
    lease = client.post("/api/v1/workers/worker-1/lease", json={"tasks": ["xor"], "resources": {"cpu_threads": 2, "memory_mb": 4096}}).json()
    assert lease["job_id"] == job["id"]
    assert client.post("/api/v1/workers/worker-1/lease", json={}).json() is None
    heartbeat = {"job_id": job["id"], "lease_id": lease["lease_id"], "metrics": {"loss": 0.5}}
    assert client.post("/api/v1/workers/worker-1/heartbeat", json=heartbeat).json()["cancel"] is False
    assert client.post(f"/api/v1/queue/{job['id']}/cancel").json()["status"] == "cancel-requested"
    assert client.post("/api/v1/workers/worker-1/heartbeat", json=heartbeat).json()["cancel"] is True
    completion = {"lease_id": lease["lease_id"], "state": "cancelled", "result": {"steps": 2}}
    assert client.post(f"/api/v1/workers/worker-1/jobs/{job['id']}/complete", json=completion).status_code == 200
    assert client.post(f"/api/v1/workers/worker-1/jobs/{job['id']}/complete", json=completion).json()["idempotent"]


def test_expired_lease_is_not_requeued(client):
    client.post("/api/v1/workers/register", json={"worker_id": "worker-1",
        "capabilities": {"tasks": ["xor"], "memory_mb": 4096}})
    job = client.post("/api/v1/queue", json={}).json()
    lease = client.post("/api/v1/workers/worker-1/lease", json={}).json()
    with client.app.state.repository.transaction() as connection:
        connection.execute("UPDATE queue_jobs SET lease_expiry=0 WHERE id=?", (job["id"],))
    response = client.post("/api/v1/workers/worker-1/heartbeat", json={"job_id": job["id"], "lease_id": lease["lease_id"]})
    assert response.status_code == 409
    assert client.get("/api/v1/queue").json()[0]["status"] == "lost"
    assert client.post("/api/v1/workers/worker-1/lease", json={}).json() is None


def test_bounded_event_history_reports_gap(tmp_path):
    repository = Repository(tmp_path, max_events_per_run=3)
    row = repository.create_run({"project_id": "local"})
    for step in range(8):
        repository.append_event(row["id"], step, "metrics", {"loss": float(step)})
    page = repository.events(row["id"])
    assert len(page["events"]) == 3
    assert page["gap"]
    assert page["oldest_sequence"] == 6
