"""Functional container smoke check using only the standard-library HTTP client."""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


def request(path, payload=None):
    encoded = json.dumps(payload).encode() if payload is not None else None
    message = urllib.request.Request("http://127.0.0.1:8000" + path, data=encoded,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(message, timeout=10) as response:
        return json.loads(response.read())


def main():
    server = subprocess.Popen([sys.executable, "-m", "neuralis"])
    try:
        for attempt in range(80):
            if server.poll() is not None:
                raise RuntimeError("Container API exited before becoming ready")
            try:
                health = request("/health")
                break
            except (OSError, urllib.error.URLError):
                time.sleep(.25)
        else:
            raise RuntimeError("Container API startup exceeded twenty seconds")
        assert health["status"] == "ok"
        capabilities = request("/api/v1/extensions/capabilities")
        assert capabilities["axislab"]["available"] and capabilities["webrtc"]["available"]
        assert capabilities["trajectory"]["storage"] == "duckdb"
        axis = request("/api/v1/extensions/axis/evaluate", {
            "positive": [[2, 1], [2, -1]], "negative": [[0, 1], [0, -1]],
        })
        assert axis["auc"] == 1
        adaptation = request("/api/v1/extensions/training/demo", {"steps": 2, "rank": 2})
        assert len(adaptation["losses"]) == 2 and adaptation["max_base_change"] == 0
        index = request("/api/v1/extensions/rag/index", {
            "text": "Container validation verifies encrypted activation research documents.", "source": "container-test",
        })
        assert index["encrypted"]
        retrieval = request("/api/v1/extensions/rag/search", {"query": "encrypted activation"})
        assert retrieval["results"][0]["source"] == "container-test"
        trajectory = request("/api/v1/extensions/arena/demo", {"steps": 20})
        assert trajectory["storage_backend"] == "duckdb" and trajectory["terminated"]
        print(json.dumps({"health": "ok", "axislab_auc": axis["auc"], "lora_steps": 2,
                          "frozen_base_change": adaptation["max_base_change"],
                          "encrypted_retrieval": True, "duckdb_episode_steps": trajectory["steps"],
                          "webrtc_installed": capabilities["webrtc"]["available"]}))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
