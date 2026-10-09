from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import signal
import sys

from .agent import Worker
from .config import WorkerConfig
from .executor import capabilities, JobProcess
from .tasks import TaskConfig


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run an authenticated, bounded Neuralis CPU worker")
    result.add_argument("--endpoint", help="HTTPS coordinator, or HTTP on loopback")
    result.add_argument("--worker-id")
    result.add_argument("--cpu-threads", type=int)
    result.add_argument("--memory-mb", type=int)
    result.add_argument("--timeout", dest="timeout_seconds", type=int)
    result.add_argument("--work-dir", type=Path)
    result.add_argument("--allow-distributed", action="store_true")
    result.add_argument("--world-size", type=int, default=2)
    result.add_argument("--once", action="store_true", help="Acquire at most one lease and exit")
    result.add_argument("--doctor", action="store_true", help="Show non-secret capabilities")
    result.add_argument("--self-test-ddp", action="store_true", help="Run actual two-rank CPU DDP XOR training without a coordinator")
    result.add_argument("--self-test", action="store_true", help="Run actual local XOR training without a coordinator")
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    values = vars(args).copy()
    doctor, self_test, once = values.pop("doctor"), values.pop("self_test"), values.pop("once")
    distributed_test = values.pop("self_test_ddp")
    if distributed_test:
        values["allow_distributed"] = True
        self_test = True
    try:
        if doctor or self_test:
            # Diagnostics and self-test need no external authentication.
            values["token"] = "local-diagnostics-only-token"
        config = WorkerConfig.from_environment(**values)
        if doctor:
            print(json.dumps(capabilities(config), indent=2))
            return 0
        if self_test:
            import time
            job = JobProcess(TaskConfig(task="ddp-xor" if distributed_test else "xor",
                steps=1500 if distributed_test else 2500, capture_every=100), config)
            try:
                while job.poll() is None:
                    time.sleep(.05)
                state, result, error = job.finish()
                print(json.dumps({"state": state, "accuracy": result.get("accuracy"), "loss": result.get("loss"), "resources": result.get("resources"), "world_size": result.get("world_size", 1),
                    "rank_metrics": result.get("metrics", [])[-1].get("ranks") if result.get("metrics") else None, "error": error}, indent=2))
                return 0 if state == "completed" and result.get("accuracy") == 1 else 1
            finally:
                job.close()
        worker = Worker(config)
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda *_: worker.stop())
        worker.run(once=once)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
