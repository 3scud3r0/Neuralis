# Laboratory deployment and workers

The runnable baseline uses a FastAPI CPU runtime, SQLite metadata and local safe-tensor/JSON artifacts. The laboratory coordinator adds project quotas, authenticated worker registration, queue acquisition, lease renewal and completion. Optional OIDC JWT validation uses issuer, audience and an HTTPS JWKS endpoint. These are operational building blocks; they do not establish a production-certified frontier laboratory system.

## Containers

The CPU API and standalone worker have separate Dockerfiles. The API installs the frozen `services/api/uv.lock`, including optional laboratory authentication dependencies. The worker has no third-party runtime dependencies for its registered XOR task. Neither image installs models, ROMs or a GPU toolkit.

```bash
cp deploy/.env.example deploy/.env
# Generate a random token and write it into deploy/.env without committing the file.
docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d api
docker compose --env-file deploy/.env -f deploy/compose.yaml --profile worker up --build -d
```

The API is published only on host loopback. The worker shares the API's container network namespace, making `http://127.0.0.1:8000` a local connection. Named volumes persist backend data and the worker's completion ledger. Both processes run as UID 10001, with dropped Linux capabilities, a read-only root filesystem, bounded temporary storage and process/memory/CPU limits. The basic profile reserves two CPUs and 4 GiB for the API and two CPUs and 768 MiB for the worker; adjust for the physical machine.

A remote deployment needs an HTTPS reverse proxy, an identity provider or securely issued service token, explicit CORS origins, backups and limits suited to the intended workload. Do not expose the basic API port to the public network using plaintext transport.

In a managed cloud environment, use its existing proxy configuration and trust bundle. The overlay mounts a combined public CA bundle into networked build steps and running containers:

```bash
docker compose --env-file deploy/.env \
  -f deploy/compose.yaml -f deploy/compose.cloud.yaml build
```

The `system_ca` BuildKit secret mounts the CA only while dependencies are installed. It never copies the session certificate into an image layer. Configure `NEURALIS_SYSTEM_CA` if the combined bundle has another path. Registry credentials and existing Docker proxy settings remain platform-managed.

PostgreSQL and S3 are not part of the baseline persistence path. Installing optional driver libraries does not switch storage adapters. Do not create database/object-store services or set invented connection variables: use the capabilities endpoint and the backend documentation to determine whether a configured adapter is supported. OIDC configuration examples in `deploy/oidc.env.example` are examples rather than an included identity-provider service.

## Worker installation and self-test

```bash
python -m pip install -e services/worker
python -m neuralis_worker --doctor
python -m neuralis_worker --self-test
```

The self-test trains a real 2–8–1 tanh/sigmoid network on XOR using full-batch binary cross-entropy gradients. It reports achieved accuracy and loss. It does not fabricate GPU statistics. The worker token is read from `NEURALIS_WORKER_TOKEN`, falling back to `NEURALIS_TOKEN`; CLI arguments do not accept a token, preventing command-line exposure.

```bash
export NEURALIS_COORDINATOR=https://your-lab.example.org
# Supply NEURALIS_WORKER_TOKEN through a protected environment binding.
python -m neuralis_worker --worker-id cpu-01 --cpu-threads 2 \
  --memory-mb 2048 --timeout 300 --work-dir /your/data/worker
```

HTTPS is mandatory for remote coordinators. Loopback HTTP is permitted for development. The worker preserves certificate and hostname verification and refuses redirects to prevent bearer-token forwarding. Endpoint credentials, query strings and fragments are refused. Authentication failures stop the worker; transient coordinator failures use bounded retry intervals. Registration and completion retries are idempotent by worker/lease identifiers; acquisition is never blindly retried after an ambiguous request.

## Queue and lease protocol

Requests use the API's existing bearer authentication and `/api/v1` prefix:

| Operation | Endpoint | Meaning |
|---|---|---|
| Register | `POST /workers/register` | Worker identity and actual supported capabilities |
| Acquire | `POST /workers/{id}/lease` | Registered tasks and available CPU/memory budget |
| Renew | `POST /workers/{id}/heartbeat` | Lease ID, job ID, measured metrics and resource counters |
| Complete | `POST /workers/{id}/jobs/{job}/complete` | State, result, error and matching lease ID |
| Enqueue | `POST /queue` | Project, registered task and bounded task configuration |

The coordinator selects eligible queued jobs transactionally. The worker executes one job at a time. It accepts only locally registered task handlers: a queue message cannot choose a command, executable, module path or arbitrary script. Task options are strictly typed and bounded. Resource requests larger than the worker's declared budget fail before process startup.

A lease specifies its ID, job ID, expiration and duration. Renewals include real training metrics. A remote cancellation terminates the job process group. If authentication fails or a lease cannot be renewed within its duration, the worker stops execution. Expired jobs are marked lost by the coordinator and require an explicit new attempt; they are never silently replayed.

The worker's SQLite WAL ledger records each lease before execution. Completion enters a durable outbox before network delivery. Temporary delivery failures retain results and prevent acquisition of new work until delivery recovers. Restarted interrupted jobs become failed completions; execution is not repeated. A duplicate lease is refused, and a reused lease ID with different job/config data is an error. Completed history is pruned to a bounded count while undelivered results are preserved. The ledger has a 128 MiB storage quota and a 1 MiB result delivery budget.

Subprocesses do not inherit coordinator tokens. They run without shell invocation. POSIX workers enforce core-dump, CPU, file-size and descriptor limits. Pure-Python tasks also get a hard address-space limit. On Linux, aggregate RSS and CPU counters cover the job's descendant processes. DDP uses aggregate RSS monitoring instead of an address-space cap because PyTorch reserves large virtual mappings. Wall-clock deadlines apply on all supported platforms. Windows pure-Python execution is supported; DDP requires POSIX and is explicitly opt-in. On non-Linux systems, enforce aggregate process memory through your platform container/job limits.

## CPU distributed training

Install a compatible PyTorch CPU build from its verified official distribution, then opt into DDP:

```bash
python -m pip install -e 'services/worker[distributed]'
python -m neuralis_worker --allow-distributed --world-size 2 --cpu-threads 2 --doctor
```

The `ddp-xor` handler uses `torch.distributed.run --standalone`, a Gloo process group, a seeded tanh network and `DistributedDataParallel`. Ranks train evenly sized shards of a repeated XOR dataset. Gradients synchronize through DDP. Captured loss and accuracy synchronize through `all_reduce`; only rank zero emits the final JSON checkpoint. Each rank receives a bounded thread allocation. It is an actual single-machine CPU DDP demonstration, with no claim of tested multinode or GPU training.

## Tests and diagnostics

```bash
PYTHONPATH=services/worker python -m unittest discover -s services/worker/tests -v
```

Tests cover analytic training convergence, reproducibility, authenticated HTTP leases, cancellation, execution timeouts, rejected unregistered tasks, quota validation and durable completion recovery. Diagnostics report only capability names and readiness, never credential values. Production operators should independently validate identity-provider behavior, backup restore, host isolation and workload-specific limits before accepting sensitive workloads.
