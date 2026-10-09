# Stream a Mario checkpoint into RL Arena

`arena.py` executes the existing PPO or Dueling Double-DQN checkpoint in the real
NES emulator and records each action through Neuralis's trajectory API. It uses
the isolated Mario Python 3.11 environment; the API runs in its own environment.
It installs no additional dependencies and does not train or modify a checkpoint.

Start the Neuralis API using the repository's instructions, then run:

```bash
cd /workspace/Neuralis/demos/super-mario
/workspace/.venvs/mario/bin/python arena.py runs/smoke/latest.pt \
  --url http://127.0.0.1:8000 --max-steps 300 --fps 15
/workspace/.venvs/mario/bin/python arena.py runs/dqn-smoke/latest-dqn.pt \
  --url http://127.0.0.1:8000 --max-steps 300 --fps 15
```

Substitute your own checkpoint paths. `--env-id` chooses an explicit stage;
otherwise the checkpoint's saved stage is used. CPU is the default.
`--stochastic` samples PPO actions from its actual categorical distribution;
DDQN evaluation uses greedy Q-value selection. The default episode ID is unique.
`--episode-id` can assign a known new ID; `--episodes` generates numbered IDs for
multiple rollouts. Existing steps must not be overwritten.

For an authenticated API, securely provide `NEURALIS_TOKEN` in the process
environment. The recorder uses `Authorization: Bearer` and never writes the token
to metadata, files, or its output. Frame ingestion requires the API's
`instance-admin` authority. Token authentication uses HTTPS for remote origins;
loopback HTTP is supported for a local API. Redirects are refused so credentials
are not forwarded to another origin.

## What gets recorded

Each successful action posts to
`POST /api/v1/extensions/arena/episodes/{episode_id}/frames` with:

| Field | Measured content |
| --- | --- |
| `step` | Zero-based action index, monotonically increasing within a new episode |
| `frame_png` | Base64 PNG of the real post-action RGB emulator frame |
| `policy` | PPO action probabilities, or DDQN raw Q values (including negative values) |
| `action` | The actual selected action index |
| `reward` | Raw NES rewards summed across the configured frame skip |
| `terminated` | True environment termination, separate from truncation |
| `metadata` | Algorithm, policy kind, pre-action diagnostics, seed, stage, scaled training reward, truncation, emulator/action counts, position, flag status, and checkpoint SHA256 |

Policy outputs describe the **pre-action** stacked observation; the image shows
the **post-action** state produced by that action. PPO metadata includes logits,
entropy, and critic value; DDQN metadata identifies Q scores and greedy selection.
These are measured neural policy diagnostics, not an internal reasoning transcript.

The Mario adapter does not expose a verified complete emulator serialization
interface. The recorder sets `snapshot_supported: false` and omits `snapshot`.
Frames and policy outputs support trajectory inspection; they do not claim that
an emulator episode can be restored from this record alone.

## Bounds and failure behavior

Defaults are one episode, at most 300 actions, and at most 15 uploads per second.
CLI bounds permit 1–100 episodes, 1–10,000 actions per episode, at most 60 uploads
per second, and an HTTP timeout of at most 60 seconds. Rendering uses the NES's
native RGB frame, bounded to 640 by 480 pixels. PNG encoding uses NumPy plus the
standard library, with verified RGB8/noninterlaced output.

Every frame waits for a JSON acknowledgment confirming `stored: true` and the
requested step. Successful episode summaries include the acknowledged storage
backend. A failed upload stops the rollout,
closes the emulator, and exits unsuccessfully; earlier acknowledged steps remain
in the trajectory store. A transport timeout has unconfirmed delivery, so the
recorder does not retry a possibly committed step. Inspect the episode before
starting a fresh recording with a new ID. Stop flags and the step limit distinguish
true termination from truncation. A short smoke checkpoint is not evidence of
perfect gameplay.

The recorder sends ordinary frame-ingestion requests. The API can fan the same
incoming frames out to its optional live WebRTC queue; this CLI does not claim a
WebRTC connection of its own. Static GitHub Pages cannot execute the emulator or
host this API. Use a running optional Neuralis backend for live recording.

## Verified smoke run

On 2026-10-09 the isolated Mario runtime executed both existing smoke checkpoints
against a local HTTP contract recorder. The PPO checkpoint had 2,048 training
transitions; the DDQN checkpoint had 512. Each actual emulator rollout uploaded
three ordered frames (`step` 0, 1, 2), reached position 44, accumulated raw reward
3.0, and ended by the configured action-limit truncation. Neither reached the flag.

The receiver decoded every RGB PNG at the NES's native 256 by 240 resolution,
checked action bounds and finite policy/reward values, confirmed PPO probabilities
sum to one and DDQN actions select the greatest Q value, and verified that token
headers never entered payloads. Additional checks verified a random RGB image
round trip, invalid image bounds/types, unsafe credential URL rejection, and
unsuccessful handling of an HTTP 409 response. These checks establish the recorder
contract and real emulator execution; they do not establish a deployed API's
availability or the smoke policies' gameplay quality.

The CLI was then exercised against the actual Neuralis API on
`http://127.0.0.1:8001`. Both `mario-cli-ppo-confirmed-20261009` and
`mario-cli-ddqn-confirmed-20261009` completed three actions. All acknowledgments
confirmed storage, and the episode summaries reported `storage_backend: duckdb`.
Read-back checks through the episode, step, and PNG-frame endpoints verified
both ordered three-step trajectories, raw return 3.0 per episode, all six native
RGB frames, PPO probability normalization, greedy DDQN action selection, final
truncation, and no snapshots. The capability endpoint also reported DuckDB; the
runtime database resides outside the source checkout under
`/workspace/neuralis-data/extensions-validation/extensions/arena.duckdb`.
No runtime frame/database outputs were added to the repository.

The browser capture environment blocks non-proxied WebRTC UDP through a managed
Chromium policy and gathers no ICE candidates. Neuralis detects and reports this
condition before negotiating. Local aiortc peers decoded real video frames in
the streaming tests; archived NES frame inspection also succeeded in Chromium.
Screenshot 026 documents the browser transport restriction. It is not evidence
of decoded browser video.
