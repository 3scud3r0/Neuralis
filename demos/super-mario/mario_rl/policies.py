"""Load versioned PPO or Double-DQN policies and compare explicit baselines."""
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical

from .checkpoint import load_policy
from .dqn import load_dqn_policy


def load_any_policy(path: Path, device_name: str = "cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("Checkpoint must contain Neuralis policy metadata")
    if data.get("format") == "neuralis.mario.ppo/v1":
        return load_policy(path, device_name)
    if data.get("format") == "neuralis.mario.double-dqn/v1":
        return load_dqn_policy(path, device_name)
    raise ValueError("Unsupported policy checkpoint format")


class BaselinePolicy(nn.Module):
    """A named control policy, explicitly distinct from trained neural agents."""
    def __init__(self, actions: int, mode: str = "random", fixed_action: int = 1):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode not in {"random", "fixed"} or not 0 <= fixed_action < actions:
            raise ValueError("Unsupported baseline mode/action")
        self.actions = actions
        self.mode = mode
        self.fixed_action = fixed_action

    def act(self, observations, actions=None, deterministic=False):
        batch = observations.shape[0]
        logits = torch.zeros((batch, self.actions), device=observations.device)
        if self.mode == "fixed":
            action = torch.full((batch,), self.fixed_action, dtype=torch.long, device=observations.device)
            logits.fill_(-20)
            logits[:, self.fixed_action] = 0
        else:
            # Random baseline remains random even during deterministic-agent tests.
            action = torch.randint(self.actions, (batch,), device=observations.device)
        distribution = Categorical(logits=logits)
        return action, distribution.log_prob(action), distribution.entropy(), torch.zeros(batch, device=observations.device)

    def inspect(self, observations):
        with torch.no_grad():
            actions = self.act(observations)[0].cpu().tolist()
        return {"layers": [], "baseline": self.mode, "actions": actions,
                "note": "Control baseline; no learned neural parameters."}
