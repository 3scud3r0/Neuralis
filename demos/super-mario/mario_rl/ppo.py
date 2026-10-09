"""PPO's clipped policy objective, clipped value loss, and timeout-aware GAE."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .config import Config
from .model import ActorCritic


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    ended: torch.Tensor

    @classmethod
    def allocate(cls, steps: int, environments: int, observation_shape: tuple, device: torch.device):
        shape = (steps, environments)
        return cls(
            torch.empty((*shape, *observation_shape), dtype=torch.uint8, device=device),
            torch.empty(shape, dtype=torch.long, device=device),
            torch.empty(shape, device=device),
            torch.empty(shape, device=device),
            torch.empty(shape, device=device),
            torch.empty(shape, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
        )

    def advantages(self, gamma: float, gae_lambda: float):
        return generalized_advantages(self.rewards, self.values, self.next_values,
                                      self.terminated, self.ended, gamma, gae_lambda)


def generalized_advantages(rewards, values, next_values, terminated, ended, gamma, gae_lambda):
    """Terminate bootstrap on death; bootstrap timeout, but stop its GAE trace.

    next_values must describe the terminal observation on automatic reset, never
    the initial frame of the next episode. This distinction prevents timeout
    value bias and cross-episode leakage.
    """
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape == ended.shape):
        raise ValueError("All GAE arrays must have the same [time, environment] shape")
    advantages = torch.zeros_like(rewards)
    trace = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]).float() - values[step]
        trace = delta + gamma * gae_lambda * (~ended[step]).float() * trace
        advantages[step] = trace
    return advantages, advantages + values


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    variance = targets.var(unbiased=False)
    if variance.item() <= 1e-12:
        return 0.0
    return float((1 - (targets - predictions).var(unbiased=False) / variance).item())


def optimize(model: ActorCritic, optimizer: torch.optim.Optimizer, rollout: Rollout, config: Config) -> dict:
    advantages, returns = rollout.advantages(config.gamma, config.gae_lambda)
    observations = rollout.observations.flatten(0, 1)
    actions = rollout.actions.flatten()
    old_log_probs = rollout.log_probs.flatten()
    old_values = rollout.values.flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()
    total = len(actions)
    if total > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sums = {name: 0.0 for name in ("loss", "policyLoss", "valueLoss", "entropy", "approxKL", "clipFraction", "gradientNorm")}
    minibatches = 0
    early_stop = False
    for epoch in range(config.update_epochs):
        indices = torch.randperm(total, device=actions.device)
        epoch_kl = []
        for start in range(0, total, config.minibatch_size):
            batch = indices[start:start + config.minibatch_size]
            _, log_probs, entropy, values = model.act(observations[batch], actions[batch])
            log_ratio = log_probs - old_log_probs[batch]
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > config.clip_coef).float().mean()
            unclipped_policy = -advantages[batch] * ratio
            clipped_policy = -advantages[batch] * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)
            policy_loss = torch.maximum(unclipped_policy, clipped_policy).mean()
            unclipped_value = (values - returns[batch]).pow(2)
            clipped_prediction = old_values[batch] + (values - old_values[batch]).clamp(-config.clip_coef, config.clip_coef)
            clipped_value = (clipped_prediction - returns[batch]).pow(2)
            value_loss = 0.5 * torch.maximum(unclipped_value, clipped_value).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss became non-finite; inspect rewards and learning rate")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm became non-finite")
            optimizer.step()
            measurements = {
                "loss": loss, "policyLoss": policy_loss, "valueLoss": value_loss,
                "entropy": entropy_mean, "approxKL": approx_kl,
                "clipFraction": clip_fraction, "gradientNorm": gradient_norm,
            }
            for name, value in measurements.items():
                sums[name] += float(value.detach().item())
            minibatches += 1
            epoch_kl.append(float(approx_kl.item()))
        if config.target_kl and sum(epoch_kl) / len(epoch_kl) > config.target_kl:
            early_stop = True
            break
    result = {name: value / minibatches for name, value in sums.items()}
    result.update({"explainedVariance": explained_variance(old_values, returns),
                   "epochs": epoch + 1, "minibatches": minibatches, "earlyStop": early_stop})
    return result
