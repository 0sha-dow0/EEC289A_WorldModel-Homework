"""Student one-step + multi-start rollout loss with curriculum horizon."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout


# Increments once per training update. Eval code paths don't touch this.
_update_counter = 0


def _curriculum_horizon(target_horizon: int, ramp_updates: int, start_horizon: int = 10) -> int:
    """Linearly ramp from start_horizon to target_horizon over ramp_updates."""
    if ramp_updates <= 0:
        return int(target_horizon)
    progress = min(_update_counter / float(ramp_updates), 1.0)
    cur = int(round(start_horizon + (target_horizon - start_horizon) * progress))
    return max(start_horizon, min(cur, int(target_horizon)))


def one_step_delta_loss(model, states, actions, normalizer):
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def _one_start_rollout(model, states, actions, normalizer, warmup_steps, horizon, tail_weight, start):
    needed = warmup_steps + horizon + 1
    sub_states = states[:, start : start + needed]
    sub_actions = actions[:, start : start + warmup_steps + horizon]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer,
                              warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    w = torch.linspace(1.0, float(tail_weight), horizon, device=preds.device)
    w = (w / w.mean()).view(1, horizon, 1)
    return (((pred_norm - target_norm) ** 2) * w).mean()


def rollout_loss(model, states, actions, normalizer, warmup_steps, horizon,
                 tail_weight=2.0, n_starts=2):
    warmup_steps, horizon, n_starts = int(warmup_steps), int(horizon), max(1, int(n_starts))
    needed = warmup_steps + horizon + 1
    if states.shape[1] < needed:
        raise ValueError(
            f"train_sequence_length too short: need >= {needed - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed
    if max_start == 0:
        starts = [0] * n_starts
    elif max_start + 1 >= n_starts:
        starts = torch.randperm(max_start + 1, device=states.device)[:n_starts].tolist()
    else:
        starts = torch.randint(0, max_start + 1, (n_starts,), device=states.device).tolist()

    losses = [_one_start_rollout(model, states, actions, normalizer,
                                 warmup_steps, horizon, tail_weight, int(s))
              for s in starts]
    return torch.stack(losses).mean()


def compute_loss(model, batch, normalizer, cfg):
    global _update_counter
    _update_counter += 1

    lc = cfg["loss"]
    s, a = batch["states"], batch["actions"]

    target_horizon = int(lc.get("rollout_train_horizon", 50))
    ramp_updates = int(lc.get("curriculum_ramp_updates", 1500))
    start_horizon = int(lc.get("curriculum_start_horizon", 10))
    n_starts = int(lc.get("rollout_n_starts", 2))
    tail_w = float(lc.get("rollout_tail_weight", 2.0))
    warmup = int(cfg["eval"].get("warmup_steps", 10))

    horizon = _curriculum_horizon(target_horizon, ramp_updates, start_horizon)
    # Cap by what the sequence length actually supports
    max_possible = s.shape[1] - warmup - 1
    horizon = max(start_horizon, min(horizon, max_possible))

    one = one_step_delta_loss(model, s, a, normalizer)
    roll = rollout_loss(model, s, a, normalizer, warmup, horizon, tail_w, n_starts)

    total = (
        float(lc.get("one_step_weight", 1.5)) * one
        + float(lc.get("rollout_weight", 1.0)) * roll
    )
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/cur_horizon": float(horizon),
    }