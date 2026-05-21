"""Student one-step + rollout loss with horizon-tail weighting."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout


def one_step_delta_loss(model, states, actions, normalizer):
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(model, states, actions, normalizer, warmup_steps, horizon, tail_weight=2.0):
    warmup_steps, horizon = int(warmup_steps), int(horizon)
    needed = warmup_steps + horizon + 1
    if states.shape[1] < needed:
        raise ValueError(
            f"train_sequence_length too short: need >= {needed - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0
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


def compute_loss(model, batch, normalizer, cfg):
    lc = cfg["loss"]
    s, a = batch["states"], batch["actions"]
    one = one_step_delta_loss(model, s, a, normalizer)
    horizon = int(lc.get("rollout_train_horizon", 50))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    tail_w = float(lc.get("rollout_tail_weight", 2.0))
    roll = rollout_loss(model, s, a, normalizer, warmup, horizon, tail_w)
    total = float(lc.get("one_step_weight", 0.5)) * one + float(lc.get("rollout_weight", 1.0)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }