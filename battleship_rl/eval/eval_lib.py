""""""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from battleship_rl.eval.schema import DefenderShiftMetrics, DistributionStats, PolicyDiagnostics

def bootstrap_ci(arr: np.ndarray, stat_fn: Callable[[np.ndarray], float], n_bootstrap: int=300, alpha: float=0.05, rng: Optional[np.random.Generator]=None) -> tuple[float, float]:
    """"""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(arr)
    samples = np.array([stat_fn(arr[rng.integers(0, n, size=n)]) for _ in range(n_bootstrap)])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return (lo, hi)

def cvar(arr: np.ndarray, alpha: float=0.1) -> float:
    """"""
    threshold = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(tail.mean())

def _distribution_stats(lengths: list[int], trunc_count: int, total_eps: int, n_bootstrap: int, trunc_reason: Optional[dict]=None) -> DistributionStats:
    """"""
    arr = np.array(lengths, dtype=np.float64)
    rng = np.random.default_rng(7)
    ci_mean = bootstrap_ci(arr, np.mean, n_bootstrap=n_bootstrap, rng=rng)
    ci_p95 = bootstrap_ci(arr, lambda x: np.percentile(x, 95), n_bootstrap=n_bootstrap, rng=rng)
    return DistributionStats(mean=round(float(arr.mean()), 4), std=round(float(arr.std()), 4), p95=round(float(np.percentile(arr, 95)), 4), cvar_10=round(cvar(arr, alpha=0.1), 4), fail_rate=round(trunc_count / max(total_eps, 1), 6), n_episodes=len(lengths), ci_mean_lo=round(ci_mean[0], 4), ci_mean_hi=round(ci_mean[1], 4), ci_p95_lo=round(ci_p95[0], 4), ci_p95_hi=round(ci_p95[1], 4), trunc_reason=trunc_reason if trunc_reason is not None else {'time_limit': 0, 'invalid_action': 0, 'other': 0})

def _masked_entropy_from_logits(logits: np.ndarray, mask: np.ndarray) -> float:
    """"""
    lg = logits.copy()
    lg[~mask] = -1000000000.0
    lg -= lg.max()
    probs = np.exp(lg)
    probs /= probs.sum()
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def _adjacency_flag(action: int, hit_grid: np.ndarray, board_h: int, board_w: int) -> bool:
    """"""
    r, c = divmod(action, board_w)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = (r + dr, c + dc)
        if 0 <= nr < board_h and 0 <= nc < board_w:
            if hit_grid[nr, nc]:
                return True
    return False

def _normalize_outcome(info: dict) -> dict:
    """"""
    ot = info.get('outcome_type')
    if ot is not None:
        ot = ot.upper()
        return {'is_hit': ot in ('HIT', 'SUNK'), 'is_sink': ot == 'SUNK', 'is_invalid': ot == 'INVALID'}
    ot_legacy = info.get('outcome')
    if ot_legacy is not None:
        ot_legacy = str(ot_legacy).upper()
        return {'is_hit': ot_legacy in ('HIT', 'SUNK'), 'is_sink': ot_legacy == 'SUNK', 'is_invalid': ot_legacy == 'INVALID'}
    return {'is_hit': False, 'is_sink': False, 'is_invalid': False}

def _make_env_fn(defender_cls, seed: int):
    """"""

    def _init():
        from sb3_contrib.common.wrappers import ActionMasker
        from battleship_rl.envs.battleship_env import BattleshipEnv
        env = BattleshipEnv(defender=defender_cls(), debug=False)
        env = ActionMasker(env, lambda e: e.get_action_mask())
        env.reset(seed=seed)
        return env
    return _init

def run_eval(model, mode_map: dict, n_episodes: int, n_envs: int, seed_offset: int=8000, collect_diagnostics: bool=True, n_bootstrap: int=300, board_h: int=10, board_w: int=10) -> tuple[dict[str, DistributionStats], PolicyDiagnostics]:
    """"""
    stats_dict: dict[str, DistributionStats] = {}
    all_entropies: list[float] = []
    all_adj_flags: list[float] = []
    all_first_hit_steps: list[float] = []
    all_first_sink_steps: list[float] = []
    all_invalid_actions: list[float] = []
    n_batches = max(1, math.ceil(n_episodes / n_envs))
    for mode_name, defender_cls in mode_map.items():
        env_fns = [_make_env_fn(defender_cls, seed=seed_offset + i) for i in range(n_envs)]
        venv = VecMonitor(SubprocVecEnv(env_fns))
        ep_lengths: list[int] = []
        trunc_count: int = 0
        total_eps: int = 0
        trunc_reason: dict = {'time_limit': 0, 'invalid_action': 0, 'other': 0}
        for _batch in range(n_batches):
            obs = venv.reset()
            dones = [False] * n_envs
            steps = [0] * n_envs
            first_hit = [None] * n_envs
            first_sink = [None] * n_envs
            hit_grids = [np.zeros((board_h, board_w), dtype=bool) for _ in range(n_envs)]
            while not all(dones):
                masks = get_action_masks(venv)
                if collect_diagnostics:
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, device=model.device)
                        dist = model.policy.get_distribution(obs_t)
                        logits_np = dist.distribution.logits.cpu().numpy()
                actions, _ = model.predict(obs, deterministic=True, action_masks=masks)
                obs, rewards, new_dones, infos = venv.step(actions)
                for i, (done, ndone) in enumerate(zip(dones, new_dones)):
                    if done:
                        continue
                    a = int(actions[i])
                    steps[i] += 1
                    if collect_diagnostics:
                        h = _masked_entropy_from_logits(logits_np[i], masks[i])
                        all_entropies.append(h)
                        adj = _adjacency_flag(a, hit_grids[i], board_h, board_w)
                        all_adj_flags.append(float(adj))
                        outcome = _normalize_outcome(infos[i])
                        invalid = outcome['is_invalid']
                        if not outcome['is_hit'] and (not outcome['is_sink']) and (not outcome['is_invalid']) and (infos[i].get('outcome_type') is None) and (infos[i].get('outcome') is None):
                            r_, c_ = divmod(a, board_w)
                            invalid = bool(hit_grids[i][r_, c_])
                        all_invalid_actions.append(float(invalid))
                        if outcome['is_hit']:
                            r_, c_ = divmod(a, board_w)
                            hit_grids[i][r_, c_] = True
                            if first_hit[i] is None:
                                first_hit[i] = steps[i]
                        if outcome['is_sink'] and first_sink[i] is None:
                            first_sink[i] = steps[i]
                    if ndone:
                        ep_lengths.append(steps[i])
                        is_timelimit = bool(infos[i].get('TimeLimit.truncated', False))
                        is_invalid = bool(infos[i].get('outcome_type') == 'INVALID' or infos[i].get('outcome', '').upper() == 'INVALID')
                        is_other_trunc = not is_timelimit and (not is_invalid) and bool(infos[i].get('truncated', False))
                        if is_timelimit:
                            trunc_count += 1
                            trunc_reason['time_limit'] += 1
                        elif is_invalid:
                            trunc_reason['invalid_action'] += 1
                        elif is_other_trunc:
                            trunc_count += 1
                            trunc_reason['other'] += 1
                        total_eps += 1
                        dones[i] = True
                        if collect_diagnostics:
                            all_first_hit_steps.append(float(first_hit[i]) if first_hit[i] is not None else float(steps[i]))
                            all_first_sink_steps.append(float(first_sink[i]) if first_sink[i] is not None else float(steps[i]))
                            hit_grids[i][:] = False
                            first_hit[i] = None
                            first_sink[i] = None
        venv.close()
        stats_dict[mode_name] = _distribution_stats(ep_lengths, trunc_count, total_eps, n_bootstrap=n_bootstrap, trunc_reason=trunc_reason)
    diagnostics = PolicyDiagnostics(action_entropy=round(float(np.mean(all_entropies)) if all_entropies else 0.0, 4), adjacency_ratio=round(float(np.mean(all_adj_flags)) if all_adj_flags else 0.0, 4), time_to_first_hit=round(float(np.mean(all_first_hit_steps)) if all_first_hit_steps else 0.0, 4), time_to_first_sink=round(float(np.mean(all_first_sink_steps)) if all_first_sink_steps else 0.0, 4), invalid_action_rate=round(float(np.mean(all_invalid_actions)) if all_invalid_actions else 0.0, 6))
    return (stats_dict, diagnostics)

def defender_shift_metrics(dk_layouts: np.ndarray, uniform_layouts: np.ndarray, ship_ids: Optional[list[int]]=None) -> DefenderShiftMetrics:
    """"""

    def _within_layout_metrics(layouts: np.ndarray):
        pairwise_dists = []
        cluster_scores = []
        H, W = (layouts.shape[1], layouts.shape[2])
        for layout in layouts:
            unique_ids = np.unique(layout[layout >= 0])
            if len(unique_ids) < 2:
                pairwise_dists.append(0.0)
            else:
                centroids = []
                for sid in unique_ids:
                    cells = np.argwhere(layout == sid)
                    centroids.append(cells.mean(axis=0))
                centroids = np.array(centroids)
                dists = []
                for a in range(len(centroids)):
                    for b in range(a + 1, len(centroids)):
                        d = np.linalg.norm(centroids[a] - centroids[b])
                        dists.append(d)
                pairwise_dists.extend(dists)
            occ = (layout >= 0).astype(int)
            adj_pairs = 0
            adj_pairs += int((occ[:H - 1, :] & occ[1:, :]).sum())
            adj_pairs += int((occ[:, :W - 1] & occ[:, 1:]).sum())
            cluster_scores.append(adj_pairs)
        return (np.array(pairwise_dists), np.array(cluster_scores))
    dk_dists, dk_clusters = _within_layout_metrics(dk_layouts)
    occ_marginal = (dk_layouts >= 0).astype(float)
    p = occ_marginal.mean(axis=0).ravel()
    eps = 1e-08
    marg_entropy = float(-np.sum(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps)) / len(p))
    return DefenderShiftMetrics(centroid_pairwise_mean=round(float(dk_dists.mean()) if len(dk_dists) > 0 else 0.0, 4), centroid_pairwise_p95=round(float(np.percentile(dk_dists, 95)) if len(dk_dists) > 0 else 0.0, 4), cluster_score=round(float(dk_clusters.mean()), 4), marginal_entropy=round(marg_entropy, 4), n_layouts=len(dk_layouts))