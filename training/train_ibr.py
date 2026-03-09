""""""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from sb3_contrib.common.wrappers import ActionMasker
from battleship_rl.agents.defender import ClusteredDefender, EdgeBiasedDefender, ParityDefender, SpreadDefender, UniformRandomDefender
from battleship_rl.agents.policies import BattleshipFeatureExtractor
from battleship_rl.envs.battleship_env import BattleshipEnv
from battleship_rl.envs.defender_env import DefenderEnv, build_layout_pool, evaluate_attacker_on_layout
from battleship_rl.eval import EvalRecord, append_eval_record, defender_shift_metrics
from battleship_rl.eval.eval_lib import run_eval
from battleship_rl.eval.schema import DistributionStats
BASE_SEED = 42
N_ATCK_ENVS = 16
POOL_SIZE = 50000
BOARD_SIZE = 10
SHIPS = [5, 4, 3, 3, 2]
SCRIPTED_DEFENDERS = {'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}
ATTACKER_PPO_KWARGS = dict(policy='CnnPolicy', learning_rate=0.0003, n_steps=2048, batch_size=1024, n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5, policy_kwargs=dict(features_extractor_class=BattleshipFeatureExtractor, features_extractor_kwargs={'features_dim': 512}, net_arch=dict(pi=[512, 512], vf=[512, 512])), device='cuda', verbose=0)

class LearnedLayoutDefender:
    """"""

    def __init__(self, layouts: np.ndarray):
        if layouts.ndim != 3 or len(layouts) == 0:
            raise ValueError(f'layouts must be (N, H, W) with N>0, got {layouts.shape}')
        self._layouts = layouts

    def sample_layout(self, board_size: tuple, ships: list, rng: np.random.Generator) -> np.ndarray:
        """"""
        idx = int(rng.integers(0, len(self._layouts)))
        return self._layouts[idx].copy()

def extract_dk_layouts(defender_model: PPO, layout_pool: np.ndarray, n_layouts: int=2000, seed: int=0) -> np.ndarray:
    """"""
    obs_shape = defender_model.observation_space.shape
    rng = np.random.default_rng(seed)
    device_orig = defender_model.device
    defender_model.policy.set_training_mode(False)
    chosen_indices = []
    for i in range(n_layouts):
        obs = rng.uniform(0.0, 1.0, size=obs_shape).astype(np.float32)
        idx, _ = defender_model.predict(obs, deterministic=False)
        idx_clipped = int(idx) % len(layout_pool)
        chosen_indices.append(idx_clipped)
    unique_indices = list(set(chosen_indices))
    dk_layouts = layout_pool[unique_indices]
    print(f'  [D_k extract] {n_layouts} rolls → {len(unique_indices)} unique layouts (pool coverage: {100 * len(unique_indices) / len(layout_pool):.1f}%)')
    return dk_layouts

def mask_fn(env):
    return env.get_action_mask()

def make_attacker_env(defender, seed: int):
    """"""

    def _init():
        env = BattleshipEnv(defender=defender, debug=False)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init

def make_attacker_vec_env(defenders: list, weights: list, base_seed: int):
    """"""
    fns = []
    for rank in range(N_ATCK_ENVS):

        def _init(r=rank):
            env = BattleshipEnv(defenders=defenders, defender_weights=weights, debug=False)
            env = ActionMasker(env, mask_fn)
            env.reset(seed=base_seed + r)
            return env
        fns.append(_init)
    return VecMonitor(SubprocVecEnv(fns))

def evaluate_attacker_scripted(attacker: MaskablePPO, n_episodes: int=100, seed: int=9000) -> dict:
    """"""
    results = {}
    for name, cls in SCRIPTED_DEFENDERS.items():
        lengths = []
        env = BattleshipEnv(defender=cls(), debug=False)
        try:
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                steps = 0
                while not done:
                    mask = env.get_action_mask()
                    action, _ = attacker.predict(obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True)
                    obs, _, term, trunc, _ = env.step(int(action[0]))
                    steps += 1
                    done = term or trunc
                lengths.append(steps)
        finally:
            env.close()
        arr = np.array(lengths)
        results[name] = {'mean': float(arr.mean()), 'p95': float(np.percentile(arr, 95))}
    return results

def evaluate_attacker_on_learned_defender(attacker: MaskablePPO, defender_model: PPO, layout_pool: np.ndarray, n_episodes: int=50, seed: int=9999) -> dict:
    """"""
    lengths = []
    pool_size = len(layout_pool)
    obs_shape = defender_model.observation_space.shape
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        obs_d = rng.uniform(0.0, 0.1, size=obs_shape).astype(np.float32)
        action_d, _ = defender_model.predict(obs_d, deterministic=False)
        layout_idx = int(action_d) % pool_size
        layout = layout_pool[layout_idx]
        mean_shots, ep_shots = evaluate_attacker_on_layout(layout=layout, attacker_policy=attacker, k_episodes=1, board_size=BOARD_SIZE, ships=SHIPS, seed=seed + ep)
        lengths.append(ep_shots[0])
    arr = np.array(lengths)
    return {'mean': float(arr.mean()), 'p95': float(np.percentile(arr, 95))}

def train_defender(generation: int, frozen_attacker: MaskablePPO, layout_pool: np.ndarray, steps: int, out_dir: Path, max_generations: int, seed: int=BASE_SEED, k_eval: int=1) -> PPO:
    """"""
    def_env = DefenderEnv(layout_pool=layout_pool, attacker_policy=frozen_attacker, k_eval_episodes=k_eval, n_eval_parallel=8, generation=generation, max_generations=max_generations, board_size=BOARD_SIZE, ships=SHIPS, seed=seed)
    def_venv = VecMonitor(DummyVecEnv([lambda: def_env]))
    defender_model = PPO(policy='MlpPolicy', env=def_venv, learning_rate=0.0003, n_steps=1024, batch_size=512, n_epochs=5, gamma=0.99, ent_coef=0.01, device='cpu', verbose=1, tensorboard_log=str(out_dir / 'tensorboard_defender'), seed=seed)
    print(f'  [Gen {generation}] Training defender for {steps:,} steps (k_eval={k_eval}) ...')
    defender_model.learn(total_timesteps=steps, progress_bar=False)
    save_path = str(out_dir / f'defender_gen_{generation}')
    defender_model.save(save_path)
    print(f'  [Gen {generation}] Defender saved → {save_path}.zip')
    def_venv.close()
    return defender_model

def train_attacker(generation: int, init_attacker: Optional[MaskablePPO], defender_model: PPO, layout_pool: np.ndarray, steps: int, out_dir: Path, seed: int=BASE_SEED, uniform_weight: float=0.33, spread_weight: float=0.17, dk_extract_n: int=2000) -> MaskablePPO:
    """"""
    dk_weight = max(0.0, 1.0 - uniform_weight - spread_weight)
    print(f'  [Gen {generation}] Extracting D_k layout distribution ({dk_extract_n} samples) ...')
    dk_layouts = extract_dk_layouts(defender_model, layout_pool, n_layouts=dk_extract_n, seed=seed)
    dk_defender = LearnedLayoutDefender(dk_layouts)
    defenders = [UniformRandomDefender(), SpreadDefender(), dk_defender]
    def_weights = [uniform_weight, spread_weight, dk_weight]
    print(f'  [Gen {generation}] Training attacker for {steps:,} steps ...')
    print(f'    Attacker mix: UNIFORM={uniform_weight:.2f}  SPREAD={spread_weight:.2f}  D_k={dk_weight:.2f}')
    print(f'    D_k unique layouts: {len(dk_layouts)}')
    train_venv = make_attacker_vec_env(defenders, def_weights, base_seed=seed)
    attacker = MaskablePPO(env=train_venv, **{k: v for k, v in ATTACKER_PPO_KWARGS.items() if k not in ('device', 'verbose')}, device='cuda', verbose=0, seed=seed, tensorboard_log=str(out_dir / 'tensorboard_attacker'))
    if init_attacker is not None:
        attacker.set_parameters(init_attacker.get_parameters())
    attacker.learn(total_timesteps=steps, progress_bar=False)
    save_path = str(out_dir / f'attacker_gen_{generation}')
    attacker.save(save_path)
    print(f'  [Gen {generation}] Attacker A_{generation} saved → {save_path}.zip')
    train_venv.close()
    return (attacker, dk_layouts)

class FixedPoolDefender:
    """"""

    def __init__(self, layouts: np.ndarray):
        if layouts.ndim != 3 or len(layouts) == 0:
            raise ValueError(f'layouts must be (N, H, W) with N>0, got {layouts.shape}')
        self._layouts = layouts

    def sample_layout(self, board_size: tuple, ships: list, rng: np.random.Generator) -> np.ndarray:
        idx = int(rng.integers(0, len(self._layouts)))
        return self._layouts[idx].copy()

def evaluate_generation(generation: int, attacker_before: MaskablePPO, attacker_after: MaskablePPO, defender_model: PPO, layout_pool: np.ndarray, out_dir: Path, dk_layouts: Optional[np.ndarray]=None, seed: int=BASE_SEED, git_hash: str='unknown', cli_args: Optional[dict]=None, n_scripted_eps: int=100, n_learned_eps: int=50, defender_budget: Optional[int]=None) -> dict:
    """"""
    n_envs_eval = min(4, n_scripted_eps)
    n_envs_dk = min(4, n_learned_eps)
    n_bootstrap = 300
    scripted_map = {'UNIFORM': UniformRandomDefender, 'SPREAD': SpreadDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'PARITY': ParityDefender}

    def _make_fixed_defender_cls(layouts):
        """"""

        class _FixedCls(FixedPoolDefender):
            pass
        _FixedCls._layouts_ref = layouts

        def _init(self):
            FixedPoolDefender.__init__(self, _FixedCls._layouts_ref)
        _FixedCls.__init__ = _init
        return _FixedCls
    dk_cls = _make_fixed_defender_cls(dk_layouts) if dk_layouts is not None and len(dk_layouts) > 0 else None
    print(f'  [Gen {generation}] Evaluating A_{{k-1}} (before) vs scripted modes ...')
    before_scripted_stats, _ = run_eval(model=attacker_before, mode_map=scripted_map, n_episodes=n_scripted_eps, n_envs=n_envs_eval, seed_offset=seed + 9000, collect_diagnostics=False, n_bootstrap=n_bootstrap)
    before_dk_stats: dict = {}
    if dk_cls is not None:
        print(f'  [Gen {generation}] Evaluating A_{{k-1}} vs D_k ...')
        dk_map = {'D_k': dk_cls}
        before_dk_eval, _ = run_eval(model=attacker_before, mode_map=dk_map, n_episodes=n_learned_eps, n_envs=n_envs_dk, seed_offset=seed + 9999, collect_diagnostics=False, n_bootstrap=n_bootstrap)
        before_dk_stats = before_dk_eval
    print(f'  [Gen {generation}] Evaluating A_k (after) vs scripted modes + D_k ...')
    mode_map_after = dict(scripted_map)
    if dk_cls is not None:
        mode_map_after['D_k'] = dk_cls
    after_stats, after_diag = run_eval(model=attacker_after, mode_map=mode_map_after, n_episodes=n_scripted_eps, n_envs=n_envs_eval, seed_offset=seed + 9000 + 1, collect_diagnostics=True, n_bootstrap=n_bootstrap)
    un_b = before_scripted_stats['UNIFORM'].mean
    sp_b = before_scripted_stats['SPREAD'].mean
    dk_b = before_dk_stats['D_k'].mean if 'D_k' in before_dk_stats else un_b
    un_a = after_stats['UNIFORM'].mean
    sp_a = after_stats['SPREAD'].mean
    dk_a = after_stats['D_k'].mean if 'D_k' in after_stats else un_a
    defender_adversarial = dk_b - un_b
    attacker_adaptation = dk_a - dk_b
    uniform_drift = un_a - un_b
    print(f'\n  --- Gen {generation} summary ---')
    print(f'  {'':20s}  {'UNIFORM':>8}  {'SPREAD':>8}  {'vs_D_k':>8}')
    print(f'  {'A_{{k-1}} (before)':20s}  {un_b:>8.1f}  {sp_b:>8.1f}  {dk_b:>8.1f}')
    print(f'  {'A_k (after)':20s}  {un_a:>8.1f}  {sp_a:>8.1f}  {dk_a:>8.1f}')
    print(f'  {'delta':20s}  {un_a - un_b:>+8.1f}  {sp_a - sp_b:>+8.1f}  {dk_a - dk_b:>+8.1f}')
    print()
    adv_flag = 'OK' if defender_adversarial > 0 else 'WARN: D_k not harder than UNIFORM'
    ada_flag = 'OK' if attacker_adaptation < 0 else 'WARN: attacker did not improve vs D_k'
    uni_flag = 'OK' if abs(uniform_drift) < 3 else f'WARN: UNIFORM drift={uniform_drift:+.1f}'
    print(f'  [check A] defender_adversarial={defender_adversarial:+.1f}  ({adv_flag})')
    print(f'  [check B] attacker_adaptation={attacker_adaptation:+.1f}  ({ada_flag})')
    print(f'  [check C] uniform_drift={uniform_drift:+.1f}  ({uni_flag})')
    if defender_adversarial < 0:
        print(f'  [sign check] WARNING: exploitability_defender={defender_adversarial:+.3f} should be >0 (D_k should be harder than UNIFORM)')
    if attacker_adaptation > 0:
        print(f'  [sign check] WARNING: exploitability_attacker={attacker_adaptation:+.3f} should be <0 (A_k should improve vs D_k)')
    results = {'generation': generation, 'before': {'scripted_modes': {k: {'mean': v.mean, 'p95': v.p95} for k, v in before_scripted_stats.items()}, 'vs_D_k': {'mean': dk_b, 'p95': before_dk_stats.get('D_k', before_scripted_stats['UNIFORM']).p95}}, 'after': {'scripted_modes': {k: {'mean': v.mean, 'p95': v.p95} for k, v in after_stats.items()}, 'vs_D_k': {'mean': dk_a, 'p95': after_stats.get('D_k', after_stats['UNIFORM']).p95}}, 'checks': {'defender_adversarial': defender_adversarial, 'attacker_adaptation': attacker_adaptation, 'uniform_drift': uniform_drift, 'delta_spread_vs_uniform': sp_a - un_a}}
    out_path = out_dir / f'eval_gen_{generation}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  [eval saved] {out_path}')
    shift = None
    if dk_layouts is not None and len(dk_layouts) > 0:
        uniform_sample_n = min(len(dk_layouts), 2000)
        rng = np.random.default_rng(generation)
        unif_idx = rng.integers(0, len(layout_pool), size=uniform_sample_n)
        uniform_sample = layout_pool[unif_idx]
        shift = defender_shift_metrics(dk_layouts=np.array(dk_layouts), uniform_layouts=uniform_sample)
        print(f'  [shift] centroid_pairwise_mean={shift.centroid_pairwise_mean:.2f}  cluster_score={shift.cluster_score:.2f}  marginal_entropy={shift.marginal_entropy:.4f}')

    def _ds_to_dict(ds: DistributionStats) -> dict:
        return vars(ds)
    stats: dict = {}
    for name, ds in before_scripted_stats.items():
        stats[f'{name}_before'] = _ds_to_dict(ds)
    if 'D_k' in before_dk_stats:
        stats['D_k_before'] = _ds_to_dict(before_dk_stats['D_k'])
    for name, ds in after_stats.items():
        stats[f'{name}_after'] = _ds_to_dict(ds)
    record = EvalRecord(regime='IBR', seed=seed if seed is not None else BASE_SEED, timesteps=generation * 1000000, generation=generation, git_hash=git_hash if git_hash is not None else 'unknown', timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'), cli_args=cli_args if cli_args is not None else {}, stats=stats, robust_gap=round(sp_a - un_a, 4), robust_gap_p95=round(after_stats['SPREAD'].p95 - after_stats['UNIFORM'].p95, 4), exploitability_defender=round(defender_adversarial, 4), exploitability_attacker=round(attacker_adaptation, 4), uniform_drift=round(uniform_drift, 4), worst_D_k_mean=None, defender_budget=defender_budget, policy=vars(after_diag), defender_shift=vars(shift) if shift is not None else None)
    append_eval_record(record, out_dir / 'eval_log.jsonl')
    return results
ST1_ROOT = Path('results/training/stage1')

def select_best_stage1_model() -> Path:
    """"""
    CORRUPTION_THRESHOLD = 5.0
    candidates = []
    for regime_dir in sorted(ST1_ROOT.iterdir()):
        if not regime_dir.is_dir():
            continue
        model_zip = regime_dir / 'final_model.zip'
        if not model_zip.exists():
            continue
        eval_file = regime_dir / 'final_eval_corrected.json'
        if not eval_file.exists():
            eval_file = regime_dir / 'final_eval.json'
        if not eval_file.exists():
            continue
        data = json.loads(eval_file.read_text())
        spread_mean = data.get('SPREAD', {}).get('mean', float('inf'))
        uniform_mean = data.get('UNIFORM', {}).get('mean', float('inf'))
        if spread_mean <= CORRUPTION_THRESHOLD or uniform_mean <= CORRUPTION_THRESHOLD:
            print(f'  [auto-select] Regime {regime_dir.name}: eval looks corrupted (SPREAD={spread_mean:.2f}, UNIFORM={uniform_mean:.2f}) — skipping')
            continue
        src = eval_file.name
        candidates.append((spread_mean, uniform_mean, regime_dir.name, model_zip, src))
    if not candidates:
        raise FileNotFoundError(f'No valid (non-corrupted) eval files found under {ST1_ROOT}. Re-run eval with masks, or pass --init_attacker explicitly.')
    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    print(f'\n[auto-select] Best Stage 1 regime: {best[2]} (from {best[4]})')
    print(f'  SPREAD mean={best[0]:.2f}  UNIFORM mean={best[1]:.2f}')
    print(f'  Model: {best[3]}')
    for s, u, name, _, src in candidates:
        marker = ' ← selected' if name == best[2] else ''
        print(f'  Regime {name}: SPREAD={s:.2f}  UNIFORM={u:.2f}  [{src}]{marker}')
    return best[3]

def _git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Iterative Best Response training loop.')
    parser.add_argument('--init_attacker', default=None, help='Path to Stage 1 attacker checkpoint (.zip). If omitted, auto-selects the regime with lowest SPREAD mean from results/training/stage1/*/final_eval.json.')
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--attacker_steps', type=int, default=1000000)
    parser.add_argument('--defender_steps', type=int, default=50000, help='PPO steps for defender training per generation. Keep small (30k-50k) — each step runs k_eval attacker rollouts.')
    parser.add_argument('--k_eval', type=int, default=1, help='Attacker rollouts per defender env step (default 1). Higher = more signal but O(k_eval) slower defender training.')
    parser.add_argument('--pool_size', type=int, default=POOL_SIZE)
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--n_eval_eps', type=int, default=100)
    parser.add_argument('--uniform_weight', type=float, default=0.50, help='Fraction of attacker training envs using UNIFORM defender (default 0.50)')
    parser.add_argument('--spread_weight', type=float, default=0.0, help='Fraction of attacker training envs using SPREAD defender (default 0.0). Remaining fraction goes to D_k (learned).')
    parser.add_argument('--out_dir', type=str, default='results/training/stage2', help='Output directory for all artifacts (default: results/training/stage2)')
    parser.add_argument('--dk_extract_n', type=int, default=5000, help='Number of samples to extract for the empirical D_k distribution (default 5000)')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.init_attacker is None:
        init_path = select_best_stage1_model()
    else:
        init_path = Path(args.init_attacker)
        if not init_path.exists():
            raise FileNotFoundError(f'--init_attacker not found: {init_path}')
    meta = {'git_hash': _git_hash(), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'init_attacker': str(init_path), 'args': vars(args)}
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[run_meta] {out_dir / 'run_meta.json'}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f'\nBuilding layout pool ({args.pool_size:,} layouts) ...')
    t0 = time.time()
    layout_pool = build_layout_pool(args.pool_size, board_size=BOARD_SIZE, ships=SHIPS, seed=args.seed)
    print(f'Done in {time.time() - t0:.1f}s  shape={layout_pool.shape}')
    print(f'\nLoading A_0 from {init_path} ...')
    dummy_env = make_attacker_vec_env([UniformRandomDefender()], [1.0], base_seed=args.seed)
    attacker = MaskablePPO.load(str(init_path), env=dummy_env, device='cuda')
    dummy_env.close()
    learned_defenders: List[PPO] = []
    all_eval_results = []
    print(f'\n{'=' * 60}')
    print(f'  IBR Training: {args.generations} generations')
    print(f'  Attacker budget: {args.attacker_steps:,} steps/gen')
    print(f'  Defender budget: {args.defender_steps:,} steps/gen')
    print(f'{'=' * 60}\n')
    git_hash = _git_hash()
    for gen in range(1, args.generations + 1):
        t_gen = time.time()
        print(f'\n{'—' * 40}')
        print(f' Generation {gen}/{args.generations}')
        print(f'{'—' * 40}')
        defender_model = train_defender(generation=gen, frozen_attacker=attacker, layout_pool=layout_pool, steps=args.defender_steps, out_dir=out_dir, max_generations=args.generations, seed=args.seed + gen * 1000, k_eval=args.k_eval)
        learned_defenders.append(defender_model)
        attacker_before = attacker
        attacker, gen_dk_layouts = train_attacker(generation=gen, init_attacker=attacker, defender_model=defender_model, layout_pool=layout_pool, steps=args.attacker_steps, out_dir=out_dir, seed=args.seed + gen * 2000, uniform_weight=args.uniform_weight, spread_weight=args.spread_weight, dk_extract_n=args.dk_extract_n)
        eval_row = evaluate_generation(generation=gen, attacker_before=attacker_before, attacker_after=attacker, defender_model=defender_model, layout_pool=layout_pool, out_dir=out_dir, dk_layouts=gen_dk_layouts, seed=args.seed, git_hash=git_hash, cli_args=vars(args), n_scripted_eps=args.n_eval_eps, n_learned_eps=50, defender_budget=args.defender_steps)
        all_eval_results.append(eval_row)
        elapsed = time.time() - t_gen
        print(f'  Generation {gen} done in {elapsed / 60:.1f} min')
    summary_path = out_dir / 'ibr_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_eval_results, f, indent=2)
    print(f'\nIBR complete. Summary → {summary_path.resolve()}')
    print('\n=== FINAL IBR RESULTS ===')
    print(f'{'Gen':>5}  {'A_k UNIFORM':>11}  {'A_k SPREAD':>10}  {'A_k vs D_k':>10}  {'UNIFORM drift':>13}  {'Exploit(def)':>12}')
    for r in all_eval_results:
        a = r['after']
        chk = r['checks']
        exploit = chk.get('exploitability_proxy', chk.get('defender_adversarial', float('nan')))
        print(f'  {r['generation']:>3}  {a['scripted_modes']['UNIFORM']['mean']:>11.1f}  {a['scripted_modes']['SPREAD']['mean']:>10.1f}  {a['vs_D_k']['mean']:>10.1f}  {chk['uniform_drift']:>+13.1f}  {exploit:>+12.3f}')
if __name__ == '__main__':
    main()