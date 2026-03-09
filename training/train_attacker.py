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
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from battleship_rl.agents.defender import ClusteredDefender, EdgeBiasedDefender, ParityDefender, SpreadDefender, UniformRandomDefender
from battleship_rl.agents.policies import BattleshipFeatureExtractor
from battleship_rl.envs.battleship_env import BattleshipEnv
from battleship_rl.eval import EvalRecord, append_eval_record, run_eval
DEFENDER_MAP = {'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}
BASE_SEED = 42
N_ENVS = 16
N_STEPS = 2048
BATCH_SIZE = 1024
N_EPOCHS = 5
TOTAL_STEPS = 2000000
EVAL_FREQ = 200000
N_EVAL_EPS = 30
FINAL_EVAL_EPS = 200
ALL_MODES = ['UNIFORM', 'EDGE', 'CLUSTER', 'SPREAD', 'PARITY']
FAST_MODES = ['UNIFORM', 'SPREAD']

def mask_fn(env):
    return env.get_action_mask()

def make_single_env(defenders, weights, seed: int, debug: bool=False):
    """"""

    def _init():
        env = BattleshipEnv(defenders=defenders, defender_weights=weights, debug=debug)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init

def make_training_vec_env(defenders, weights, base_seed=BASE_SEED, n_envs=N_ENVS):
    fns = [make_single_env(defenders, weights, base_seed + rank) for rank in range(n_envs)]
    venv = SubprocVecEnv(fns)
    return VecMonitor(venv)

def make_eval_env_for_defender(defender_cls, seed: int):
    """"""

    def _init():
        env = BattleshipEnv(defender=defender_cls(), debug=False)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init

class TimestepEvalCallback(BaseCallback):
    """"""

    def __init__(self, eval_timesteps: list, out_dir: Path, regime: str, seed: int, git_hash: str, cli_args: dict, n_envs: int=4, intermediate_eps: int=30, final_eps: int=200, verbose: int=1):
        super().__init__(verbose)
        self.eval_timesteps = sorted(eval_timesteps)
        self.out_dir = out_dir
        self.regime = regime
        self.seed = seed
        self.git_hash = git_hash
        self.cli_args = cli_args
        self.n_envs = n_envs
        self.intermediate_eps = intermediate_eps
        self.final_eps = final_eps
        self._fired: set = set()
        self.log_path = out_dir / 'eval_log.jsonl'
        self.intermediate_modes = {'UNIFORM': UniformRandomDefender, 'SPREAD': SpreadDefender}
        self.all_modes = {'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        for target_ts in self.eval_timesteps:
            if ts >= target_ts and target_ts not in self._fired:
                self._fired.add(target_ts)
                is_final = target_ts == max(self.eval_timesteps)
                self._run_eval(ts, is_final=is_final)
        return True

    def _run_eval(self, timesteps: int, is_final: bool) -> None:
        import time as _time
        modes = self.all_modes if is_final else self.intermediate_modes
        n_eps = self.final_eps if is_final else self.intermediate_eps
        n_bs = 2000 if is_final else 300
        seed_off = BASE_SEED + 8000
        if self.verbose:
            label = 'FINAL' if is_final else 'intermediate'
            print(f'\n[eval @ {timesteps:,}] {label} — {len(modes)} modes, {n_eps} eps each ...')
        stats_dict, diagnostics = run_eval(model=self.model, mode_map=modes, n_episodes=n_eps, n_envs=self.n_envs, seed_offset=seed_off, collect_diagnostics=True, n_bootstrap=n_bs)
        for name, s in stats_dict.items():
            if self.verbose:
                print(f'  [{name:7s}] mean={s.mean:.1f}  p95={s.p95:.1f}  CVaR={s.cvar_10:.1f}  fail={s.fail_rate:.4f}  CI=[{s.ci_mean_lo:.1f},{s.ci_mean_hi:.1f}]')
        u = stats_dict.get('UNIFORM')
        sp = stats_dict.get('SPREAD')
        robust_gap = round(sp.mean - u.mean, 4) if u and sp else 0.0
        robust_gap_p95 = round(sp.p95 - u.p95, 4) if u and sp else 0.0
        record = EvalRecord(regime=self.regime, seed=self.seed, timesteps=timesteps, generation=0, git_hash=self.git_hash, timestamp=_time.strftime('%Y-%m-%dT%H:%M:%S'), cli_args=self.cli_args, stats={k: vars(v) for k, v in stats_dict.items()}, robust_gap=robust_gap, robust_gap_p95=robust_gap_p95, exploitability_defender=None, exploitability_attacker=None, uniform_drift=None, worst_D_k_mean=None, policy=vars(diagnostics), defender_shift=None)
        append_eval_record(record, self.log_path)
        if self.verbose:
            print(f'  [eval record] written → {self.log_path}')

class IBRSwitchCallback(BaseCallback):
    """"""

    def __init__(self, switch_n: int, verbose: int=0):
        super().__init__(verbose)
        self.switch_n = switch_n
        self._phase = 0
        self._update = 0

    def _on_rollout_end(self) -> None:
        self._update += 1
        if self._update % self.switch_n == 0:
            self._phase ^= 1
            new_weights = [0.0, 1.0] if self._phase else [1.0, 0.0]
            name = 'SpreadDefender' if self._phase else 'UniformRandomDefender'
            self.training_env.env_method('set_defender_weights', new_weights)
            if self.verbose:
                print(f'\n[IBR @update {self._update}] Active defender: {name}\n')

    def _on_step(self) -> bool:
        return True

def build_regime(regime: str, ibr_switch_n: int):
    if regime == 'A':
        defenders = [UniformRandomDefender()]
        weights = [1.0]
    elif regime == 'B':
        defenders = [UniformRandomDefender(), EdgeBiasedDefender(), ParityDefender(), SpreadDefender()]
        weights = [0.25, 0.25, 0.25, 0.25]
    elif regime == 'C':
        defenders = [UniformRandomDefender(), SpreadDefender()]
        weights = [1.0, 0.0]
    else:
        raise ValueError(f'Unknown regime: {regime!r}')
    return (defenders, weights)

def _git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'

def _write_run_meta(out_dir: Path, args) -> None:
    """"""
    meta = {'git_hash': _git_hash(), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'args': vars(args)}
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))

def _final_eval(model, out_dir: Path, n_episodes: int, n_envs: int, regime: str, seed: int, git_hash: str, cli_args: dict, timesteps: int) -> None:
    """"""
    print('\n[final eval] Running full 5-mode evaluation (mask-aware, n_bootstrap=2000) ...')
    stats_dict, diagnostics = run_eval(model=model, mode_map={'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}, n_episodes=n_episodes, n_envs=n_envs, seed_offset=BASE_SEED + 8000, collect_diagnostics=True, n_bootstrap=2000)
    for name, s in stats_dict.items():
        flag = '  ⚠ fail_rate > 0' if s.fail_rate > 0 else ''
        print(f'  [{name:7s}] mean={s.mean:.1f}  p95={s.p95:.1f}  CVaR={s.cvar_10:.1f}  fail={s.fail_rate:.4f}  CI=[{s.ci_mean_lo:.1f},{s.ci_mean_hi:.1f}]{flag}')
    import json as _json
    legacy = {k: {'mean': v.mean, 'p95': v.p95, 'fail_rate': v.fail_rate} for k, v in stats_dict.items()}
    (out_dir / 'final_eval.json').write_text(_json.dumps(legacy, indent=2))
    print(f'  Saved → {out_dir / 'final_eval.json'}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--total_steps', type=int, default=TOTAL_STEPS)
    parser.add_argument('--n_envs', type=int, default=N_ENVS)
    parser.add_argument('--ibr_switch_n', type=int, default=10, help='(Regime C) Switch defender every N PPO updates')
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ, help='(legacy) Evaluate every N global env steps — overridden by --eval_timesteps')
    parser.add_argument('--eval_eps', type=int, default=N_EVAL_EPS, help='Episodes per defender mode per intermediate eval')
    parser.add_argument('--eval_timesteps', nargs='+', type=int, default=[200000, 500000, 1000000, 2000000], help='Global timestep counts at which to run intermediate eval')
    parser.add_argument('--eval_modes', nargs='+', default=FAST_MODES, choices=ALL_MODES, metavar='MODE', help='(legacy) Defender modes for intermediate eval')
    parser.add_argument('--final_eval_eps', type=int, default=FINAL_EVAL_EPS, help='Episodes/mode for final full 5-mode eval')
    parser.add_argument('--out_dir', type=str, default=None, help='Override output directory (default: results/training/stage1/<regime>)')
    args = parser.parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path('results/training/stage1') / args.regime
    out_dir.mkdir(parents=True, exist_ok=True)
    Path('results/training/stage2').mkdir(parents=True, exist_ok=True)
    _write_run_meta(out_dir, args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    defenders, weights = build_regime(args.regime, args.ibr_switch_n)
    print(f'\n{'=' * 60}')
    print(f'  Regime {args.regime} | {args.total_steps:,} steps | {args.n_envs} envs | seed={args.seed}')
    print(f'  Defenders: {[type(d).__name__ for d in defenders]}')
    print(f'  Weights:   {weights}')
    print(f'{'=' * 60}\n')
    train_venv = make_training_vec_env(defenders, weights, base_seed=args.seed, n_envs=args.n_envs)
    policy_kwargs = dict(features_extractor_class=BattleshipFeatureExtractor, features_extractor_kwargs={'features_dim': 512}, net_arch=dict(pi=[512, 512], vf=[512, 512]))
    model = MaskablePPO(policy='CnnPolicy', env=train_venv, learning_rate=0.0003, n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5, policy_kwargs=policy_kwargs, tensorboard_log=str(out_dir / 'tensorboard'), device='cuda', verbose=1, seed=args.seed)
    git_hash = _git_hash()
    cli_dict = vars(args)
    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=200000 // args.n_envs, save_path=str(out_dir / 'checkpoints'), name_prefix='model'))
    eval_ts = sorted(set([0] + args.eval_timesteps + [args.total_steps]))
    callbacks.append(TimestepEvalCallback(eval_timesteps=eval_ts, out_dir=out_dir, regime=args.regime, seed=args.seed, git_hash=git_hash, cli_args=cli_dict, n_envs=4, intermediate_eps=args.eval_eps, final_eps=args.final_eval_eps, verbose=1))
    if args.regime == 'C':
        callbacks.append(IBRSwitchCallback(switch_n=args.ibr_switch_n, verbose=1))
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0
    model.save(str(out_dir / 'final_model'))
    train_venv.close()
    _final_eval(model=model, out_dir=out_dir, n_episodes=args.final_eval_eps, n_envs=4, regime=args.regime, seed=args.seed, git_hash=git_hash, cli_args=cli_dict, timesteps=args.total_steps)
    print(f'\nRegime {args.regime} done in {elapsed / 3600:.1f}h  ({args.total_steps / elapsed:.0f} steps/sec)')
    print(f'Artifacts: {out_dir.resolve()}\n')
if __name__ == '__main__':
    main()