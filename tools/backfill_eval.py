""""""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from battleship_rl.agents.defender import ClusteredDefender, EdgeBiasedDefender, ParityDefender, SpreadDefender, UniformRandomDefender
from battleship_rl.eval import EvalRecord, append_eval_record
from battleship_rl.eval.eval_lib import run_eval
try:
    from sb3_contrib import MaskablePPO
except ImportError as exc:
    raise ImportError('sb3-contrib is required. Install via: pip install sb3-contrib>=2.2') from exc
try:
    import subprocess
    _git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    _git = 'unknown'
MODE_MAP = {'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}

def _find_checkpoints(model_dir: Path) -> list[Path]:
    """"""
    zips = []
    zips.extend(model_dir.glob('checkpoint_*.zip'))
    zips.extend(model_dir.glob('attacker_gen_*.zip'))
    zips.extend(model_dir.glob('final_model*.zip'))
    final = [z for z in zips if z.stem == 'final_model']
    others = [z for z in zips if z.stem != 'final_model']
    return sorted(others) + final

def backfill(model_dir: Path, n_episodes: int, n_envs: int, seed: int) -> None:
    model_dir = model_dir.resolve()
    checkpoints = _find_checkpoints(model_dir)
    if not checkpoints:
        print(f'[backfill] No .zip checkpoints found in {model_dir}')
        return
    print(f'[backfill] Found {len(checkpoints)} checkpoint(s) in {model_dir}')
    corrected_jsonl = model_dir / 'eval_log_corrected.jsonl'
    if corrected_jsonl.exists():
        corrected_jsonl.unlink()
    final_corrected: dict | None = None
    for ckpt in checkpoints:
        print(f'\n[backfill] Loading {ckpt.name} ...')
        model = MaskablePPO.load(str(ckpt))
        stats, diag = run_eval(model=model, mode_map=MODE_MAP, n_episodes=n_episodes, n_envs=n_envs, seed_offset=seed, collect_diagnostics=True, n_bootstrap=300)
        try:
            ts = int(''.join(filter(str.isdigit, ckpt.stem)))
        except Exception:
            ts = -1
        record = EvalRecord(regime='BACKFILL', seed=seed, timesteps=ts, generation=None, git_hash=_git, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'), cli_args={'checkpoint': str(ckpt), 'n_episodes': n_episodes}, stats={k: vars(v) for k, v in stats.items()}, robust_gap=round(stats.get('SPREAD', stats.get('EDGE', list(stats.values())[0])).mean - stats['UNIFORM'].mean, 4), robust_gap_p90=round(stats.get('SPREAD', stats.get('EDGE', list(stats.values())[0])).p90 - stats['UNIFORM'].p90, 4), exploitability_defender=None, exploitability_attacker=None, uniform_drift=None, worst_D_k_mean=None, defender_budget=None, policy=vars(diag), defender_shift=None)
        append_eval_record(record, corrected_jsonl)
        if ckpt.stem == 'final_model':
            final_corrected = {k: vars(v) for k, v in stats.items()}
            out_json = model_dir / 'final_eval_corrected.json'
            out_json.write_text(json.dumps(final_corrected, indent=2))
            print(f'  [corrected final eval] → {out_json}')
        print(f'  UNIFORM: mean={stats['UNIFORM'].mean:.1f}  SPREAD: mean={stats.get('SPREAD', stats['UNIFORM']).mean:.1f}  first_hit={diag.time_to_first_hit:.1f}  first_sink={diag.time_to_first_sink:.1f}')
    print(f'\n[backfill] Done. Corrected JSONL: {corrected_jsonl}')

def main():
    parser = argparse.ArgumentParser(description='Post-hoc diagnostic backfill using fixed eval_lib (Fix A).')
    parser.add_argument('--model_dir', required=True, help='Directory containing checkpoint .zip files')
    parser.add_argument('--n_episodes', type=int, default=200, help='Episodes per mode per checkpoint (default: 200)')
    parser.add_argument('--n_envs', type=int, default=4, help='Parallel envs for run_eval (default: 4)')
    parser.add_argument('--seed', type=int, default=8000, help='Seed offset for eval (default: 8000)')
    args = parser.parse_args()
    backfill(model_dir=Path(args.model_dir), n_episodes=args.n_episodes, n_envs=args.n_envs, seed=args.seed)
if __name__ == '__main__':
    main()