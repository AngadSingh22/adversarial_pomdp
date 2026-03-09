""""""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from battleship_rl.agents.defender import EdgeBiasedDefender, SpreadDefender, UniformRandomDefender, ClusteredDefender, ParityDefender
from battleship_rl.eval.eval_lib import run_eval
try:
    from sb3_contrib import MaskablePPO
except ImportError as exc:
    raise ImportError('Install sb3-contrib: pip install sb3-contrib>=2.2') from exc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--n_envs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=8000)
    args = parser.parse_args()
    model = MaskablePPO.load(args.model)
    mode_map = {'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}
    stats, diag = run_eval(model=model, mode_map=mode_map, n_episodes=args.n_episodes, n_envs=args.n_envs, seed_offset=args.seed, collect_diagnostics=True, n_bootstrap=50)
    print('\n=== Stage-1 Smoke Eval ===')
    print(f'{'Mode':10s}  {'mean':>7}  {'std':>6}  {'p90':>7}  {'p95':>7}  {'cvar10':>7}')
    for name, ds in stats.items():
        print(f'{name:10s}  {ds.mean:>7.1f}  {ds.std:>6.1f}  {ds.p90:>7.1f}  {ds.p95:>7.1f}  {ds.cvar_10:>7.1f}')
    print(f'\n=== Diagnostics ===')
    print(f'action_entropy      = {diag.action_entropy:.4f}')
    print(f'adjacency_ratio     = {diag.adjacency_ratio:.4f}')
    print(f'time_to_first_hit   = {diag.time_to_first_hit:.2f}  (should be << mean shots)')
    print(f'time_to_first_sink  = {diag.time_to_first_sink:.2f}')
    print(f'invalid_action_rate = {diag.invalid_action_rate:.6f}  (should ≈ 0.0 with masks)')
    assert stats['UNIFORM'].std > 0.0, 'FAIL: std == 0 → likely degenerate'
    assert diag.time_to_first_hit > 0, 'FAIL: time_to_first_hit == 0 → likely uninitialized'
    assert diag.time_to_first_hit < stats['UNIFORM'].mean, f'FAIL: first_hit ({diag.time_to_first_hit:.1f}) >= mean ({stats['UNIFORM'].mean:.1f})'
    print('\nAll sanity checks passed ✓')
if __name__ == '__main__':
    main()