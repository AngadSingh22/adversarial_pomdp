""""""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
SEEDS = [42, 123, 777]
REGIMES = ['A', 'B', 'C']
TOTAL_STEPS = 2000000
EVAL_TIMESTEPS = ['200000', '500000', '1000000', '2000000']
PYTHON = '.venv/bin/python3'
ROOT = Path(__file__).resolve().parent.parent

def run(cmd: list[str], dry_run: bool) -> None:
    print(f'\n[run] {' '.join(cmd)}\n')
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)

def train_stage1(regime: str, seed: int, dry_run: bool) -> None:
    out_dir = f'results/experiments/{regime}/seed_{seed}'
    cmd = [PYTHON, '-u', 'training/train_attacker.py', '--regime', regime, '--seed', str(seed), '--total_steps', str(TOTAL_STEPS), '--out_dir', out_dir, '--eval_timesteps', *EVAL_TIMESTEPS, '--final_eval_eps', '200', '--eval_eps', '30']
    run(cmd, dry_run)

def train_ibr(seed: int, dry_run: bool) -> None:
    init_path = Path(f'results/experiments/B/seed_{seed}/final_model.zip')
    if not (ROOT / init_path).exists():
        raise FileNotFoundError(f'IBR init model missing for seed {seed}: {init_path}\nRun Regime B training first (python tools/run_experiment.py --regime B).')
    out_dir = f'results/experiments/IBR/seed_{seed}'
    cmd = [PYTHON, '-u', 'training/train_ibr.py', '--init_attacker', str(init_path), '--generations', '3', '--attacker_steps', '1000000', '--defender_steps', '50000', '--k_eval', '1', '--uniform_weight', '0.33', '--spread_weight', '0.17', '--seed', str(seed), '--out_dir', out_dir]
    run(cmd, dry_run)

def main() -> None:
    parser = argparse.ArgumentParser(description='Run full 12-job preprint experiment matrix')
    parser.add_argument('--regime', choices=['A', 'B', 'C', 'IBR'], default=None, help='Run only this regime (default: all)')
    parser.add_argument('--seed', type=int, default=None, help='Run only this seed (default: all)')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without executing')
    args = parser.parse_args()
    regimes = [args.regime] if args.regime else REGIMES + ['IBR']
    seeds = [args.seed] if args.seed else SEEDS
    if args.dry_run:
        print('[dry_run] -- no commands will be executed --\n')
    for regime in regimes:
        for seed in seeds:
            if regime == 'IBR':
                train_ibr(seed, args.dry_run)
            else:
                train_stage1(regime, seed, args.dry_run)
    print('\nAll jobs completed.')
if __name__ == '__main__':
    main()