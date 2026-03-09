""""""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from battleship_rl.agents.defender import ClusteredDefender, EdgeBiasedDefender, ParityDefender, SpreadDefender, UniformRandomDefender
from battleship_rl.eval.eval_lib import defender_shift_metrics
BOARD_SIZE = (10, 10)
SHIPS = [5, 4, 3, 3, 2]
DEFENDERS = {'UNIFORM': UniformRandomDefender, 'EDGE': EdgeBiasedDefender, 'CLUSTER': ClusteredDefender, 'SPREAD': SpreadDefender, 'PARITY': ParityDefender}

def _generate_layouts(defender_cls, n: int, seed: int) -> np.ndarray:
    """"""
    rng = np.random.default_rng(seed)
    defender = defender_cls()
    board_h, board_w = BOARD_SIZE
    layouts = []
    for _ in range(n):
        grid = np.full((board_h, board_w), -1, dtype=np.int32)
        layout = defender.sample_layout(board_size=BOARD_SIZE, ships=SHIPS, rng=rng)
        layouts.append(layout)
    return np.array(layouts)

def _quadrant_mass_std(layouts: np.ndarray) -> float:
    """"""
    H, W = (layouts.shape[1], layouts.shape[2])
    hh, hw = (H // 2, W // 2)
    stds = []
    for layout in layouts:
        q1 = np.sum(layout[:hh, :hw] >= 0)
        q2 = np.sum(layout[:hh, hw:] >= 0)
        q3 = np.sum(layout[hh:, :hw] >= 0)
        q4 = np.sum(layout[hh:, hw:] >= 0)
        stds.append(float(np.std([q1, q2, q3, q4])))
    return float(np.mean(stds))

def compute_metrics(n_layouts: int, seed: int) -> dict:
    """"""
    print(f'Generating {n_layouts} UNIFORM layouts (seed={seed}) as baseline ...')
    uniform_layouts = _generate_layouts(UniformRandomDefender, n_layouts, seed=seed + 9999)
    results = {}
    for name, cls in DEFENDERS.items():
        print(f'  Computing metrics for {name} ...')
        layouts = _generate_layouts(cls, n_layouts, seed=seed)
        shift = defender_shift_metrics(dk_layouts=layouts, uniform_layouts=uniform_layouts)
        q_std = _quadrant_mass_std(layouts)
        results[name] = {'centroid_pairwise_mean': round(shift.centroid_pairwise_mean, 4), 'cluster_score': round(shift.cluster_score, 4), 'marginal_entropy': round(shift.marginal_entropy, 6), 'quadrant_mass_std': round(q_std, 4)}
    metadata = {'generation_parameters': {'n_layouts': n_layouts, 'seed': seed, 'uniform_baseline_seed': seed + 9999, 'board_size': list(BOARD_SIZE), 'ships': SHIPS}, 'metric_definitions': {'centroid_pairwise_mean': 'Mean all-pairs Euclidean distance between ship centroids within each layout, averaged over N layouts. Higher = ships placed farther apart on average.', 'cluster_score': 'Mean count of 4-connected (horizontal + vertical) adjacent occupied-cell pairs per layout. Counts same-ship adjacencies (ships are always internally adjacent) plus cross-ship adjacency. Higher = tighter clustering.', 'marginal_entropy': 'Shannon entropy of the marginal per-cell occupation probability distribution, averaged over all H*W cells: mean_{c} [-p_c log(p_c) - (1-p_c) log(1-p_c)]. Higher = more uniform spatial distribution; lower = ships concentrated in specific cells. Measured in nats.', 'quadrant_mass_std': 'Standard deviation of the occupied-cell count across the 4 board quadrants (top-left, top-right, bottom-left, bottom-right), averaged over N layouts. Higher = ships systematically concentrated in certain quadrants.'}, 'interpretation_notes': 'All metrics are within-layout: no cross-layout ship-ID matching is required. The UNIFORM baseline uses a separate fixed seed (seed+9999) and the same N layouts. defender_steps for D_k layouts are logged separately in eval_log.jsonl under EvalRecord.defender_budget. For preprint citation: run with --n_layouts=500 --seed=42 to reproduce §5 shift table.', 'defender_descriptions': {'UNIFORM': 'Uniform-random ship placement (rejection-sampled).', 'EDGE': 'Edge-biased: ships placed preferentially near board boundaries.', 'CLUSTER': 'Clustered: ships placed near each other (attraction kernel).', 'SPREAD': 'Spread: ships placed to maximise minimum pairwise centroid distance.', 'PARITY': 'Parity: alternating horizontal/vertical placement on even/odd rows.'}}
    return {'metadata': metadata, 'results': results}

def _markdown_table(metrics: dict) -> str:
    """"""
    results = metrics.get('results', metrics)
    header = '| Defender | centroid_pairwise_mean | cluster_score | marginal_entropy | quadrant_mass_std |'
    sep = '|---|---|---|---|---|'
    rows = [header, sep]
    for name, m in results.items():
        rows.append(f'| {name} | {m['centroid_pairwise_mean']:.4f} | {m['cluster_score']:.4f} | {m['marginal_entropy']:.6f} | {m['quadrant_mass_std']:.4f} |')
    return '\n'.join(rows)

def main():
    parser = argparse.ArgumentParser(description='Compute defender distribution-shift metrics for preprint.')
    parser.add_argument('--n_layouts', type=int, default=500, help='Layouts per defender (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed (default: 42)')
    parser.add_argument('--out', type=str, default='results/defender_metrics.json', help='Output JSON path')
    args = parser.parse_args()
    metrics = compute_metrics(args.n_layouts, args.seed)
    print('\n' + _markdown_table(metrics) + '\n')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f'[saved] {out_path.resolve()}')
    first_key = next(iter(metrics['results']))
    print(f'\n[JSON header preview] {first_key}: {json.dumps(metrics['results'][first_key])}')
if __name__ == '__main__':
    main()