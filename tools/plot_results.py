""""""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('[WARN] matplotlib not installed — figures will not be generated, only table will print')
REGIME_COLORS = {'A': '#E15759', 'B': '#4E79A7', 'C': '#F28E2B', 'IBR': '#59A14F'}
REGIME_LABELS = {'A': 'Regime A (UNIFORM-only)', 'B': 'Regime B (Fixed mixture)', 'C': 'Regime C (IBR-lite)', 'IBR': 'IBR (Full adversarial)'}

def load_records(results_root: Path) -> list[dict]:
    """"""
    records = []
    for jsonl_path in sorted(results_root.rglob('eval_log.jsonl')):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records

def group_by_regime(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in records:
        regime = r.get('regime', '?')
        grouped.setdefault(regime, []).append(r)
    return grouped

def plot_learning_curves(records: list[dict], out_path: Path) -> None:
    if not HAS_MPL:
        return
    grouped = group_by_regime(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle('Learning Curves: Mean Shots-to-Win', fontsize=14, fontweight='bold')
    for ax, mode_key in zip(axes, ['UNIFORM', 'SPREAD']):
        ax.set_title(f'{mode_key} defender', fontsize=12)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('E[τ] (shots-to-win, lower = better)')
        for regime, recs in sorted(grouped.items()):
            color = REGIME_COLORS.get(regime, 'gray')
            label = REGIME_LABELS.get(regime, regime)
            by_seed: dict[int, list[tuple[int, float, float, float]]] = {}
            for r in recs:
                if r.get('generation', 0) != 0:
                    continue
                ts = r.get('timesteps', 0)
                stats = r.get('stats', {})
                mode_stats = stats.get(mode_key) or stats.get(f'{mode_key}_after')
                if mode_stats is None:
                    continue
                seed = r.get('seed', 0)
                by_seed.setdefault(seed, []).append((ts, mode_stats['mean'], mode_stats['ci_mean_lo'], mode_stats['ci_mean_hi']))
            if not by_seed:
                continue
            all_ts_set: set[int] = set()
            for seed, pts in by_seed.items():
                pts_sorted = sorted(pts, key=lambda x: x[0])
                ts_arr = np.array([p[0] for p in pts_sorted])
                m_arr = np.array([p[1] for p in pts_sorted])
                all_ts_set.update(ts_arr.tolist())
                ax.plot(ts_arr, m_arr, color=color, alpha=0.25, linewidth=1)
            all_ts = sorted(all_ts_set)
            means_by_ts: dict[int, list[float]] = {ts: [] for ts in all_ts}
            lo_by_ts: dict[int, list[float]] = {ts: [] for ts in all_ts}
            hi_by_ts: dict[int, list[float]] = {ts: [] for ts in all_ts}
            for seed, pts in by_seed.items():
                for ts, m, lo, hi in pts:
                    if ts in means_by_ts:
                        means_by_ts[ts].append(m)
                        lo_by_ts[ts].append(lo)
                        hi_by_ts[ts].append(hi)
            ts_arr = np.array([t for t in all_ts if means_by_ts[t]])
            mean_arr = np.array([np.mean(means_by_ts[t]) for t in ts_arr])
            lo_arr = np.array([np.mean(lo_by_ts[t]) for t in ts_arr])
            hi_arr = np.array([np.mean(hi_by_ts[t]) for t in ts_arr])
            ax.plot(ts_arr, mean_arr, color=color, linewidth=2.5, label=label)
            ax.fill_between(ts_arr, lo_arr, hi_arr, color=color, alpha=0.15)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1000000.0:.1f}M'))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[figure 1] saved → {out_path}')

def plot_tail_curves(records: list[dict], out_path: Path) -> None:
    if not HAS_MPL:
        return
    grouped = group_by_regime(records)
    regime_order = ['A', 'B', 'C', 'IBR']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tail Performance: p95 and CVaR(0.1) — lower is better', fontsize=14, fontweight='bold')
    for ax, (mode_key, stat_key, y_label) in zip(axes, [('SPREAD', 'p95', 'p95 of τ under SPREAD'), ('SPREAD', 'cvar_10', 'CVaR(0.1) of τ under SPREAD')]):
        ax.set_title(f'{mode_key} — {stat_key}', fontsize=12)
        ax.set_ylabel(y_label)
        x_positions = np.arange(len(regime_order))
        means, errs_lo, errs_hi = ([], [], [])
        for regime in regime_order:
            recs = grouped.get(regime, [])
            final_recs = [r for r in recs if r.get('generation', 0) == 0 or r.get('generation') == max((r2.get('generation', 0) for r2 in recs), default=0)]
            vals = []
            for r in final_recs:
                stats = r.get('stats', {})
                ms = stats.get(mode_key) or stats.get(f'{mode_key}_after')
                if ms:
                    vals.append(ms.get(stat_key, 0.0))
            if vals:
                means.append(float(np.mean(vals)))
                err = float(np.std(vals))
                errs_lo.append(err)
                errs_hi.append(err)
            else:
                means.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
        colors_ordered = [REGIME_COLORS.get(r, 'gray') for r in regime_order]
        bars = ax.bar(x_positions, means, color=colors_ordered, alpha=0.85, yerr=[errs_lo, errs_hi], capsize=5, ecolor='black')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([REGIME_LABELS.get(r, r) for r in regime_order], fontsize=8, rotation=15, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[figure 2] saved → {out_path}')

def plot_ibr_generation(records: list[dict], out_path: Path) -> None:
    if not HAS_MPL:
        return
    ibr_recs = [r for r in records if r.get('regime') == 'IBR' and r.get('generation', 0) > 0]
    if not ibr_recs:
        print('[figure 3] No IBR generation records found — skipping')
        return
    by_seed: dict[int, list[dict]] = {}
    for r in ibr_recs:
        by_seed.setdefault(r['seed'], []).append(r)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('IBR Generation Diagnostics — 3 proof conditions', fontsize=12, fontweight='bold')
    ax.set_xlabel('IBR Generation k')
    ax.set_ylabel('Δ shots-to-win')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    metric_style = {'exploitability_defender': dict(color='#E15759', label='defender_adversarial [check A: must > 0]'), 'exploitability_attacker': dict(color='#4E79A7', label='attacker_adaptation [check B: must < 0]'), 'uniform_drift': dict(color='#F28E2B', label='uniform_drift [check C: must ≈ 0]')}
    all_gens_by_metric: dict[str, dict[int, list[float]]] = {m: {} for m in metric_style}
    for seed, recs in by_seed.items():
        recs_sorted = sorted(recs, key=lambda r: r['generation'])
        for metric, style in metric_style.items():
            xs = [r['generation'] for r in recs_sorted if r.get(metric) is not None]
            ys = [r[metric] for r in recs_sorted if r.get(metric) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, color=style['color'], alpha=0.3, linewidth=1.2)
            for x, y in zip(xs, ys):
                all_gens_by_metric[metric].setdefault(x, []).append(y)
    for metric, style in metric_style.items():
        gen_data = all_gens_by_metric[metric]
        if not gen_data:
            continue
        xs = sorted(gen_data.keys())
        ys = [np.mean(gen_data[x]) for x in xs]
        lo = [np.mean(gen_data[x]) - np.std(gen_data[x]) for x in xs]
        hi = [np.mean(gen_data[x]) + np.std(gen_data[x]) for x in xs]
        ax.plot(xs, ys, color=style['color'], linewidth=2.5, marker='o', label=style['label'])
        ax.fill_between(xs, lo, hi, color=style['color'], alpha=0.12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[figure 3] saved → {out_path}')

def print_summary_table(records: list[dict]) -> None:
    grouped = group_by_regime(records)
    header = f'{'Regime':>6}  {'Seed':>5}  {'UNIF_mean':>10}  {'UNIF_p90':>9}  {'SPREAD_mean':>12}  {'SPREAD_p90':>10}  {'Dk_mean':>8}  {'Dk_p90':>7}  {'worst_Dk':>9}  {'rob_gap':>8}  {'fail':>6}'
    print('\n' + '=' * len(header))
    print('TABLE 1: Final Evaluation Summary')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for regime in ['A', 'B', 'C', 'IBR']:
        recs = grouped.get(regime, [])
        final_recs = [r for r in recs if r.get('generation', 0) == 0 or r.get('generation') == max((r2.get('generation', 0) for r2 in recs), default=0)]
        per_seed_rows = {}
        for r in final_recs:
            seed = r['seed']
            stats = r.get('stats', {})

            def get_stat(key: str, field: str) -> float:
                ms = stats.get(key) or stats.get(f'{key}_after') or {}
                return float(ms.get(field, 0.0))
            per_seed_rows[seed] = {'u_mean': get_stat('UNIFORM', 'mean'), 'u_p90': get_stat('UNIFORM', 'p90'), 'sp_mean': get_stat('SPREAD', 'mean'), 'sp_p90': get_stat('SPREAD', 'p90'), 'dk_mean': get_stat('D_k_after', 'mean'), 'dk_p90': get_stat('D_k_after', 'p90'), 'worst_dk': r.get('worst_D_k_mean') or 0.0, 'rob_gap': r.get('robust_gap', 0.0), 'fail': get_stat('UNIFORM', 'fail_rate')}
        for seed, row in sorted(per_seed_rows.items()):
            print(f'  {regime:>6}  {seed:>5}  {row['u_mean']:>10.1f}  {row['u_p90']:>9.1f}  {row['sp_mean']:>12.1f}  {row['sp_p90']:>10.1f}  {row['dk_mean']:>8.1f}  {row['dk_p90']:>7.1f}  {row['worst_dk']:>9.1f}  {row['rob_gap']:>8.2f}  {row['fail']:>6.4f}')
        if len(per_seed_rows) > 1:
            vals = list(per_seed_rows.values())
            mn = lambda k: float(np.mean([v[k] for v in vals]))
            sd = lambda k: float(np.std([v[k] for v in vals]))
            print(f'  {regime:>6}  {'μ±σ':>5}  {mn('u_mean'):>10.1f}  {mn('u_p90'):>9.1f}  {mn('sp_mean'):>12.1f}  {mn('sp_p90'):>10.1f}  {mn('dk_mean'):>8.1f}  {mn('dk_p90'):>7.1f}  {mn('worst_dk'):>9.1f}  {mn('rob_gap'):>8.2f}  {mn('fail'):>6.4f}    ± [{sd('sp_mean'):.1f} SPREAD  {sd('u_mean'):.1f} UNIFORM]')
        print()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, default='results/experiments')
    parser.add_argument('--out_dir', type=str, default='results/figures')
    parser.add_argument('--table_only', action='store_true')
    args = parser.parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(results_root)
    if not records:
        print(f'[ERROR] No eval_log.jsonl records found under {results_root}')
        return
    print(f'Loaded {len(records)} eval records from {results_root}')
    print_summary_table(records)
    if not args.table_only:
        plot_learning_curves(records, out_dir / 'fig1_learning_curves.pdf')
        plot_tail_curves(records, out_dir / 'fig2_tail_curves.pdf')
        plot_ibr_generation(records, out_dir / 'fig3_ibr_generation.pdf')
if __name__ == '__main__':
    main()