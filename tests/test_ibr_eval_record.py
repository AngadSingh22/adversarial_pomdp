""""""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from battleship_rl.eval.schema import DistributionStats, EvalRecord
REQUIRED_DIST_FIELDS = {'mean', 'std', 'p90', 'p95', 'cvar_10', 'fail_rate', 'n_episodes', 'ci_mean_lo', 'ci_mean_hi', 'ci_p90_lo', 'ci_p90_hi'}

def _make_real_dist_stats() -> DistributionStats:
    """"""
    from battleship_rl.eval.eval_lib import _distribution_stats
    lengths = [85, 90, 92, 88, 95, 91, 89, 93, 87, 94]
    return _distribution_stats(lengths, trunc_count=0, total_eps=10, n_bootstrap=50)

class TestDistributionStatsNonZero:
    """"""

    def test_std_is_nonzero(self):
        ds = _make_real_dist_stats()
        assert ds.std > 0.0, f'std must be > 0 for non-constant data, got {ds.std}'

    def test_p95_is_nonzero(self):
        ds = _make_real_dist_stats()
        assert ds.p95 > 0.0, f'p95 must be > 0, got {ds.p95}'

    def test_n_episodes_correct(self):
        ds = _make_real_dist_stats()
        assert ds.n_episodes == 10

    def test_ci_range_is_positive_width(self):
        ds = _make_real_dist_stats()
        assert ds.ci_mean_hi > ds.ci_mean_lo, 'CI must have positive width'

    def test_all_required_fields_present(self):
        ds = _make_real_dist_stats()
        d = vars(ds)
        missing = REQUIRED_DIST_FIELDS - set(d.keys())
        assert not missing, f'DistributionStats is missing fields: {missing}'

    def test_no_placeholder_zeros_in_non_constant_data(self):
        """"""
        ds = _make_real_dist_stats()
        assert ds.std != 0.0, 'std must not be placeholder 0.0'
        assert ds.cvar_10 != 0.0, 'cvar_10 must not be placeholder 0.0'
        assert ds.n_episodes != 0, 'n_episodes must not be placeholder 0'

class TestEvalRecordStatsSchema:
    """"""

    def test_ibr_stats_keys_contain_mode_before_after(self):
        """"""
        ds = _make_real_dist_stats()
        stats = {'UNIFORM_before': vars(ds), 'SPREAD_before': vars(ds), 'D_k_before': vars(ds), 'UNIFORM_after': vars(ds), 'SPREAD_after': vars(ds), 'D_k_after': vars(ds)}
        for key, d in stats.items():
            missing = REQUIRED_DIST_FIELDS - set(d.keys())
            assert not missing, f"stats['{key}'] missing fields: {missing}"

    def test_ibr_stats_no_stub_zeros(self):
        """"""
        ds = _make_real_dist_stats()
        d = vars(ds)
        is_stub = d['std'] == 0.0 and d['n_episodes'] == 0
        assert not is_stub, 'EvalRecord stat entry looks like a placeholder stub'

class TestEvalRecordJSONL:
    """"""

    def test_roundtrip(self, tmp_path):
        from battleship_rl.eval import append_eval_record
        ds = _make_real_dist_stats()
        record = EvalRecord(regime='IBR', seed=42, timesteps=1000000, generation=1, git_hash='test', timestamp='2026-01-01T00:00:00', cli_args={}, stats={'UNIFORM_before': vars(ds), 'UNIFORM_after': vars(ds), 'D_k_after': vars(ds)}, robust_gap=1.5, robust_gap_p95=2.0, exploitability_defender=0.5, exploitability_attacker=-1.2, uniform_drift=-0.3, worst_D_k_mean=None, defender_budget=50000, policy={'action_entropy': 3.5, 'adjacency_ratio': 0.6, 'time_to_first_hit': 12.3, 'time_to_first_sink': 45.0, 'invalid_action_rate': 0.0}, defender_shift=None)
        out = tmp_path / 'eval_log.jsonl'
        append_eval_record(record, out)
        row = json.loads(out.read_text().strip().split('\n')[0])
        for mode_key, d in row['stats'].items():
            missing = REQUIRED_DIST_FIELDS - set(d.keys())
            assert not missing, f"Roundtripped stats['{mode_key}'] missing: {missing}"
            assert d['n_episodes'] > 0, f"stats['{mode_key}']['n_episodes'] is 0 — looks like a stub"
            assert d['std'] > 0.0, f"stats['{mode_key}']['std'] is 0.0 — placeholder?"
        assert row['exploitability_defender'] == pytest.approx(0.5)
        assert row['exploitability_attacker'] == pytest.approx(-1.2)
        assert row['policy']['time_to_first_hit'] == pytest.approx(12.3)