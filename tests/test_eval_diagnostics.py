""""""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from battleship_rl.eval.eval_lib import _normalize_outcome

class TestNormalizeOutcome:
    """"""

    def test_hit(self):
        out = _normalize_outcome({'outcome_type': 'HIT'})
        assert out['is_hit'] is True
        assert out['is_sink'] is False
        assert out['is_invalid'] is False

    def test_sunk(self):
        out = _normalize_outcome({'outcome_type': 'SUNK'})
        assert out['is_hit'] is True
        assert out['is_sink'] is True
        assert out['is_invalid'] is False

    def test_miss(self):
        out = _normalize_outcome({'outcome_type': 'MISS'})
        assert out['is_hit'] is False
        assert out['is_sink'] is False
        assert out['is_invalid'] is False

    def test_invalid(self):
        out = _normalize_outcome({'outcome_type': 'INVALID'})
        assert out['is_hit'] is False
        assert out['is_sink'] is False
        assert out['is_invalid'] is True

    def test_case_insensitive(self):
        """"""
        out = _normalize_outcome({'outcome_type': 'sunk'})
        assert out['is_hit'] is True
        assert out['is_sink'] is True

    def test_none_value(self):
        """"""
        out = _normalize_outcome({'outcome_type': None})
        assert out == {'is_hit': False, 'is_sink': False, 'is_invalid': False}

class TestNormalizeOutcomeLegacy:
    """"""

    def test_hit(self):
        out = _normalize_outcome({'outcome': 'hit'})
        assert out['is_hit'] is True
        assert out['is_sink'] is False

    def test_sunk(self):
        out = _normalize_outcome({'outcome': 'sunk'})
        assert out['is_hit'] is True
        assert out['is_sink'] is True

    def test_miss(self):
        out = _normalize_outcome({'outcome': 'miss'})
        assert out['is_hit'] is False

    def test_missing_key(self):
        """"""
        out = _normalize_outcome({})
        assert out == {'is_hit': False, 'is_sink': False, 'is_invalid': False}

class TestNormalizeOutcomePriority:
    """"""

    def test_battleship_wins(self):
        out = _normalize_outcome({'outcome_type': 'MISS', 'outcome': 'hit'})
        assert out['is_hit'] is False

    def test_battleship_wins_hit(self):
        out = _normalize_outcome({'outcome_type': 'HIT', 'outcome': 'miss'})
        assert out['is_hit'] is True

class TestHitGridUpdatesCorrectly:
    """"""

    def test_hit_updates_grid_and_first_hit(self):
        board_h, board_w = (10, 10)
        hit_grid = np.zeros((board_h, board_w), dtype=bool)
        first_hit = None
        steps = 1
        action = 3 * board_w + 4
        info = {'outcome_type': 'HIT'}
        outcome = _normalize_outcome(info)
        if outcome['is_hit']:
            r_, c_ = divmod(action, board_w)
            hit_grid[r_, c_] = True
            if first_hit is None:
                first_hit = steps
        assert hit_grid[3, 4] is np.bool_(True), 'hit_grid should be updated on HIT'
        assert first_hit == 1, 'first_hit should be set on the first HIT step'

    def test_miss_does_not_update_grid(self):
        board_h, board_w = (10, 10)
        hit_grid = np.zeros((board_h, board_w), dtype=bool)
        first_hit = None
        steps = 1
        action = 7 * board_w + 2
        info = {'outcome_type': 'MISS'}
        outcome = _normalize_outcome(info)
        if outcome['is_hit']:
            r_, c_ = divmod(action, board_w)
            hit_grid[r_, c_] = True
            if first_hit is None:
                first_hit = steps
        assert not hit_grid.any(), 'hit_grid must not be updated on MISS'
        assert first_hit is None, 'first_hit must remain None on MISS'

    def test_old_key_would_fail(self):
        """"""
        info = {'outcome_type': 'HIT'}
        old_outcome = info.get('outcome', None)
        old_is_hit = old_outcome in ('hit', 'sunk')
        assert old_is_hit is False, "REGRESSION GUARD: old code could not detect HIT from 'outcome_type' key"
        new_outcome = _normalize_outcome(info)
        assert new_outcome['is_hit'] is True