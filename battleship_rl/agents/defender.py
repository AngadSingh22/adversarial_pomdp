""""""
from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np
from battleship_rl.envs.placement import _normalize_board_size, _normalize_ships, decode_placement_action, sample_placement

def _valid_mask_h(occupied: np.ndarray, length: int) -> np.ndarray:
    """"""
    H, W = occupied.shape
    if W < length:
        return np.zeros((H, 0), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(occupied, length, axis=1)
    return ~windows.any(axis=2)

def _valid_mask_v(occupied: np.ndarray, length: int) -> np.ndarray:
    """"""
    H, W = occupied.shape
    if H < length:
        return np.zeros((0, W), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(occupied, length, axis=0)
    return ~windows.any(axis=2)

def _propose(rng: np.random.Generator, occupied: np.ndarray, length: int, K: int=200, score_fn=None, greedy: bool=False) -> tuple[int, int, int]:
    """"""
    H, W = occupied.shape
    vh = _valid_mask_h(occupied, length)
    vv = _valid_mask_v(occupied, length)
    rsh, csh = np.nonzero(vh)
    rsv, csv = np.nonzero(vv)
    rs_all = np.concatenate([rsh, rsv])
    cs_all = np.concatenate([csh, csv])
    ors_all = np.concatenate([np.zeros(len(rsh), dtype=np.int8), np.ones(len(rsv), dtype=np.int8)])
    n_total = len(rs_all)
    if n_total == 0:
        raise ValueError('No legal placements available.')
    if n_total <= K:
        idx_pool = np.arange(n_total)
    else:
        idx_pool = rng.choice(n_total, size=K, replace=False)
    rs = rs_all[idx_pool]
    cs = cs_all[idx_pool]
    ors = ors_all[idx_pool]
    if score_fn is not None:
        w = score_fn(rs, cs, ors, occupied, H, W, length)
        if greedy:
            chosen = int(np.argmax(w))
            return (int(rs[chosen]), int(cs[chosen]), int(ors[chosen]))
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total > 0:
            p = w / total
        else:
            p = None
    else:
        p = None
    chosen = rng.choice(len(rs), p=p)
    return (int(rs[chosen]), int(cs[chosen]), int(ors[chosen]))

def _apply(ship_id_grid: np.ndarray, r: int, c: int, orient: int, length: int, ship_id: int) -> None:
    """"""
    if orient == 0:
        ship_id_grid[r, c:c + length] = ship_id
    else:
        ship_id_grid[r:r + length, c] = ship_id

def _occupied_from_grid(ship_id_grid: np.ndarray) -> np.ndarray:
    return ship_id_grid >= 0

def _edge_weights(H: int, W: int) -> np.ndarray:
    """"""
    rs, cs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dist = np.minimum(np.minimum(rs, H - 1 - rs), np.minimum(cs, W - 1 - cs))
    return (1.0 / (dist + 1)).astype(np.float32)

def _placement_center(rs: np.ndarray, cs: np.ndarray, ors: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """"""
    half = (length - 1) / 2.0
    r_c = np.where(ors == 0, rs.astype(float), rs + half)
    c_c = np.where(ors == 0, cs + half, cs.astype(float))
    return (r_c, c_c)

class BaseDefender:

    def sample_layout(self, board_size: int | Sequence[int], ships: Sequence[int] | dict, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

class UniformRandomDefender(BaseDefender):
    """"""

    def sample_layout(self, board_size, ships, rng):
        return sample_placement(board_size, ships, rng)

class EdgeBiasedDefender(BaseDefender):
    """"""
    _WEIGHT_CACHE: dict[tuple, np.ndarray] = {}

    def _weight_grid(self, H: int, W: int) -> np.ndarray:
        key = (H, W)
        if key not in self._WEIGHT_CACHE:
            self._WEIGHT_CACHE[key] = _edge_weights(H, W)
        return self._WEIGHT_CACHE[key]

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        wgrid = self._weight_grid(H, W)
        for ship_id, length in enumerate(ship_lengths):

            def score(rs, cs, ors, occ, H, W, L, _wg=wgrid, _L=length):
                r_c, c_c = _placement_center(rs, cs, ors, _L)
                r_i = np.clip(r_c.astype(int), 0, H - 1)
                c_i = np.clip(c_c.astype(int), 0, W - 1)
                return _wg[r_i, c_i]
            r, c, o = _propose(rng, _occupied_from_grid(ship_id_grid), length, score_fn=score)
            _apply(ship_id_grid, r, c, o, length, ship_id)
        return ship_id_grid
BiasedDefender = EdgeBiasedDefender

class ClusteredDefender(BaseDefender):
    """"""

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        center_r, center_c = (H / 2.0, W / 2.0)
        for ship_id, length in enumerate(ship_lengths):
            occupied = _occupied_from_grid(ship_id_grid)
            placed = np.argwhere(occupied)
            if len(placed) == 0:

                def score_center(rs, cs, ors, occ, H, W, L, cr=center_r, cc=center_c):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    dist = np.hypot(r_c - cr, c_c - cc)
                    return np.exp(-1.0 * dist)
                r, c, o = _propose(rng, occupied, length, score_fn=score_center)
            else:
                centroid_r = placed[:, 0].mean()
                centroid_c = placed[:, 1].mean()

                def score_cluster(rs, cs, ors, occ, H, W, L, cr=centroid_r, cc=centroid_c):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    dist = np.hypot(r_c - cr, c_c - cc)
                    return np.exp(-2.0 * dist)
                r, c, o = _propose(rng, occupied, length, score_fn=score_cluster)
            _apply(ship_id_grid, r, c, o, length, ship_id)
        return ship_id_grid

class SpreadDefender(BaseDefender):
    """"""

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        rows_idx, cols_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        for ship_id, length in enumerate(ship_lengths):
            occupied = _occupied_from_grid(ship_id_grid)
            placed = np.argwhere(occupied)
            if len(placed) == 0:
                r, c, o = _propose(rng, occupied, length, score_fn=None)
            else:
                dr = (rows_idx[None, :, :] - placed[:, 0:1, None]).astype(np.float32)
                dc = (cols_idx[None, :, :] - placed[:, 1:2, None]).astype(np.float32)
                dist_map = np.hypot(dr, dc).min(axis=0)

                def score_spread(rs, cs, ors, occ, H, W, L, dm=dist_map):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    r_i = np.clip(r_c.astype(int), 0, H - 1)
                    c_i = np.clip(c_c.astype(int), 0, W - 1)
                    return dm[r_i, c_i]
                r, c, o = _propose(rng, occupied, length, score_fn=score_spread, greedy=True)
            _apply(ship_id_grid, r, c, o, length, ship_id)
        return ship_id_grid

class ParityDefender(BaseDefender):
    """"""

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        qH, qW = (H // 2, W // 2)
        for ship_id, length in enumerate(ship_lengths):
            occupied = _occupied_from_grid(ship_id_grid)
            if ship_id == 0:

                def score(rs, cs, ors, occ, H, W, L, qH=qH, qW=qW):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    in_quad = (r_c < qH - 0.5) & (c_c < qW - 0.5)
                    return in_quad.astype(float)
                r, c, o = _propose(rng, occupied, length, score_fn=score)
            else:
                r, c, o = _propose(rng, occupied, length, score_fn=None)
            _apply(ship_id_grid, r, c, o, length, ship_id)
        return ship_id_grid

class AdversarialDefender(BaseDefender):
    """"""

    def __init__(self, model_path: str | None=None, deterministic: bool=False):
        self.deterministic = deterministic
        self.model = None
        if model_path:
            try:
                from sb3_contrib import MaskablePPO
                self.model = MaskablePPO.load(model_path)
            except ImportError:
                print('Warning: sb3-contrib not installed or model load failed.')
            except Exception as e:
                print(f'Warning: Failed to load adversarial model: {e}')

    def sample_layout(self, board_size, ships, rng):
        if self.model is None:
            return sample_placement(board_size, ships, rng)
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        board = np.full((H, W), -1, dtype=np.int32)
        for ship_idx, length in enumerate(ship_lengths):
            obs = np.zeros((3, H, W), dtype=np.float32)
            obs[0] = (board != -1).astype(np.float32)
            obs[1] = float(length) / max(ship_lengths)
            obs[2] = float(ship_idx) / len(ship_lengths)
            mask = np.zeros(H * W * 2, dtype=bool)
            for r in range(H):
                for c in range(W):
                    if c + length <= W and np.all(board[r, c:c + length] == -1):
                        mask[r * W + c] = True
                    if r + length <= H and np.all(board[r:r + length, c] == -1):
                        mask[H * W + r * W + c] = True
            if not np.any(mask):
                return sample_placement(board_size, ships, rng)
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
            r, c, orientation = decode_placement_action(int(action), H, W)
            if orientation == 0:
                board[r, c:c + length] = ship_idx
            else:
                board[r:r + length, c] = ship_idx
        return board