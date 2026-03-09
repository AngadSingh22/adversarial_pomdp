""""""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Optional

@dataclasses.dataclass
class DistributionStats:
    """"""
    mean: float
    std: float
    p90: float
    p95: float
    cvar_10: float
    fail_rate: float
    n_episodes: int
    ci_mean_lo: float
    ci_mean_hi: float
    ci_p90_lo: float
    ci_p90_hi: float
    trunc_reason: dict

@dataclasses.dataclass
class PolicyDiagnostics:
    """"""
    action_entropy: float
    adjacency_ratio: float
    time_to_first_hit: float
    time_to_first_sink: float
    invalid_action_rate: float

@dataclasses.dataclass
class DefenderShiftMetrics:
    """"""
    centroid_pairwise_mean: float
    centroid_pairwise_p95: float
    cluster_score: float
    marginal_entropy: float
    n_layouts: int

@dataclasses.dataclass
class EvalRecord:
    """"""
    regime: str
    seed: int
    timesteps: int
    generation: int
    git_hash: str
    timestamp: str
    cli_args: dict
    stats: dict
    robust_gap: float
    robust_gap_p95: float
    exploitability_defender: Optional[float]
    exploitability_attacker: Optional[float]
    uniform_drift: Optional[float]
    worst_D_k_mean: Optional[float]
    defender_budget: Optional[int]
    policy: dict
    defender_shift: Optional[dict]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

def append_eval_record(record: EvalRecord, path: Path) -> None:
    """"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(record.to_json() + '\n')