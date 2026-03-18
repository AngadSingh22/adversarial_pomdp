from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "formulation" / "paper" / "figures"
BOARD_SIZE = (10, 10)
SHIP_LENGTHS = [5, 4, 3, 3, 2]
DEFENDER_NAMES = ["UNIFORM", "EDGE", "CLUSTER", "SPREAD", "PARITY"]
EARLY_SHOT_WINDOW = 5

THEME = {
    "ink": "#25313D",
    "slate": "#667481",
    "fog": "#EEF1F4",
    "mist": "#D8E0E8",
    "blue": "#416B8A",
    "blue_light": "#A8BED0",
    "rust": "#B8644A",
    "rust_light": "#E2B7A8",
    "ochre": "#C28B3B",
    "sand": "#E8D9C7",
    "line": "#CBD4DC",
    "white": "#FFFFFF",
}

REGIME_STYLE = {
    "A": {"color": THEME["slate"], "marker": "o", "label": "nominal-only baseline"},
    "B": {"color": THEME["blue"], "marker": "s", "label": "balanced robustness point"},
    "C": {"color": THEME["rust"], "marker": "^", "label": "stress-specialized point"},
}

SEED_STYLE = {
    42: {"color": THEME["blue"]},
    123: {"color": THEME["ochre"]},
    777: {"color": THEME["rust"]},
}

STAGE1_ROWS = {
    "A": {"uniform_mean": 90.00, "uniform_sd": 1.72, "spread_mean": 100.33, "spread_sd": 1.18},
    "B": {"uniform_mean": 91.33, "uniform_sd": 0.62, "spread_mean": 94.47, "spread_sd": 0.62},
    "C": {"uniform_mean": 93.33, "uniform_sd": 2.04, "spread_mean": 84.44, "spread_sd": 1.62},
}

IBR_ROWS = [
    {"seed": 42, "generation": 1, "defender_adversarial": -2.15, "attacker_adaptation": -1.37, "uniform_drift": -0.96},
    {"seed": 42, "generation": 2, "defender_adversarial": -0.14, "attacker_adaptation": -0.45, "uniform_drift": 0.06},
    {"seed": 42, "generation": 3, "defender_adversarial": 0.26, "attacker_adaptation": 0.73, "uniform_drift": 0.18},
    {"seed": 123, "generation": 1, "defender_adversarial": 1.47, "attacker_adaptation": 1.49, "uniform_drift": -0.31},
    {"seed": 123, "generation": 2, "defender_adversarial": 1.93, "attacker_adaptation": 1.17, "uniform_drift": 0.35},
    {"seed": 123, "generation": 3, "defender_adversarial": -1.35, "attacker_adaptation": -0.23, "uniform_drift": 0.53},
    {"seed": 777, "generation": 1, "defender_adversarial": -1.93, "attacker_adaptation": -0.50, "uniform_drift": -1.04},
    {"seed": 777, "generation": 2, "defender_adversarial": 0.18, "attacker_adaptation": 0.05, "uniform_drift": -0.13},
    {"seed": 777, "generation": 3, "defender_adversarial": 3.19, "attacker_adaptation": 3.38, "uniform_drift": -0.88},
]

GEOMETRY_ROWS = {
    "UNIFORM": {"centroid": 0.000, "cluster": 18.05, "entropy": 0.451, "asymmetry": 0.012},
    "EDGE": {"centroid": 1.866, "cluster": 18.04, "entropy": 0.449, "asymmetry": 0.013},
    "CLUSTER": {"centroid": 2.266, "cluster": 21.55, "entropy": 0.399, "asymmetry": 0.016},
    "SPREAD": {"centroid": 2.947, "cluster": 15.97, "entropy": 0.416, "asymmetry": 0.026},
    "PARITY": {"centroid": 1.502, "cluster": 18.50, "entropy": 0.434, "asymmetry": 0.034},
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.edgecolor": THEME["ink"],
            "axes.labelcolor": THEME["ink"],
            "axes.linewidth": 1.0,
            "xtick.color": THEME["slate"],
            "ytick.color": THEME["slate"],
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": THEME["line"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "figure.facecolor": THEME["white"],
            "savefig.facecolor": THEME["white"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_repo_modules():
    package_roots = {
        "battleship_rl": REPO_ROOT / "battleship_rl",
        "battleship_rl.envs": REPO_ROOT / "battleship_rl" / "envs",
        "battleship_rl.agents": REPO_ROOT / "battleship_rl" / "agents",
        "battleship_rl.baselines": REPO_ROOT / "battleship_rl" / "baselines",
    }
    for name, path in package_roots.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    def load_module(name: str, path: Path):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    placement = load_module("battleship_rl.envs.placement", REPO_ROOT / "battleship_rl" / "envs" / "placement.py")
    defender = load_module("battleship_rl.agents.defender", REPO_ROOT / "battleship_rl" / "agents" / "defender.py")
    return placement, defender, None


def build_defender_map():
    _, defender_module, _ = ensure_repo_modules()
    defender_map = {
        "UNIFORM": defender_module.UniformRandomDefender,
        "EDGE": defender_module.EdgeBiasedDefender,
        "CLUSTER": defender_module.ClusteredDefender,
        "SPREAD": defender_module.SpreadDefender,
        "PARITY": defender_module.ParityDefender,
    }
    return defender_map


def sample_layouts(defender_cls, n_layouts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    defender = defender_cls()
    layouts = []
    for _ in range(n_layouts):
        layout = defender.sample_layout(board_size=BOARD_SIZE, ships=SHIP_LENGTHS, rng=rng)
        layouts.append(layout)
    return np.asarray(layouts, dtype=np.int32)


def ship_cell_lookup(layout: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    return {int(ship_id): {tuple(cell) for cell in np.argwhere(layout == ship_id)} for ship_id in np.unique(layout[layout >= 0])}


def layout_consistent(layout: np.ndarray, hits: np.ndarray, misses: np.ndarray, sunk_ships: set[int]) -> bool:
    occupied = layout >= 0
    if np.any(hits & ~occupied):
        return False
    if np.any(misses & occupied):
        return False
    for ship_id in sunk_ships:
        ship_mask = layout == ship_id
        if not np.any(ship_mask):
            return False
        if np.any(ship_mask & ~hits):
            return False
    return True


def replenish_particles(defender, rng: np.random.Generator, hits: np.ndarray, misses: np.ndarray, sunk_ships: set[int], target_count: int) -> list[np.ndarray]:
    particles: list[np.ndarray] = []
    max_trials = max(target_count * 16, 160)
    for _ in range(max_trials):
        layout = defender.sample_layout(board_size=BOARD_SIZE, ships=SHIP_LENGTHS, rng=rng)
        if layout_consistent(layout, hits, misses, sunk_ships):
            particles.append(layout)
            if len(particles) >= target_count:
                break
    return particles


def simulate_posterior_search(defender_cls, rng: np.random.Generator, horizon: int, particle_count: int = 90) -> list[tuple[int, int]]:
    defender = defender_cls()
    hidden_layout = defender.sample_layout(board_size=BOARD_SIZE, ships=SHIP_LENGTHS, rng=rng)
    hidden_ship_cells = ship_cell_lookup(hidden_layout)
    hidden_ship_sizes = {ship_id: len(cells) for ship_id, cells in hidden_ship_cells.items()}
    hidden_ship_hits = {ship_id: 0 for ship_id in hidden_ship_cells}
    hits = np.zeros(hidden_layout.shape, dtype=bool)
    misses = np.zeros(hidden_layout.shape, dtype=bool)
    sunk_ships: set[int] = set()
    particles = replenish_particles(defender, rng, hits, misses, sunk_ships, target_count=particle_count)
    actions: list[tuple[int, int]] = []

    while len(actions) < horizon:
        if len(particles) < particle_count // 3:
            particles.extend(replenish_particles(defender, rng, hits, misses, sunk_ships, target_count=particle_count - len(particles)))
        if particles:
            particle_array = np.asarray(particles, dtype=np.int32)
            marginal = (particle_array >= 0).mean(axis=0)
        else:
            marginal = np.ones(hidden_layout.shape, dtype=np.float64)
        marginal[hits | misses] = -1.0
        best_cells = np.argwhere(marginal == np.max(marginal))
        if best_cells.size == 0:
            best_cells = np.argwhere(~(hits | misses))
        choice = best_cells[int(rng.integers(0, len(best_cells)))]
        row, col = int(choice[0]), int(choice[1])
        actions.append((row, col))

        if hidden_layout[row, col] >= 0:
            ship_id = int(hidden_layout[row, col])
            hits[row, col] = True
            hidden_ship_hits[ship_id] += 1
            if hidden_ship_hits[ship_id] == hidden_ship_sizes[ship_id]:
                sunk_ships.add(ship_id)
        else:
            misses[row, col] = True

        if np.all((hidden_layout < 0) | hits):
            break
        particles = [layout for layout in particles if layout_consistent(layout, hits, misses, sunk_ships)]
    return actions


def compute_heatmap_inputs(layout_samples: int, behavior_episodes: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    defender_map = build_defender_map()
    occupancy_maps: dict[str, np.ndarray] = {}
    behavior_maps: dict[str, np.ndarray] = {}
    for index, name in enumerate(DEFENDER_NAMES):
        layouts = sample_layouts(defender_map[name], layout_samples, seed=200 + index * 17)
        occupancy_maps[name] = (layouts >= 0).mean(axis=0)

        behavior_counts = np.zeros(BOARD_SIZE, dtype=np.float64)
        rng = np.random.default_rng(700 + index * 31)
        for episode_idx in range(behavior_episodes):
            for row, col in simulate_posterior_search(defender_map[name], rng=rng, horizon=EARLY_SHOT_WINDOW):
                behavior_counts[row, col] += 1.0
        behavior_maps[name] = behavior_counts / (behavior_episodes * EARLY_SHOT_WINDOW)
    return occupancy_maps, behavior_maps


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(THEME["white"])


def plot_stage1_pareto(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    xmin = min(row["uniform_mean"] - row["uniform_sd"] for row in STAGE1_ROWS.values()) - 0.8
    xmax = max(row["uniform_mean"] + row["uniform_sd"] for row in STAGE1_ROWS.values()) + 0.8
    ymin = min(row["spread_mean"] - row["spread_sd"] for row in STAGE1_ROWS.values()) - 1.0
    ymax = max(row["spread_mean"] + row["spread_sd"] for row in STAGE1_ROWS.values()) + 1.0

    lower_left = np.array(
        [
            [xmin, ymin],
            [xmin, ymin + (ymax - ymin) * 0.35],
            [xmin + (xmax - xmin) * 0.35, ymin],
        ]
    )
    ax.fill(lower_left[:, 0], lower_left[:, 1], color=THEME["fog"], zorder=0)

    for regime, row in STAGE1_ROWS.items():
        style = REGIME_STYLE[regime]
        x = row["uniform_mean"]
        y = row["spread_mean"]
        ax.errorbar(
            x,
            y,
            xerr=row["uniform_sd"],
            yerr=row["spread_sd"],
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.1,
            alpha=0.45,
            capsize=0,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=170,
            marker=style["marker"],
            color=style["color"],
            edgecolor=THEME["white"],
            linewidth=1.4,
            zorder=3,
        )
        label_dx = {"A": 0.18, "B": 0.16, "C": 0.18}[regime]
        label_dy = {"A": 1.15, "B": 1.10, "C": -1.45}[regime]
        ax.text(
            x + label_dx,
            y + label_dy,
            f"{regime}: {style['label']}",
            color=THEME["ink"],
            fontsize=10,
            weight="bold" if regime != "A" else None,
        )

    ax.annotate(
        "better",
        xy=(xmin + 0.2, ymin + 0.2),
        xytext=(xmin + 1.1, ymin + 2.1),
        color=THEME["slate"],
        fontsize=9.5,
        arrowprops={"arrowstyle": "-|>", "color": THEME["slate"], "lw": 0.9},
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("UNIFORM mean shots-to-win")
    ax.set_ylabel("SPREAD mean shots-to-win")
    ax.set_title("Nominal-versus-stress Pareto frontier", loc="left", color=THEME["ink"], pad=12)
    ax.grid(True, linestyle=(0, (1.2, 3.2)))
    style_axis(ax)
    fig.savefig(out_path)
    plt.close(fig)


def plot_ibr_phase_portrait(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    rows = []
    for row in IBR_ROWS:
        residual = 0.5 * row["attacker_adaptation"] + 0.5 * row["uniform_drift"]
        rows.append({**row, "residual": residual})
    xs = [row["defender_adversarial"] for row in rows]
    ys = [row["residual"] for row in rows]
    x_margin = 0.55
    y_margin = 0.35
    xmin = min(xs) - x_margin
    xmax = max(xs) + x_margin
    ymin = min(ys) - y_margin
    ymax = max(ys) + y_margin

    favorable = Rectangle((0.0, ymin), xmax, 0.0 - ymin, facecolor=THEME["fog"], edgecolor="none", zorder=0)
    ax.add_patch(favorable)
    ax.axvline(0.0, color=THEME["slate"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.axhline(0.0, color=THEME["slate"], linewidth=1.0, linestyle=(0, (4, 3)))

    for seed in (42, 123, 777):
        seed_rows = [row for row in rows if row["seed"] == seed]
        seed_rows.sort(key=lambda row: row["generation"])
        color = SEED_STYLE[seed]["color"]
        for start, end in zip(seed_rows[:-1], seed_rows[1:]):
            ax.annotate(
                "",
                xy=(end["defender_adversarial"], end["residual"]),
                xytext=(start["defender_adversarial"], start["residual"]),
                arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": color, "alpha": 0.78},
                zorder=2,
            )
        for row in seed_rows:
            ax.scatter(
                row["defender_adversarial"],
                row["residual"],
                s=170,
                color=color,
                edgecolor=THEME["white"],
                linewidth=1.2,
                zorder=3,
            )
            ax.text(
                row["defender_adversarial"],
                row["residual"],
                str(row["generation"]),
                ha="center",
                va="center",
                fontsize=8.5,
                color=THEME["white"],
                weight="bold",
                zorder=4,
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SEED_STYLE[seed]["color"], markeredgecolor=THEME["white"], markersize=8, label=f"Seed {seed}")
        for seed in (42, 123, 777)
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    ax.text(
        xmax - 0.05,
        ymin + 0.12,
        "certificate-consistent region",
        ha="right",
        va="bottom",
        fontsize=9,
        color=THEME["slate"],
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\mathrm{defender\_adversarial}_k$")
    ax.set_ylabel(r"$R_k = 0.5\,\mathrm{attacker\_adaptation}_k + 0.5\,\mathrm{uniform\_drift}_k$")
    ax.set_title("IBR diagnostic phase portrait", loc="left", color=THEME["ink"], pad=10)
    ax.grid(True, linestyle=(0, (1.2, 3.2)))
    style_axis(ax)
    fig.savefig(out_path)
    plt.close(fig)


def plot_defender_geometry_map(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    cmap = LinearSegmentedColormap.from_list("entropy_map", [THEME["blue_light"], THEME["sand"], THEME["rust"]])
    entropy_values = np.array([GEOMETRY_ROWS[name]["entropy"] for name in DEFENDER_NAMES])
    asymmetry_values = np.array([GEOMETRY_ROWS[name]["asymmetry"] for name in DEFENDER_NAMES])
    asym_min = asymmetry_values.min()
    asym_span = max(asymmetry_values.max() - asym_min, 1e-6)
    marker_sizes = {
        name: 170.0 + 720.0 * ((GEOMETRY_ROWS[name]["asymmetry"] - asym_min) / asym_span)
        for name in DEFENDER_NAMES
    }

    xs = [GEOMETRY_ROWS[name]["centroid"] for name in DEFENDER_NAMES]
    ys = [GEOMETRY_ROWS[name]["cluster"] for name in DEFENDER_NAMES]
    ax.scatter(
        xs,
        ys,
        s=[marker_sizes[name] for name in DEFENDER_NAMES],
        c=entropy_values,
        cmap=cmap,
        edgecolor=THEME["white"],
        linewidth=1.4,
        zorder=3,
    )

    label_offsets = {
        "UNIFORM": (0.10, 0.30),
        "EDGE": (0.12, -0.45),
        "CLUSTER": (0.14, 0.25),
        "SPREAD": (0.10, -0.38),
        "PARITY": (0.14, 0.32),
    }
    for name in DEFENDER_NAMES:
        dx, dy = label_offsets[name]
        ax.text(
            GEOMETRY_ROWS[name]["centroid"] + dx,
            GEOMETRY_ROWS[name]["cluster"] + dy,
            name,
            fontsize=9.5,
            color=THEME["ink"],
            weight="bold" if name in {"UNIFORM", "SPREAD", "CLUSTER"} else None,
        )

    ax.annotate(
        "largest geometric shift",
        xy=(GEOMETRY_ROWS["SPREAD"]["centroid"], GEOMETRY_ROWS["SPREAD"]["cluster"]),
        xytext=(1.42, 15.48),
        color=THEME["rust"],
        fontsize=9,
        arrowprops={"arrowstyle": "-|>", "color": THEME["rust"], "lw": 1.0},
    )
    ax.annotate(
        "highest local clustering",
        xy=(GEOMETRY_ROWS["CLUSTER"]["centroid"], GEOMETRY_ROWS["CLUSTER"]["cluster"]),
        xytext=(0.36, 21.62),
        color=THEME["blue"],
        fontsize=9,
        arrowprops={"arrowstyle": "-|>", "color": THEME["blue"], "lw": 1.0},
    )

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=entropy_values.min(), vmax=entropy_values.max()), cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label("Marginal entropy")

    ax.set_xlabel("Centroid distance from UNIFORM")
    ax.set_ylabel("Cluster score")
    ax.set_xlim(-0.15, 3.35)
    ax.set_ylim(15.35, 22.15)
    ax.set_title("Defender geometry map", loc="left", color=THEME["ink"], pad=12)
    ax.grid(True, linestyle=(0, (1.2, 3.2)))
    style_axis(ax)
    fig.savefig(out_path)
    plt.close(fig)


def style_board_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, BOARD_SIZE[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_SIZE[0], 1), minor=True)
    ax.grid(which="minor", color=THEME["white"], linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color(THEME["line"])
        spine.set_linewidth(0.9)


def plot_heatmap_grid(out_path: Path, occupancy_maps: dict[str, np.ndarray], behavior_maps: dict[str, np.ndarray]) -> None:
    occupancy_cmap = LinearSegmentedColormap.from_list("occupancy", [THEME["fog"], THEME["blue_light"], THEME["blue"]])
    behavior_cmap = LinearSegmentedColormap.from_list("behavior", [THEME["fog"], THEME["sand"], THEME["rust"]])
    occ_values = np.concatenate([occupancy_maps[name].ravel() for name in DEFENDER_NAMES])
    beh_values = np.concatenate([behavior_maps[name].ravel() for name in DEFENDER_NAMES])
    occ_vmax = float(np.quantile(occ_values, 0.995))
    beh_vmax = float(np.quantile(beh_values, 0.99))

    fig, axes = plt.subplots(len(DEFENDER_NAMES), 2, figsize=(8.0, 14.0), constrained_layout=True)
    for row_idx, name in enumerate(DEFENDER_NAMES):
        ax_occ = axes[row_idx, 0]
        ax_beh = axes[row_idx, 1]
        occ_im = ax_occ.imshow(occupancy_maps[name], vmin=0.0, vmax=occ_vmax, cmap=occupancy_cmap, origin="upper")
        beh_im = ax_beh.imshow(behavior_maps[name], vmin=0.0, vmax=beh_vmax, cmap=behavior_cmap, origin="upper")
        style_board_axis(ax_occ)
        style_board_axis(ax_beh)
        ax_occ.set_ylabel(name, rotation=0, labelpad=26, color=THEME["ink"], fontsize=10, weight="bold", va="center")

        if row_idx == 0:
            ax_occ.set_title("Occupancy probability", color=THEME["ink"], pad=8, fontsize=10.5)
            ax_beh.set_title("First-five-shot frequency", color=THEME["ink"], pad=8, fontsize=10.5)

    cbar_occ = fig.colorbar(occ_im, ax=axes[:, 0], fraction=0.024, pad=0.02)
    cbar_occ.set_label("Cell probability")
    cbar_occ.ax.tick_params(labelsize=8)
    cbar_beh = fig.colorbar(beh_im, ax=axes[:, 1], fraction=0.024, pad=0.02)
    cbar_beh.set_label("Shot density")
    cbar_beh.ax.tick_params(labelsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate appendix diagram assets for the paper.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for figure PDFs.")
    parser.add_argument("--layout-samples", type=int, default=300, help="Number of sampled layouts per scripted defender for occupancy heatmaps.")
    parser.add_argument("--behavior-episodes", type=int, default=36, help="Number of scripted posterior-search episodes per defender for behavior heatmaps.")
    args = parser.parse_args()

    configure_matplotlib()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plot_stage1_pareto(args.outdir / "appendix_pareto_frontier.pdf")
    plot_ibr_phase_portrait(args.outdir / "appendix_ibr_phase_portrait.pdf")
    plot_defender_geometry_map(args.outdir / "appendix_defender_geometry.pdf")
    occupancy_maps, behavior_maps = compute_heatmap_inputs(args.layout_samples, args.behavior_episodes)
    plot_heatmap_grid(args.outdir / "appendix_board_heatmaps.pdf", occupancy_maps, behavior_maps)


if __name__ == "__main__":
    main()
