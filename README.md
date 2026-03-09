# Battleship-RL Core Repository

This repository contains the core implementation for the Battleship adversarial latent-state training benchmark. It is the codebase associated with the paper on adversarial latent-initial-state training for robustness in partially observable domains. The repository implements the Battleship environment, defender families, attacker training pipelines, restricted iterative best response, evaluation and logging utilities, and a lightweight C backend for high-throughput environment execution.

The archive is intended to expose the implementation clearly and in a form suitable for academic inspection. In particular, it includes the environment and algorithmic code, the training and evaluation scripts, the C core, and the full test suite. It should be read as the executable companion to the paper, rather than as a polished software package for general users.

## Scope and design

The benchmark studies a restricted adversarial setting in which the defender does not perturb the agent online. Instead, it selects a hidden initial latent state, or a distribution over such latent states, before the episode begins. In Battleship, this hidden latent is the ship layout. The attacker then acts under partial observability by firing sequential shots, observing only miss, hit, and sunk events.

The repository therefore supports two distinct but connected goals. The first is empirical benchmarking of robustness under latent distribution shift. The second is the implementation of the training and evaluation machinery needed for the paper’s adversarial latent-state experiments, including fixed-mixture training, alternating stress exposure, and restricted iterative best response.

## Repository structure

The code is organized into a small number of top-level components.

### `battleship_rl/envs/`

This directory contains the environment definitions.

- `battleship_env.py` implements the main Gymnasium-compatible Battleship environment.
- `diagnosis_env.py` implements the secondary diagnosis-style POMDP used in the broader benchmark.
- `defender_env.py` implements the restricted defender-training environment used for iterative best response.
- `observations.py`, `rewards.py`, `masks.py`, `placement.py`, and related files contain environment-side utilities.

The main Battleship environment uses a three-channel observation tensor of shape `(3, H, W)` corresponding to Hit, Miss, and Unknown. Action space is discrete over board cells. Invalid actions are either truncated with a fixed penalty or raised as errors in debug mode. Action masks are computed directly from the hit and miss grids.

### `battleship_rl/agents/`

This directory contains the learned-policy and defender-side components.

- `policies.py` defines the neural feature extractor used by the masked PPO attacker.
- `defender.py` defines the scripted defender families, including nominal and shifted layout distributions.
- `callbacks.py` contains training callbacks and utility hooks.

The policy architecture used in the main experiments is a masked PPO attacker operating on the public board tensor, with a convolutional feature extractor and separate actor/critic heads.

### `battleship_rl/baselines/`

This directory contains the scripted baselines.

- `random_agent.py` implements a uniform-random valid-action baseline.
- `heuristic_probmap.py` implements the probability-map baseline.
- `particle_belief.py` implements the particle-belief baseline.
- `diagnosis_baselines.py` contains baselines for the diagnosis environment.

These baselines provide the absolute performance context for the paper. The strongest scripted baselines remain stronger than the learned attackers at the budgets used in the reported experiments.

### `battleship_rl/bindings/` and `csrc/`

These directories contain the C backend and its Python interface.

- `csrc/src/` and `csrc/include/` implement the core Battleship environment logic in C.
- `battleship_rl/bindings/c_api.py` provides the Python binding layer.
- `battleship_rl/bindings/env_wrapper.py` contains auxiliary wrappers.

The C core is used to increase training throughput and keep environment-step execution stable under large parallel rollout workloads. The Python environment delegates low-level board updates, hit/miss bookkeeping, and termination checks to this backend.

### `battleship_rl/eval/`

This directory contains evaluation and logging code.

- `eval_lib.py` implements the shared evaluation routines.
- `schema.py` defines the evaluation record dataclasses.
- `evaluate.py` contains evaluation entrypoints.

The logging schema is designed to support the paper’s theorem-to-metric bridge. In particular, it records performance statistics, policy diagnostics, and, in Stage 2, the attacker-defender quantities interpreted through the approximate best-response theorems.

### `training/`

This directory contains the main training entrypoints.

- `train_attacker.py` runs Stage 1 attacker training for Regimes A, B, and C.
- `train_ibr.py` runs Stage 2 restricted iterative best response.

These scripts are the primary entrypoints for reproducing the learning pipeline described in the paper.

### `tools/`

This directory contains analysis and utility scripts.

- `compute_defender_metrics.py` computes defender distribution-shift metrics.
- `plot_results.py` produces summary plots from logged evaluations.
- `backfill_eval.py` regenerates corrected evaluation logs from saved checkpoints.
- `run_experiment.py` is a batch driver for experiment orchestration.
- `smoke_eval_stage1.py` and `verify_ibr_metrics.py` provide lightweight verification utilities.

These tools are intended mainly for analysis, diagnostics, and post-training artifact generation rather than for core algorithmic functionality.

### `tests/`

This directory contains the test suite. It includes tests for observation semantics, action masking, invalid-action behavior, seeding reproducibility, SB3 integration, diagnosis-environment behavior, iterative best-response logging, and evaluation diagnostics. The tests are an important part of the repository and should be regarded as part of the scientific evidence for implementation correctness.

## Core implementation properties

A few implementation details are central to the paper and are worth stating explicitly.

The Battleship environment uses a hidden legal layout sampled at reset time. The attacker never observes this layout directly. Public observations are represented as hit, miss, and unknown channels only. Sunk events are emitted through the `info` dictionary rather than encoded as a persistent observation channel.

Action masking is strict. Valid actions are exactly those cells that have not yet been fired upon. The environment exposes a Boolean mask over the flattened action space, and training uses `sb3-contrib`’s masked PPO utilities to ensure that invalid moves are removed from the policy support.

The default reward is a step penalty. This aligns the learned objective with shots-to-win and is the reward used in the main reported experiments. Shaped rewards are supported by the codebase but are treated as optional ablations rather than the default benchmark setting.

The defender families include both a nominal UNIFORM layout generator and several structured shift families, including edge-biased, clustered, spread, and parity-biased variants. These are used for both Stage 1 stress exposure and Stage 2 adversarial evaluation.

Restricted iterative best response is implemented through a defender-side training environment over a latent layout pool, followed by attacker retraining against a mixture of the learned defender-induced latent distribution and the nominal defender. This is the mechanism behind the Stage 2 experiments.

## Installation

The repository assumes Python 3.10+ and a working local compiler toolchain for the C backend.

A minimal installation path is:

```bash
make build_c
pip install -e .
```

The Python dependencies declared in `pyproject.toml` are intentionally small:

- `gymnasium`
- `stable-baselines3>=2.2`
- `sb3-contrib>=2.2`
- `torch`
- `numpy`

A fuller environment may also require packages used by auxiliary scripts, plotting utilities, or local tooling. If you are reproducing all training and analysis commands from the paper, it is safest to install from `requirements.txt` in addition to the editable package installation.

## Building the C backend

The low-level Battleship logic is implemented in `csrc/`. The provided `Makefile` builds the backend and is the expected route for local setup. A minimal build command is:

```bash
make build_c
```

The `csrc/README.md` provides a brief directory-level overview. The core public header is `csrc/include/battleship.h`.

## Running tests

To run the full test suite:

```bash
pytest tests/
```

The tests are designed to check not only basic software behavior but also implementation invariants that matter to the paper, including observation consistency, action-mask correctness, invalid-action handling, seed reproducibility, evaluation-log integrity, and iterative best-response metric generation.

## Training entrypoints

### Stage 1: attacker training

The main attacker-training script is:

```bash
python training/train_attacker.py --regime A
python training/train_attacker.py --regime B
python training/train_attacker.py --regime C
```

The three regimes correspond to:

- `A`: nominal-only training on the UNIFORM defender
- `B`: fixed-mixture training over multiple defender families
- `C`: alternating exposure between nominal and stress distributions

The script supports evaluation checkpoints, run metadata logging, and final evaluation output.

### Stage 2: restricted iterative best response

The Stage 2 training script is:

```bash
python training/train_ibr.py
```

This script initializes an attacker from a Stage 1 checkpoint, trains a restricted defender against it, extracts the defender-induced latent distribution, retrains the attacker on a defender-nominal mixture, and logs the generation-level diagnostics described in the paper.

## Evaluation and analysis

The repository includes both direct evaluation scripts and post-hoc analysis tools.

A standard evaluation entrypoint is:

```bash
python -m battleship_rl.eval.evaluate
```

Useful analysis scripts include:

```bash
python tools/compute_defender_metrics.py
python tools/plot_results.py
python tools/backfill_eval.py
python tools/verify_ibr_metrics.py
```

These are especially relevant when generating the paper’s figures, defender-shift tables, or corrected evaluation records after training.

## Reproducibility notes

This repository is intended to make the implementation transparent and auditable. However, a code archive alone is not sufficient to reproduce the full paper unless the corresponding model checkpoints and result artifacts are also available. In particular, final tables and plots in the paper depend on saved training outputs and evaluation logs.

As a result, this repository should be interpreted as the executable core of the benchmark, not as a self-contained results package. It contains the environment, algorithms, scripts, and tests needed to understand and rerun the method, but exact numerical replication requires the associated experiment outputs.

## What this repository is for

This repository should be useful in three ways.

First, it provides a concrete reference implementation of adversarial latent-state training in a finite-horizon partially observable benchmark.

Second, it makes the benchmark and its experimental machinery inspectable by readers of the paper, including reviewers interested in implementation details, environment semantics, or logging correctness.

Third, it serves as a starting point for future work on hidden-latent robustness in sequential domains. Although the current implementation centers on Battleship, the broader motivation of the paper is that similar adversarial latent-state structures arise in other sequential decision problems, including physically constrained or process-conditioned settings in graphics and machine learning.

## Citation

If you use this repository, please cite the associated paper.

```bibtex
@misc{ahuja2026adversariallatent,
  title={Adversarial Latent-State Training for Robust Policies in Partially Observable Domains},
  author={Angad Singh Ahuja},
  year={2026},
  note={Code repository accompanying the paper}
}
```

## Contact

For questions about the benchmark, implementation, or the associated paper, please contact Angad Singh Ahuja at [ahujaangadsingh@gmail.com]

## Acknowledgements

Parts of this repository were developed with the assistance of Claude 4.6 Opus and OpenAI Codex namely GPT 5.4 Thinking with human supervision and verification at each step.
