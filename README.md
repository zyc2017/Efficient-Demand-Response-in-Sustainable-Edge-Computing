# WMRS: Joint Workload Migration and Resource Scheduling for Demand Response in Sustainable Edge Computing

This repository contains the reference implementation of **WMRS**, a decentralized
multi-agent reinforcement learning approach for demand response in edge computing.

## Repository structure

```
.
├── main.py                 # Training entry point (training loop)
├── ddpg.py                 # Per-agent multi-agent actor-critic (WMRS) agent
├── model.py                # Networks: local control (actor), reward sharing, critic
├── memory.py               # Experience replay buffer
├── random_process.py       # Ornstein-Uhlenbeck exploration noise
├── zfliter.py              # Running observation normalisation (z-filter)
├── util.py                 # Reward-sharing weight conversion + tensor helpers
├── requirements.txt
└── envs
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.0, NumPy 1.24, pandas 1.5.
the code automatically uses CUDA when available.

## Quick start

Train WMRS on the default 15-node setting:

```bash
python main.py --output output
```

Training logs and checkpoints are written to the `--output` directory (see
[Outputs](#outputs)). Training runs for `--train_iter` environment steps
(default 1,000,000).

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--output` | `output` | Directory for logs and checkpoints |
| `--train_iter` | `1000000` | Total environment steps |
| `--seed` | `1` | Random seed |
| `--warmup` | `1500` | Steps of random exploration before learning starts |
| `--bsize` | `128` | Mini-batch size |
| `--rmsize` | `100000` | Replay buffer size |
| `--hidden1` / `--hidden2` | `256` / `128` | Hidden layer sizes (all networks) |
| `--rate` / `--prate` / `--wrate` / `--frate` | `1e-4` | Learning rates (critic / actor offload / reward sharing / actor freq) |
| `--tau` | `0.001` | Soft target-update rate |
| `--ou_theta` / `--ou_sigma` / `--ou_mu` | `0.15` / `0.5` / `0.0` | Ornstein-Uhlenbeck exploration noise |
| `--policy-freq` | `4` | Actor / reward-sharing update period relative to the critic |
| `--obs-norm-freeze-step` | `5000` | Freeze observation normalisation after this many steps |
| `--save-every-episodes` | `50` | Checkpoint period (episodes); `0` disables periodic saves |

The default experiments use a 15-node topology
provided in `envs/Adj_Matrix.csv` and `envs/Dev_Num.csv`.

To run on the larger networks used in the scalability study (N = 20, 30, 40, 50),
point the environment to the corresponding CSVs via the `--adj-path` argument and
the `DEV_NUM_PATH` environment variable:

```bash
DEV_NUM_PATH=envs/Dev_Num_N30.csv python main.py \
    --adj-path envs/Adj_Matrix_N30.csv --output output_N30
```

The grid compensation model is selectable via the `DR_MODEL` environment variable:

```bash
DR_MODEL=quadratic python main.py   # default: quadratic penalty
DR_MODEL=linear    python main.py   # linear 
DR_MODEL=tiered    python main.py   # discrete 5-tier compliance scoring (100/75/50/25/0%)
```

## Outputs

All files are written to the `--output` directory:

- **CSV logs** (one row per training episode).
- **Model checkpoints**: `agent_{i}/actor.pkl`, `agent_{i}/critic.pkl`,
  `agent_{i}/weight.pkl` for each edge node `i`.
- **Observation normalisation stats**: `zfilter_{i}.npz`.
