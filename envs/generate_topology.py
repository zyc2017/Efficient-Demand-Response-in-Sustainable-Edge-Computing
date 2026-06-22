#!/usr/bin/env python3
"""Generate Adj_Matrix and Dev_Num CSVs for scalability experiments.

Used to evaluate WMRS on systems of N = {20, 30, 40, 50, ...} edge nodes,
matching the ring-topology / U_i-distribution assumptions of the original
15-node EUA dataset (see Parametersetting.py and Section V.A of the paper).

Each invocation writes two files to --output-dir:
    Adj_Matrix_N{N}.csv  — N rows, columns = [self, left_neighbor, right_neighbor]
                            (1-indexed, ring topology)
    Dev_Num_N{N}.csv     — N rows, single column with U_i (number of tasks at
                            node i), sampled from a uniform integer
                            distribution matching the original dataset.

Example:
    python generate_topology.py --N 30
    python generate_topology.py --N 50 --user-low 6 --user-high 22 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_ring_adj_matrix(N):
    """Return an (N, 3) int array encoding a ring topology.

    Row i: [self_idx+1, left_neighbor+1, right_neighbor+1] using 1-based
    indexing to match the existing Adj_Matrix.csv convention. The MEC
    environment's action discretiser (MultiMEC.py) reads:
        column 0 -> "stay local"
        column 1 -> "send to left neighbor"
        column 2 -> "send to right neighbor"
    so the column ordering must be preserved.
    """
    adj = np.zeros((N, 3), dtype=int)
    for i in range(N):
        adj[i, 0] = i + 1                  # self
        adj[i, 1] = ((i - 1) % N) + 1      # left neighbour (wrap-around)
        adj[i, 2] = ((i + 1) % N) + 1      # right neighbour (wrap-around)
    return adj


def generate_dev_num(N, low, high, seed):
    """Sample N integer user counts U_i ~ U[low, high] (inclusive).

    Default range [6, 22] matches the empirical min/max of the original
    15-node Dev_Num.csv. Determinism via numpy.default_rng(seed) means the
    same --N + --seed always produces the same CSV — important for
    reproducible scalability experiments.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low, high + 1, size=N)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, required=True,
                        help='Number of edge nodes (must match Parametersetting._BASE_N for extension to kick in correctly)')
    parser.add_argument('--user-low', type=int, default=6,
                        help='Minimum U_i (inclusive). Default matches the original 15-node dataset min.')
    parser.add_argument('--user-high', type=int, default=22,
                        help='Maximum U_i (inclusive). Default matches the original 15-node dataset max.')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for Dev_Num sampling. Adj_Matrix is deterministic and ignores this.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Where to write the CSVs. Defaults to this script\'s directory (Codes/Our/envs).')
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    adj = generate_ring_adj_matrix(args.N)
    dev = generate_dev_num(args.N, args.user_low, args.user_high, args.seed)

    adj_path = out_dir / f'Adj_Matrix_N{args.N}.csv'
    dev_path = out_dir / f'Dev_Num_N{args.N}.csv'

    pd.DataFrame(adj).to_csv(adj_path, header=False, index=False)
    pd.DataFrame(dev).to_csv(dev_path, header=False, index=False)

    print(f"Wrote {adj_path}  ({args.N} rows, ring topology)")
    print(f"Wrote {dev_path}  ({args.N} rows, U_i ∈ [{int(dev.min())}, {int(dev.max())}], mean={dev.mean():.2f}, total tasks={int(dev.sum())})")


if __name__ == '__main__':
    main()
