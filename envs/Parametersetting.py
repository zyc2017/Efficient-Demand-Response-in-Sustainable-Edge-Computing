import os
from pathlib import Path
import numpy as np
import pandas as pd

sim_length = 100
slot_len = 1 # 1s

CPU_cycles = 1000
Task_size = 1600
EDR_precent = 0.25
EDR_penalty = 1

### task parameters: data size, CPU requirements, task delay, and utility
para_task_size = np.array([500, Task_size]) * 10 **(3)
para_task_proc_density = para_task_size * CPU_cycles
para_task_delay = np.array([0.25, 0.3])
para_task_unit_price = np.array([1.8487, 2.4107, 3.153, 1.802, 2.1347, 1.9599, 2.1307, 1.4894, 1.201, 2.4269, 1.4961, 3.4246, 1.5920, 1.974, 2.4174]) * 10 ** (-9)
para_task_compensation = np.array([1.3349379, 1.53631574, 1.13327868, 2.77466122, 2.14108963, 2.53770292, 2.01276763, 1.73730495, 2.23728322, 1.82392111, 2.09624026, 2.58309748, 3.41902154, 2.17389385, 3.079473])
### EC parameters, including task number, CPU cycles, task, utility function, and energy
para_EC_cpu_freq = np.array([15, 25], dtype=np.int64) * 10 ** 9
para_EC_cpu_energy_coff = 1 * 10 ** (-28)
para_EC_static_power = 100
para_EC_dr_percent = np.array([0.18, 0.25, 0.20,  0.25, 0.20, 0.20, 0.15, 0.20, 0.05, 0.10, 0.25, 0.23, 0.16, 0.25, 0.20]) * (EDR_precent / 0.25)
para_EC_dr_unit_reward = np.array([1.71, 1.66, 1.12, 1.421, 1.368, 1.23, 1.65, 1.36, 1.553, 1.313, 1.74, 1.774, 1.694, 1.501, 1.254]) * 0.3
para_EC_dr_unsatisfied_coff = np.array([10, 11, 10, 13, 12, 10, 11, 12, 10, 12, 19.6, 11, 12, 13, 10]) * 10**(-5) * EDR_penalty
opt_tramission_rate = 10 * 10 ** 6
opt_trasmissin_power = 1
# ──────────────────────────────────────────────────────────────────────────────
# Scalability support: when running with more than 15 edge nodes (controlled by
# the DEV_NUM_PATH env var that points to an alternative Dev_Num CSV), extend
# the per-agent parameter arrays above with additional values sampled from the
# SAME distributions used by the original 15-node dataset.
# ──────────────────────────────────────────────────────────────────────────────
_BASE_N = 15
_PARAM_DIR = Path(__file__).resolve().parent

def _detect_target_agent_num():
    """Read the Dev_Num CSV (env-var override or default) to determine N."""
    override = os.environ.get('DEV_NUM_PATH', '').strip()
    candidate = Path(override).expanduser() if override else (_PARAM_DIR / 'Dev_Num.csv')
    try:
        return int(len(pd.read_csv(candidate, header=None)))
    except Exception:
        # Fall back silently to base N if the CSV can't be read (e.g. when
        # Parametersetting is imported in a non-experiment context like
        # unit tests or static analysis).
        return _BASE_N

def _extend_per_agent_arrays(target_n):
    """Append values sampled from the paper-described distributions to each
    per-agent array, taking the array length up to `target_n` while leaving
    the original `_BASE_N` entries untouched."""
    global para_task_unit_price, para_task_compensation
    global para_EC_dr_percent, para_EC_dr_unit_reward, para_EC_dr_unsatisfied_coff

    if target_n <= _BASE_N:
        return  # Nothing to do at default scale.

    extra = target_n - _BASE_N
    # Deterministic per-N seed → identical CSVs produce identical parameters.
    rng = np.random.default_rng(seed=42 + target_n)

    # Same scaling chain as the original 15 entries above so EDR_precent /
    # EDR_penalty knobs continue to work uniformly across all N.
    para_task_unit_price = np.concatenate([
        para_task_unit_price,
        rng.uniform(1.0, 4.0, extra) * 10 ** (-9),
    ])
    para_task_compensation = np.concatenate([
        para_task_compensation,
        rng.uniform(1.0, 4.0, extra),
    ])
    para_EC_dr_percent = np.concatenate([
        para_EC_dr_percent,
        rng.uniform(0.05, 0.25, extra) * (EDR_precent / 0.25),
    ])
    para_EC_dr_unit_reward = np.concatenate([
        para_EC_dr_unit_reward,
        rng.uniform(1.1, 1.8, extra) * 0.3,
    ])
    para_EC_dr_unsatisfied_coff = np.concatenate([
        para_EC_dr_unsatisfied_coff,
        rng.uniform(10.0, 14.0, extra) * 10 ** (-5) * EDR_penalty,
    ])
_extend_per_agent_arrays(_detect_target_agent_num())
average_cpu_density = (para_task_proc_density[0] + para_task_proc_density[1]) / 2 * 9
_DR_MODEL = os.environ.get('DR_MODEL', 'quadratic').strip().lower()
_SUPPORTED_DR_MODELS = ('quadratic', 'linear', 'tiered')
if _DR_MODEL not in _SUPPORTED_DR_MODELS:
    raise ValueError(
        f"Invalid DR_MODEL={_DR_MODEL!r}. Supported values: {_SUPPORTED_DR_MODELS}"
    )


def compute_dr_revenue(total_energy, dr_target, dr_reward, quad_coeff):
    """Monetary compensation from the grid for one edge node in one time slot.

    Args:
        total_energy: actual energy consumption E_i^t.
        dr_target:    grid-imposed energy cap Ê_i^t.
        dr_reward:    base compensation q̂_i^t paid when the target is met.
        quad_coeff:   per-node quadratic penalty coefficient ξ_i^t
                      (= para_EC_dr_unsatisfied_coff[i]).

    Returns:
        Non-negative compensation under the model selected by DR_MODEL.
    """
    # All three models pay the full reward when the target is met / beaten.
    if total_energy <= dr_target:
        return dr_reward

    delta = total_energy - dr_target

    if _DR_MODEL == 'quadratic':
        penalty = quad_coeff * delta ** 2
        return max(0.0, dr_reward - penalty)

    if _DR_MODEL == 'linear':
        xi_lin = 0.1 * quad_coeff * dr_target
        penalty = xi_lin * delta
        return max(0.0, dr_reward - penalty)

    if _DR_MODEL == 'tiered':
        r = delta / max(dr_target, 1e-9)
        if r <= 0.05:
            return 0.75 * dr_reward
        if r <= 0.10:
            return 0.50 * dr_reward
        if r <= 0.20:
            return 0.25 * dr_reward
        return 0.0

    raise ValueError(f"Unsupported DR_MODEL: {_DR_MODEL}")


