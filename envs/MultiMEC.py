import numpy as np
from copy import deepcopy
import os
import pandas as pd
from util import *
from zfliter import ZFilter
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent


def find_existing_path(candidates, description):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate {description}. Searched:\n{searched}")

parameter_candidates = [base_dir / "Parametersetting.py"]
parameter_file = find_existing_path(parameter_candidates, "Parametersetting.py")
parameter_dir = parameter_file.parent
if str(parameter_dir) not in sys.path:
    sys.path.append(str(parameter_dir))

from envs.Parametersetting import *

# Allow overriding the per-node task count file via the DEV_NUM_PATH env var
# (parallel to --adj-path for the adjacency matrix). Used to switch between
# the default 15-node setting and the larger scalability settings
# (e.g. Dev_Num_N20.csv ... Dev_Num_N50.csv).
_dev_num_override = os.environ.get('DEV_NUM_PATH', '').strip()
_dev_num_candidates = []
if _dev_num_override:
    _dev_num_candidates.append(Path(_dev_num_override).expanduser().resolve())
_dev_num_candidates.append(base_dir / "Dev_Num.csv")
dev_num_path = find_existing_path(_dev_num_candidates, "Dev_Num.csv")

adj_matrix_path = find_existing_path([base_dir / "Adj_Matrix.csv"], "Adj_Matrix.csv")

# Number of tasks per edge node and inter-node adjacency (ring topology)
para_EC_task_number = np.array(pd.read_csv(dev_num_path, header=None)).flatten()
adj_matrix = np.array(pd.read_csv(adj_matrix_path, header=None))


class MEC(object):
    """Multi-access Edge Computing environment for joint workload migration and
    resource scheduling under demand response (WMRS paper)."""

    def __init__(self, args):
        self.args = args
        self.time_step = 0
        self.CPU_cycles = CPU_cycles
        self.Task_size = Task_size
        self.EDR_precent = EDR_precent
        self.EDR_penalty = EDR_penalty
        self.para_task_unit_price = para_task_unit_price
        self.para_task_compensation = para_task_compensation
        self.para_EC_dr_unit_reward = para_EC_dr_unit_reward
        self.para_EC_dr_unsatisfied_coff = para_EC_dr_unsatisfied_coff
        self.done = False
        self.adj_matrix = self.args.adj_matrix
        self.sim_len = sim_length
        self.agent_num = len(para_EC_task_number)

        # ── Task-requirement uncertainty (R1 setup) ───────────────────────
        # TASK_CPU_MAX_EPS > 0 enables the "imperfect computing-requirement
        # information" mode: agent observes a noisy estimate ρ̂_ij of each
        # task's true CPU cycles, plus the per-task relative bound ε_ij ∈
        # [0, TASK_CPU_MAX_EPS]. The environment still uses the TRUE ρ_ij
        # for delay/energy. When the env var is unset / 0, behaviour is
        # bit-for-bit identical to the original code.
        self.task_cpu_max_eps = float(os.environ.get('TASK_CPU_MAX_EPS', '0'))
        _test_eps = os.environ.get('TASK_CPU_TEST_EPS', '').strip()
        # If TEST_EPS is set, every task gets this exact ε (useful for
        # sweeping a single noise level at evaluation time). Otherwise ε is
        # sampled per task from U[0, MAX_EPS] for domain randomization.
        self.task_cpu_test_eps = float(_test_eps) if _test_eps else None

        seed = self.args.seed if getattr(self.args, 'seed', -1) and self.args.seed > 0 else None
        self.rng = np.random.default_rng(seed)
        self.EC_cpu = self.rng.uniform(para_EC_cpu_freq[0], para_EC_cpu_freq[1], self.agent_num)

        _, self.state_dim = self.observation()
        self.action_dim = np.array(para_EC_task_number).flatten()
        self.agent_attr = []
        self.task_attr = []
        self.state_norm = [ZFilter(self.state_dim[ii], clip=10) for ii in range(self.agent_num)]
        self.obs_norm_update = True

    def set_obs_norm_update(self, update):
        self.obs_norm_update = bool(update)

    def reset(self):
        self.time_step = 0
        self.done = False
        self.agent_attr = []
        self.task_attr = []
        obs, _ = self.observation()
        for ii in range(self.agent_num):
            obs[ii] = self.state_norm[ii](obs[ii], update=self.obs_norm_update)
        return obs

    def observation(self):
        """Sample a new set of tasks and DR parameters for each edge node.

        Task-requirement uncertainty (TASK_CPU_MAX_EPS > 0):
            * ``task_cpu_true`` is the underlying actual CPU-cycle requirement
              used by the environment to compute the realised delay and energy.
            * ``task_cpu_observed`` is the noisy estimate the agent sees in its
              observation, modelling R1's bounded-error scheduling info:
              ``ρ̂_ij = ρ_ij · (1 + δ)`` with ``δ ∈ U[-ε_ij, +ε_ij]``.
            * ``task_cpu_eps`` is the per-task relative bound ε_ij (also
              included in the observation so the agent can reason about
              uncertainty).
        When ``task_cpu_max_eps == 0`` (default), observed ≡ true, no ε is
        appended, so the observation layout matches the original code exactly.
        """
        self.obs = []
        self.obs_len = []
        self.agent_attr = []
        self.task_attr = []
        for ii in range(self.agent_num):
            task_size = self.rng.uniform(para_task_size[0], para_task_size[1], para_EC_task_number[ii])
            # ── Sample TRUE CPU requirement first (this is what the env uses) ──
            task_cpu_true = self.rng.uniform(para_task_proc_density[0],
                                              para_task_proc_density[1],
                                              para_EC_task_number[ii])
            # ── Derive the observed estimate ρ̂ and per-task bound ε ─────────
            if self.task_cpu_max_eps > 0:
                if self.task_cpu_test_eps is not None:
                    task_cpu_eps = np.full(para_EC_task_number[ii], self.task_cpu_test_eps)
                else:
                    task_cpu_eps = self.rng.uniform(0.0, self.task_cpu_max_eps,
                                                     para_EC_task_number[ii])
                delta = self.rng.uniform(-task_cpu_eps, task_cpu_eps,
                                          para_EC_task_number[ii])
                task_cpu_observed = np.maximum(task_cpu_true * (1.0 + delta), 1.0)
            else:
                task_cpu_observed = task_cpu_true
                task_cpu_eps = None

            task_delay = self.rng.uniform(para_task_delay[0], para_task_delay[1], para_EC_task_number[ii])
            # Per-FLOP pricing — two revenue values are needed under
            # uncertainty:
            #   * task_revenue_observed = ρ̂·price  is what the agent thinks
            #     it will earn at decision time (goes into the obs).
            #   * task_revenue_true     = ρ_true·price  is the actual payment
            #     received once the task is run (goes into task_attr so the
            #     env's task_utility, and therefore the RL reward, reflects
            #     the truth, not the estimate).
            # When task_cpu_max_eps == 0 we have observed == true, so both
            # quantities collapse to the original ``task_revenue`` and the
            # behaviour is bit-for-bit identical to the pre-uncertainty code.
            task_revenue_observed = task_cpu_observed * para_task_unit_price[ii]
            task_revenue_true     = task_cpu_true     * para_task_unit_price[ii]
            task_compensation = para_task_compensation[ii] + np.zeros(para_EC_task_number[ii])

            energy_max = para_EC_static_power + para_EC_cpu_energy_coff * average_cpu_density ** 3
            dr_pct = para_EC_dr_percent[ii]
            energy_dr_target = energy_max * (1 - dr_pct)
            energy_dr_paid = energy_max * dr_pct * para_EC_dr_unit_reward[ii]

            # ── Build obs: agent sees ESTIMATE-based revenue; include ε
            #     only when uncertainty mode is active.
            if self.task_cpu_max_eps > 0:
                obs_each = np.hstack((
                    task_size, task_cpu_observed, task_cpu_eps,
                    task_delay, task_revenue_observed,
                    task_compensation, energy_dr_target, energy_dr_paid
                )).flatten()
            else:
                obs_each = np.hstack((
                    task_size, task_cpu_observed, task_delay,
                    task_revenue_observed,
                    task_compensation, energy_dr_target, energy_dr_paid
                )).flatten()
            self.agent_attr.append(np.array([energy_dr_target, energy_dr_paid]))
            # task_attr columns: [size, cpu_TRUE, delay, revenue_TRUE,
            #                     tolerance, compensation_power]
            # TRUE values stored here so env-side delay, energy AND revenue
            # (i.e. the RL reward signal) reflect actual physical outcomes.
            self.task_attr.append(
                np.hstack((task_size, task_cpu_true, task_delay,
                           task_revenue_true, task_delay,
                           task_compensation)).flatten()
            )
            self.obs.append(obs_each)
            self.obs_len.append(len(obs_each))
        return self.obs, self.obs_len

    def step(self, action, weight_in):
        """Execute one time-slot decision.

        Args:
            action: list of length agent_num; each element is a 1-D array whose
                    first U_i entries are task-migration decisions in [-1, 1]
                    and the last entry is the CPU-frequency action in [-1, 1].
            weight_in: (agent_num x 3) array of incoming reward-sharing weights.

        Returns:
            weighted_utility: shaped reward for each agent (used for RL training).
            EC_utility_no_sharing: unshaped local reward for each agent (used for logging).
            new_obs: next observation list.
            done: bool indicating episode end.
            system_task_utility: sum task utility over all edge nodes.
            system_energy: total energy consumption over all edge nodes.
        """
        self.time_step += 1

        # ── 1. Discretise task-migration actions ────────────────────────────
        action_after = deepcopy(action)
        for ii in range(self.agent_num):
            task_offloading = action[ii][:-1]
            for jj in range(len(task_offloading)):
                a = np.clip(task_offloading[jj], -1.0, 1.0)
                if a < -0.33:
                    action_after[ii][jj] = self.adj_matrix[ii, 1]  # left neighbor
                elif a < 0.33:
                    action_after[ii][jj] = self.adj_matrix[ii, 0]  # local
                else:
                    action_after[ii][jj] = self.adj_matrix[ii, 2]  # right neighbor

        # ── 2. Build per-node task queues and account for transmission cost ──
        # Each task entry: [size, cpu, delay, revenue, tolerance, compensation, origin_id]
        Task_set_all = [[] for _ in range(self.agent_num)]
        transmission_energy = np.zeros(self.agent_num)

        for ii in range(self.agent_num):
            task_set = self.task_attr[ii].reshape(-1, para_EC_task_number[ii])
            for jj in range(para_EC_task_number[ii]):
                dest = int(action_after[ii][jj])
                task_row = np.append(task_set[:, jj], ii)
                if dest == ii + 1:
                    # local execution — no transmission overhead
                    Task_set_all[dest - 1].append(task_row)
                else:
                    # migrated task: subtract transmission delay from tolerance budget
                    tx_time = task_set[0, jj] / opt_tramission_rate
                    transmission_energy[ii] += opt_trasmissin_power * tx_time
                    task_row = task_row.copy()
                    task_row[2] = max(0.0, task_row[2] - tx_time)
                    Task_set_all[dest - 1].append(task_row)

        # ── 2b. Enforce paper Section III.A queueing priority: each edge node
        #        processes its own local tasks BEFORE any tasks migrated in
        #        from other nodes. Python's sort is stable, so the original
        #        arrival order within the "local" group and within the
        #        "external" group is preserved (FCFS within each queue).
        #        Local task ↔ origin_id (stored in task_row[-1]) == node index.
        for k in range(self.agent_num):
            Task_set_all[k].sort(key=lambda t: 0 if int(t[-1]) == k else 1)

        # ── 3. CPU frequency: from the learned policy head (freq_head output).
        agent_frequencies = np.array([np.clip(action[ii][-1], -1.0, 1.0) for ii in range(self.agent_num)])

        # ── 4. Compute per-node utility and energy ───────────────────────────
        energy_utility = []
        original_task_utility = []
        system_task_utility = 0.0
        system_energy = 0.0

        for ii in range(self.agent_num):
            _, energy_rev, _, energy_con, task_util = self.computing_task(
                ii, Task_set_all[ii], agent_frequencies[ii], transmission_energy[ii]
            )
            energy_utility.append(energy_rev)
            original_task_utility.append(task_util)
            system_task_utility += task_util
            system_energy += energy_con

        # ── 5. Reward shaping via incoming weights (paper Eq. 7) ─────────────
        EC_utility_no_sharing = (np.array(energy_utility) + np.array(original_task_utility)).flatten()
        weighted_utility = np.zeros(self.agent_num)
        for ii in range(self.agent_num):
            weighted_utility[ii] = (
                weight_in[ii, 0] * EC_utility_no_sharing[ii]
                + weight_in[ii, 1] * EC_utility_no_sharing[int(self.adj_matrix[ii, 1] - 1)]
                + weight_in[ii, 2] * EC_utility_no_sharing[int(self.adj_matrix[ii, 2] - 1)]
            )

        # ── 6. Next observation ───────────────────────────────────────────────
        new_obs, _ = self.observation()
        for ii in range(self.agent_num):
            new_obs[ii] = self.state_norm[ii](new_obs[ii], update=self.obs_norm_update)

        if self.time_step == sim_length:
            self.done = True
            self.time_step = 0

        return weighted_utility, EC_utility_no_sharing, new_obs, self.done, system_task_utility, system_energy

    def computing_task(self, agent_id, task_set, frequency, transmission_energy):
        """Compute task utilities and energy consumption for one edge node.

        Args:
            agent_id: index of the edge node processing these tasks.
            task_set: list of task vectors [size, cpu, delay, revenue, tolerance, comp_power, origin_id].
            frequency: CPU-frequency scaling factor in [-1, 1]; maps to
                       f = (frequency/2 + 0.50001) * EC_cpu[agent_id].
            transmission_energy: energy spent transmitting outgoing tasks (joules).

        Returns:
            task_utility_set: list of [utility, origin_id] pairs.
            energy_revenue: DR monetary compensation for this node.
            task_delay_sum: cumulative queueing+compute delay across all tasks.
            total_energy: total energy consumption (compute + transmit + static).
            task_utility_total: sum of individual task utilities.
        """
        freq = (frequency / 2 + 0.50001) * self.EC_cpu[agent_id]
        task_queue_delay = 0.0
        task_utility_set = []
        task_delay_sum = 0.0
        task_utility_total = 0.0
        energy_consumption = 0.0

        for task in task_set:
            compute_delay = task[1] / freq
            total_delay = task_queue_delay + compute_delay
            task_queue_delay += compute_delay

            if total_delay <= task[2]:
                utility = task[3]
            elif total_delay - task[2] < task[4]:
                utility = max(0.0, task[3] - (total_delay - task[2]) ** task[5])
            else:
                utility = 0.0

            task_delay_sum += total_delay
            task_utility_set.append(np.array([utility, task[-1]]))
            energy_consumption += para_EC_cpu_energy_coff * task[1] * freq ** 2
            task_utility_total += utility

        total_energy = energy_consumption + transmission_energy + para_EC_static_power
        dr_target = self.agent_attr[agent_id][0]
        dr_reward = self.agent_attr[agent_id][1]
        # Pluggable DR revenue: switches between quadratic (paper Eq. 1-2, default),
        # linear, and tiered models via DR_MODEL env var; see Parametersetting.py.
        energy_revenue = compute_dr_revenue(
            total_energy=total_energy,
            dr_target=dr_target,
            dr_reward=dr_reward,
            quad_coeff=para_EC_dr_unsatisfied_coff[agent_id],
        )

        return task_utility_set, energy_revenue, task_delay_sum, total_energy, task_utility_total
