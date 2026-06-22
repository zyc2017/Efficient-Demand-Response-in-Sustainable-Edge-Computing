#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import pandas as pd
import random
from pathlib import Path
from envs.MultiMEC import MEC
from ddpg import DDPG
from util import *


def find_existing_path(candidates, description):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate {description}. Searched:\n{searched}")


def save_csv(output_dir, file_name, values):
    np.savetxt(output_dir / file_name, values, delimiter=",")


def save_zfilters(output_dir, env):
    for ii, zf in enumerate(env.state_norm):
        np.savez(
            output_dir / f'zfilter_{ii}.npz',
            n=np.array([zf.rs._n]),
            M=zf.rs._M,
            S=zf.rs._S,
        )


def train(num_iterations, agent, env, args):
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for ii in range(len(agent)):
        agent[ii].is_training = True

    step = episode = 0
    episode_reward = 0.
    episode_energy = 0
    episode_task_utility = 0
    episode_loss_critic = 0
    episode_loss_actor = 0
    episode_loss_weight = 0
    actor_loss_all = []
    critic_loss_all = []
    weight_loss_all = []
    reward_all = []
    task_utility_all = []
    energy_all = []
    observation = None
    while step < num_iterations:
        if args.obs_norm_freeze_step > 0:
            env.set_obs_norm_update(step < args.obs_norm_freeze_step)

        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())  # Get the start observation
            for ii in range(len(agent)):
                agent[ii].reset(observation[ii])

        # Two action-selection modes:
        #   initial warmup → fully random (no trained policy exists yet)
        #   normal training → trained policy + OU noise
        in_warmup = step <= args.warmup

        # agent pick weight ...
        weight_list = []
        for ii in range(len(agent)):
            if in_warmup:
                weight_list.append(agent[ii].random_weight())
            else:
                weight_list.append(agent[ii].select_weight(observation[ii]))

        weight_array = np.array(weight_list).flatten().reshape(len(agent), -1)
        weight_nor = softmax(weight_array)
        weight_after = weight_convert_single_out_in(weight_nor, args.adj_matrix)

        # agent pick action ...
        action_list = []
        for ii in range(len(agent)):
            if in_warmup:
                action_list.append(agent[ii].random_action())
            else:
                action_list.append(agent[ii].select_action(observation[ii], weight_after[ii, :]))

        # env response with next_observation, reward, terminate_info

        reward, actual_reward, observation2, done, task_utility, energy_consumption = env.step(
            action_list, weight_after
        )
        observation2 = deepcopy(observation2)
        scaled_reward = np.array(reward, dtype=np.float32) / args.reward_scale

        # agent observe and update policy
        for ii in range(len(agent)):
            agent[ii].observe(
                scaled_reward[ii],
                weight_after[ii, :],
                observation2[ii],
                done,
            )

        # Skip network updates during the initial warmup period.
        if step > args.warmup and agent[0].memory.nb_entries > agent[0].batch_size:
            sample_index = agent[0].get_sample_ind()

            # Obtain the training samples for all agents
            samples_list = []
            for ii in range(len(agent)):
                samples_list.append(agent[ii].get_training_sample(sample_index))

            # Obtain the sharing weights for all agents.
            # samples_list layout: [state, action, reward, weight, next_state,
            #   terminal, weight_batch_new]; the next-step sharing weight is the
            # last element.
            next_weight_out = []
            for ii in range(len(agent)):
                next_weight_out.append(samples_list[ii][-1])

            # exchange the sharing weights
            next_weight_in = weight_out_to_in(next_weight_out, args.adj_matrix, is_traing=True)

            # Update the critic network every step
            critic_loss = 0
            for ii in range(len(agent)):
                critic_lo = agent[ii].update_critic_policy(samples_list[ii], next_weight_in[ii])
                critic_loss = critic_loss + critic_lo
            critic_loss = critic_loss / len(agent)
            actor_loss = 0
            weight_loss = 0
            steps_since_warmup = step - args.warmup
            if steps_since_warmup % args.policy_freq == 0:
                # (a) Forward all weight networks with grad → check_w_outs
                check_w_outs = []
                for ii in range(len(agent)):
                    state_batch_ii = samples_list[ii][0]
                    check_w_outs.append(agent[ii].weight(to_tensor(state_batch_ii)))

                # (b) Cross-agent exchange preserving the gradient graph
                check_w_ins = weight_out_to_in_tensor(check_w_outs, args.adj_matrix)

                # (c) Zero all relevant gradients before forward
                for ii in range(len(agent)):
                    agent[ii].actor.zero_grad()
                    agent[ii].weight.zero_grad()
                    agent[ii].critic.zero_grad()

                # (d) Each agent computes J_i (no backward yet)
                per_agent_losses = []
                for ii in range(len(agent)):
                    loss_ii = agent[ii].compute_actor_loss(samples_list[ii], check_w_ins[ii])
                    per_agent_losses.append(loss_ii)

                # (e) Joint backward: gradients flow through the exchange to
                # all agents' weight networks (paper's "exchange gradients")
                total_actor_loss = sum(per_agent_losses) / len(agent)
                total_actor_loss.backward()

                # (f) Step actor + weight optimizers for every agent
                for ii in range(len(agent)):
                    agent[ii].apply_actor_update()

                actor_loss = float(total_actor_loss.detach().cpu().numpy())
                # weight_loss is now implicit in actor_loss (same J drives both)
                weight_loss = actor_loss

            episode_loss_critic = episode_loss_critic + critic_loss
            episode_loss_actor = episode_loss_actor + actor_loss
            episode_loss_weight = episode_loss_weight + weight_loss

        # update
        step += 1

        episode_reward = episode_reward + np.array(actual_reward).flatten().sum()
        episode_task_utility = episode_task_utility + task_utility
        episode_energy = episode_energy + energy_consumption
        observation = deepcopy(observation2)

        if done: # end of episode
            # Reset without appending an unexecuted terminal action. The last real
            # transition has already been stored through agent.observe(...).
            observation = None
            reward_all.append(episode_reward)

            actor_loss_all.append(episode_loss_actor / env.sim_len)
            critic_loss_all.append(episode_loss_critic / env.sim_len)
            weight_loss_all.append(episode_loss_weight / env.sim_len)
            task_utility_all.append(episode_task_utility / env.sim_len)
            energy_all.append(episode_energy / env.sim_len)
            print('Ep:', episode, '===Reward:', episode_reward / env.sim_len)
            file_name = 'reward' + '_CPU' + str(env.CPU_cycles) + '_Size' + str(env.Task_size) + '_EDRprcent' + str(env.EDR_precent) + '_EDRpena' + str(env.EDR_penalty) + '.csv'
            file_name_crit = 'critic_loss' + '_CPU' + str(env.CPU_cycles) + '_Size' + str(env.Task_size) + '_EDRprcent' + str(env.EDR_precent) + '_EDRpena' + str(env.EDR_penalty) + '.csv'
            file_name_acto = 'actor_loss' + '_CPU' + str(env.CPU_cycles) + '_Size' + str(env.Task_size) + '_EDRprcent' + str(env.EDR_precent) + '_EDRpena' + str(env.EDR_penalty) + '.csv'
            file_name_weigh = 'weight_loss' + '_CPU' + str(env.CPU_cycles) + '_Size' + str(env.Task_size) + '_EDRprcent' + str(env.EDR_precent) + '_EDRpena' + str(env.EDR_penalty) + '.csv'
            file_name_task_utility = 'task_utility' + '_CPU' + str(env.CPU_cycles) + '_Size' + str(env.Task_size) + '_EDRprcent' + str(env.EDR_precent) + '_EDRpena' + str(env.EDR_penalty) + '.csv'
            file_name_energy_consumption = 'energy_consumption' + '_CPU' + str(env.CPU_cycles) + '_Size' + str(env.Task_size) + '_EDRprcent' + str(env.EDR_precent) + '_EDRpena' + str(env.EDR_penalty) + '.csv'

            save_csv(output_dir, file_name, reward_all)
            save_csv(output_dir, file_name_crit, critic_loss_all)
            save_csv(output_dir, file_name_weigh, weight_loss_all)
            save_csv(output_dir, file_name_acto, actor_loss_all)
            save_csv(output_dir, file_name_task_utility, task_utility_all)
            save_csv(output_dir, file_name_energy_consumption, energy_all)
            episode_reward = 0.
            episode += 1
            episode_energy = 0
            episode_task_utility = 0
            episode_loss_critic = 0
            episode_loss_actor = 0
            episode_loss_weight = 0

            # Periodic full checkpoint every `save_every_episodes` episodes:
            # save model weights and observation-normalisation stats, so that
            # training progress is persisted even if the run is interrupted.
            save_every = getattr(args, 'save_every_episodes', 50)
            if save_every > 0 and episode % save_every == 0:
                for ii in range(len(agent)):
                    agent[ii].save_model(output_dir / f'agent_{ii}')
                save_zfilters(output_dir, env)
                print(f'[Checkpoint] Saved at episode {episode} → {output_dir}')

    # Final save at the end of training (ensures the latest state is captured
    # even if the total episode count isn't a multiple of save_every_episodes).
    for ii in range(len(agent)):
        agent[ii].save_model(output_dir / f'agent_{ii}')
    save_zfilters(output_dir, env)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='WMRS: Joint Workload Migration and Resource Scheduling for Demand Response in Edge Computing')
    parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.0001, type=float, help='learning rate of critic network')
    parser.add_argument('--wrate', default=0.0001, type=float, help='learning rate of sharing weights')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy learning rate')
    parser.add_argument('--frate', default=0.0001, type=float, help='frequency learning rate')
    parser.add_argument('--warmup', default=1500, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.5, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--init_w', default=0.1, type=float, help='')
    parser.add_argument('--train_iter', default=1000000, type=int, help='train iters')
    parser.add_argument('--epsilon', default=10000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--adj-path', default='', type=str, help='Optional path to Adj_Matrix.csv')
    parser.add_argument('--save-every-episodes', default=50, type=int,
                        help='Save model checkpoint + zfilter every N episodes. Set 0 to disable periodic saves (only saves at end of training). CSV log files are always saved every episode regardless of this setting.')
    parser.add_argument('--policy-freq', default=4, type=int, help='Actor and weight network update frequency relative to critic (TD3-style delayed update)')
    parser.add_argument('--reward-scale', default=100.0, type=float, help='Divide training reward by this constant before storing it in replay')
    parser.add_argument('--obs-norm-freeze-step', default=5000, type=int, help='Stop updating observation normalization after this many environment steps; <=0 disables freezing')

    args = parser.parse_args()

    if args.seed > 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    base_dir = Path(__file__).resolve().parent
    candidate_adj_paths = [base_dir / "envs" / "Adj_Matrix.csv"]
    if args.adj_path:
        candidate_adj_paths.insert(0, Path(args.adj_path).expanduser().resolve())
    adj_matrix_path = find_existing_path(candidate_adj_paths, "Adj_Matrix.csv")
    adj_matrix = np.array(pd.read_csv(adj_matrix_path, header=None))  # Adjacency matrix with shape N x 3: [self, left neighbor, right neighbor]
    args.adj_matrix = adj_matrix
    env = MEC(args)
    state_dim = env.state_dim
    action_dim = env.action_dim
    weight_dim = 3
    agent_list = [DDPG(state_dim[ii], action_dim[ii], weight_dim, args) for ii in range(env.agent_num)]
    train(args.train_iter, agent_list, env, args)
