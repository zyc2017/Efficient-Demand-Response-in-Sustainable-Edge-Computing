import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from util import *
from model import Actor, Critic, Weight
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, nb_weights, args):

        self.args = args
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.nb_weights = nb_weights
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_weights, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_weights, self.nb_actions, **net_cfg)
        # SEPARATE optimizers per sub-network so freezing one branch truly
        # leaves its parameters and Adam momentum untouched (a single Adam
        # with two param groups would still apply momentum-driven updates
        # even with zeroed gradients).
        self.actor_offload_optim = Adam(self.actor.offload_parameters(), lr=args.prate)
        self.actor_freq_optim    = Adam(self.actor.freq_parameters(),    lr=args.frate)

        self.actor_unchange = Actor(self.nb_states, self.nb_weights, self.nb_actions, **net_cfg)
        self.actor_unchange_offload_optim = Adam(self.actor_unchange.offload_parameters(), lr=0.0001)
        self.actor_unchange_freq_optim    = Adam(self.actor_unchange.freq_parameters(),    lr=0.0001)

        self.weight = Weight(self.nb_states, self.nb_weights, **net_cfg)
        self.weight_target = Weight(self.nb_states, self.nb_weights, **net_cfg)
        self.weight_optim = Adam(self.weight.parameters(), lr=args.wrate)

        self.critic = Critic(self.nb_states, self.nb_actions, self.nb_weights, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, self.nb_weights,  **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.actor_unchange, self.actor)
        hard_update(self.critic_target, self.critic)
        hard_update(self.weight_target, self.weight)

        # LR schedulers: cosine anneal from initial LR down to 1% over the training horizon.
        # Critic is updated every step; actor/weight are updated every policy_freq steps —
        # their T_max must match their actual call count, not the total step count.
        T_max_critic = max(1, args.train_iter - args.warmup)
        T_max_actor  = max(1, (args.train_iter - args.warmup) // args.policy_freq)
        self.actor_offload_scheduler = CosineAnnealingLR(self.actor_offload_optim, T_max=T_max_actor, eta_min=args.prate * 0.1)
        self.actor_freq_scheduler    = CosineAnnealingLR(self.actor_freq_optim,    T_max=T_max_actor, eta_min=args.frate * 0.1)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=T_max_critic, eta_min=args.rate  * 0.1)
        self.weight_scheduler = CosineAnnealingLR(self.weight_optim, T_max=T_max_actor,  eta_min=args.wrate * 0.1)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=1)
        self.random_process_action = OrnsteinUhlenbeckProcess(size=nb_actions + 1, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)
        self.random_process_weight = OrnsteinUhlenbeckProcess(size=nb_weights, theta=args.ou_theta, mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None
        self.a_t = None
        self.is_training = True
        self.actor_update_step = 0  # counts how many times compute_actor_loss has been called

        if USE_CUDA:
            self.cuda()

    def get_sample_ind(self):
        return self.memory.sample_indx(self.batch_size)

    def get_training_sample(self, sample_index):
        # Sample batch
        state_batch, action_batch, reward_batch, weight_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size, sample_index)
        with torch.no_grad():
            weight_batch_new = to_numpy(self.weight_target(to_tensor(next_state_batch)).detach())
        return [state_batch, action_batch, reward_batch, weight_batch, next_state_batch, terminal_batch, weight_batch_new]

    def update_critic_policy(self, xIput, weight_next_coming):

        state_batch, action_batch, reward_batch, weight_batch, next_state_batch, terminal_batch, weight_in = xIput

        with torch.no_grad():
            next_weight_tensor = to_tensor(weight_next_coming)
            next_state_tensor = to_tensor(next_state_batch)
            action_next_full = self.actor_target([next_state_tensor, next_weight_tensor])
            # Target policy smoothing: add clipped noise to prevent critic overfitting to sharp Q peaks
            noise = torch.clamp(torch.randn_like(action_next_full) * 0.1, -0.2, 0.2)
            action_next_full = torch.clamp(action_next_full + noise, -1.0, 1.0)
            action_next = action_next_full[:, :-1]
            fre_next = action_next_full[:, -1:].contiguous()
            next_q_values = self.critic_target([next_state_tensor, action_next, fre_next, next_weight_tensor])
            target_q_batch = to_tensor(reward_batch)  # + self.discount * to_tensor(terminal_batch) * next_q_values
            # No temporal bootstrapping is used here. In our setting, the per-slot state is sampled i.i.d. at every time step
            # and is independent of the action, i.e. s_{t+1} does not depend on a_t.  Q*(s,a) = r(s, a) + gamma * E[V*(s_{
            # t+1})], where the second term is constant with respect to the action. Hence the immediate reward is a sufficient
            # critic target, and the gamma * Q(s_{t+1}, .) bootstrapping term can be safely omitted.
        # Critic update
        self.critic.zero_grad()
        action_batch_tensor = to_tensor(action_batch)
        q_batch = self.critic([
            to_tensor(state_batch),
            action_batch_tensor[:, :-1],
            action_batch_tensor[:, -1:].contiguous(),
            to_tensor(weight_batch)
        ])
        value_loss = criterion(q_batch, target_q_batch)
        critic_loss_re = value_loss.detach().cpu().numpy() + 0
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()
        self.critic_scheduler.step()


        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss_re

    def Low_level_optimal(self, xIput, check_w_in_detached):
        """Approximate inner-loop optimal action a_hat = π^ϛ(o, w) — paper Eq. appromax.

        Args:
            xIput: training sample tuple.
            check_w_in_detached: tensor (batch, 3) — CURRENT weight network's
                prediction for this state, DETACHED. The inner loop is computed
                against this current weight context.
        """
        state_batch, action_batch, reward_batch, weight_batch, next_state_batch, terminal_batch, weight_in = xIput
        w_input = check_w_in_detached
        hard_update(self.actor_unchange, self.actor)
        for _ in range(2):
            self.actor_unchange.zero_grad()
            action_full = self.actor_unchange([to_tensor(state_batch), w_input])
            policy_loss = - self.critic([
                to_tensor(state_batch),
                action_full[:, :-1],
                action_full[:, -1:].contiguous(),
                w_input
            ])
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_unchange_offload_optim.step()
            self.actor_unchange_freq_optim.step()

        return self.actor_unchange([to_tensor(state_batch), w_input])

    def compute_actor_loss(self, xIput, check_w_in):
        """Paper-aligned: compute J = -(Q_check + ι·log(barrier_arg)) given an
        EXTERNAL check_w_in tensor whose gradients flow back to the (own +
        neighbours') weight networks. Does NOT call backward or step. Caller
        is responsible for summing losses across agents and doing joint backward
        + step (paper Eq. weightupdate + Eq. localupdate)."""
        state_batch, action_batch, reward_batch, weight_batch, next_state_batch, terminal_batch, weight_in = xIput
        self.actor_update_step += 1

        # Step 1: a_hat via inner loop using DETACHED check_w_in
        optimal_action_batch = self.Low_level_optimal(xIput, check_w_in.detach())
        a_hat_offload = optimal_action_batch.detach()

        # Inner loop accumulated grads in critic; clear them.
        self.critic.zero_grad()

        # Step 2: forward actor with LIVE check_w_in (grad flows to actor θ_2
        # AND through check_w_in to weight networks θ_1).
        a_check_full = self.actor([to_tensor(state_batch), check_w_in])
        a_check_offload = a_check_full[:, :-1]
        a_check_freq = a_check_full[:, -1:].contiguous()

        # Step 3: Q_check uses LIVE check_w_in (paper Eq. funcationvalue)
        Q_check = self.critic([to_tensor(state_batch), a_check_offload, a_check_freq, check_w_in])

        # Step 4: Q_hat is fully detached (used only as a reference)
        Q_hat = self.critic([to_tensor(state_batch), a_hat_offload[:, :-1], a_hat_offload[:, -1:].contiguous(),
                             check_w_in.detach()]).detach()

        # Step 5: barrier argument (paper Eq. funcationvalue inner expression)
        a_check_concat = a_check_full
        barrier_arg = (Q_check - Q_hat
                       + (0.2 / 2) * a_check_concat.norm(dim=1, keepdim=True).pow(2)
                       + 1.0)
        barrier_arg = torch.clamp(barrier_arg, min=1e-6)
        # Step 6: single-level objective J (paper Eq. funcationvalue).
        # Caller will sum across agents and backward — gradients then flow to
        # all weight networks involved in the exchange.
        actor_loss = -(Q_check + 0.01 * torch.log(barrier_arg)).mean()
        return actor_loss

    def apply_actor_update(self):
        """Step actor + weight optimizers after the caller has done backward on
        the summed cross-agent loss. Also soft-updates target networks."""
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.weight.parameters(), max_norm=1.0)
        self.actor_offload_optim.step()
        self.actor_offload_scheduler.step()
        self.actor_freq_optim.step()
        self.actor_freq_scheduler.step()
        self.weight_optim.step()
        self.weight_scheduler.step()
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.weight_target, self.weight, self.tau)

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.actor_unchange.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
        self.weight.cuda()
        self.weight_target.cuda()

    def observe(self, r_t, weight, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, weight, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions + 1)
        self.a_t = action
        return action

    def random_weight(self):
        weight = np.random.uniform(-1., 1.,self.nb_weights)
        return weight

    def select_action(self, s_t, weight, decay_epsilon=True):
        action = to_numpy(self.actor([to_tensor(s_t.reshape(1, -1)), to_tensor(weight.reshape(1,-1))])).flatten()
        action += self.is_training * max(self.epsilon, 0)*self.random_process_action.sample()
        action = np.clip(action, -1., 1.)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        self.a_t = action
        return action

    def select_weight(self, s_t, decay_epsilon=True):
        weight = to_numpy(self.weight(to_tensor(s_t))).flatten()
        weight += self.is_training * max(self.epsilon, 0) * self.random_process_weight.sample()
        weight = np.clip(weight, -1., 1.)
        return weight

    def reset(self, obs):
        self.s_t = obs
        self.random_process_action.reset_states()
        self.random_process_weight.reset_states()

    def save_model(self,output):
        output_path = Path(output).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            output_path / 'actor.pkl'
        )
        torch.save(
            self.critic.state_dict(),
            output_path / 'critic.pkl'
        )
        torch.save(
            self.weight.state_dict(),
            output_path / 'weight.pkl'
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
