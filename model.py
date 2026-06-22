
import numpy as np

import torch
import torch.nn as nn


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Weight(nn.Module):
    def __init__(self, nb_states, nb_weights, hidden1=256, hidden2=128, init_w=3e-3):
        super(Weight, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_weights)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Actor(nn.Module):
    """Two fully-independent sub-networks: one for offload, one for freq.
    The freq sub-network receives the offload decisions as additional input,
    so that the chosen CPU frequency conditions on what tasks each node ends
    up processing, which is determined by the offload decisions."""
    def __init__(self, nb_states, nb_weights, nb_actions,  hidden1=256, hidden2=128, init_w=3e-3):
        super(Actor, self).__init__()
        self.nb_actions = nb_actions
        # Offload sub-network
        self.offload_fc1 = nn.Linear(nb_states, hidden1)
        self.offload_fc2 = nn.Linear(hidden1 + nb_weights, hidden2)
        self.offload_head = nn.Linear(hidden2, nb_actions)
        # Freq sub-network — input is (state ⊕ offload), so it conditions on
        # what tasks will actually be processed at this node.
        self.freq_fc1 = nn.Linear(nb_states + nb_actions, hidden1)
        self.freq_fc2 = nn.Linear(hidden1 + nb_weights, hidden2)
        self.freq_head = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.offload_fc1.weight.data = fanin_init(self.offload_fc1.weight.data.size())
        self.offload_fc2.weight.data = fanin_init(self.offload_fc2.weight.data.size())
        self.offload_head.weight.data.uniform_(-init_w, init_w)
        self.freq_fc1.weight.data = fanin_init(self.freq_fc1.weight.data.size())
        self.freq_fc2.weight.data = fanin_init(self.freq_fc2.weight.data.size())
        self.freq_head.weight.data.uniform_(-init_w, init_w)

    def offload_parameters(self):
        return (list(self.offload_fc1.parameters())
                + list(self.offload_fc2.parameters())
                + list(self.offload_head.parameters()))

    def freq_parameters(self):
        return (list(self.freq_fc1.parameters())
                + list(self.freq_fc2.parameters())
                + list(self.freq_head.parameters()))

    def forward_heads(self, xs):
        x, weight = xs
        # Step 1: offload from (state, weight)
        o = self.relu(self.offload_fc1(x))
        o = self.relu(self.offload_fc2(torch.cat([o, weight], 1)))
        offload = self.tanh(self.offload_head(o))
        # Step 2: freq from (state, offload.detach(), weight). Detaching offload
        # keeps the two sub-networks independent: freq's Q-gradient cannot leak
        # back into the offload sub-network. Both branches still learn from the
        # same Critic but along disjoint parameter paths.
        f = self.relu(self.freq_fc1(torch.cat([x, offload.detach()], dim=1)))
        f = self.relu(self.freq_fc2(torch.cat([f, weight], 1)))
        freq = self.tanh(self.freq_head(f))
        return offload, freq

    def forward(self, xs):
        offload, freq = self.forward_heads(xs)
        return torch.cat([offload, freq], dim=1)

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_weights, hidden1=256, hidden2=128, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions + 1, hidden2)
        self.fc3 = nn.Linear(hidden2 + nb_weights, hidden2)
        self.fc4 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a, fre, weight = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, torch.cat([a, fre.reshape(-1, 1)], 1)], 1))
        out = self.relu(out)

        out = self.fc3(torch.cat([out, weight], 1))
        out = self.relu(out)
        out = self.fc4(out)
        return out
