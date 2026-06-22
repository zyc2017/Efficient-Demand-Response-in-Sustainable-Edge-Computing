
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

adj_num = 3


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    tensor = torch.as_tensor(ndarray, dtype=torch.float32, device='cuda' if USE_CUDA else 'cpu')
    if requires_grad:
        tensor.requires_grad_()
    return tensor.type(dtype)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def weight_out_to_in(w_out_before, adj_matrix, is_traing=False):
    if is_traing:
        w_out_before_copy = w_out_before.copy()
        agent_num = len(w_out_before)
        batch_size = len(w_out_before[0])
        for kk in range(batch_size):
            w_out_integraion = np.zeros((agent_num, adj_num))
            for ii in range(agent_num):
                w_out_integraion[ii, :] = w_out_before[ii][kk, :]
            w_out = softmax(w_out_integraion)
            full_w_matrix = weight_convert_single_out_in(w_out, adj_matrix)
            for ii in range(agent_num):
                w_out_before_copy[ii][kk, :] = full_w_matrix[ii, :]
    else:
        w_out = softmax(w_out_before)
        w_out_before_copy = weight_convert_single_out_in(w_out, adj_matrix)
    return w_out_before_copy


def weight_convert_single_out_in(w_out, adj_matrix):
    """Convert outgoing weights to incoming weights.

    w_out[i, j]: weight agent i sends via its j-th neighbour slot (adj_matrix[i, j]).
    Returns weight_in[i, j]: weight agent i receives from its j-th neighbour (adj_matrix[i, j]).

    For each agent i and slot j, the sending neighbour s = adj_matrix[i, j]-1 sends its
    reward via whichever of its own columns points back to i.  We look that column up
    explicitly so the result index always matches adj_matrix[i, j].
    """
    agent_num = adj_matrix.shape[0]
    w_matrix = np.zeros((agent_num, adj_num))
    for ii in range(agent_num):
        w_matrix[ii, 0] = w_out[ii, 0]          # self weight
        for jj in range(1, adj_num):
            neighbor = adj_matrix[ii, jj] - 1   # 0-indexed source
            for kk in range(adj_num):            # find column of neighbour that points back to ii
                if adj_matrix[neighbor, kk] - 1 == ii:
                    w_matrix[ii, jj] = w_out[neighbor, kk]
                    break
    return w_matrix


def softmax(x):
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def weight_out_to_in_tensor(w_out_list, adj_matrix):
    """Torch version of weight_out_to_in that PRESERVES GRADIENTS through the
    cross-agent exchange. Required for the paper-aligned joint optimization
    where each agent's J back-propagates through neighbours' weight networks
    (paper Eq. weightgradient).

    Args:
        w_out_list: list of N tensors, each shape (batch_size, 3), with grad.
                    w_out_list[i] is agent i's RAW weight output (before softmax).
        adj_matrix: numpy array shape (N, 3), 1-indexed neighbour table.

    Returns:
        list of N tensors shape (batch_size, 3), with grad preserved.
        result[i][:, k] = agent i's incoming weight from neighbour adj_matrix[i,k]
                          (softmax-normalised per source agent).
    """
    agent_num = len(w_out_list)
    # Per-agent softmax so each agent's outgoing weights sum to 1
    w_out_soft = [torch.softmax(w, dim=-1) for w in w_out_list]

    w_in_list = []
    for ii in range(agent_num):
        slots = []
        # Column 0: self share (own column 0 of softmax output)
        slots.append(w_out_soft[ii][:, 0:1])
        # Columns 1..adj_num-1: contributions from neighbours
        for jj in range(1, adj_num):
            neighbour = adj_matrix[ii, jj] - 1  # 0-indexed
            # Find which column of `neighbour` points back to `ii`
            kk_back = -1
            for kk in range(adj_num):
                if adj_matrix[neighbour, kk] - 1 == ii:
                    kk_back = kk
                    break
            if kk_back < 0:
                raise ValueError(f"adj_matrix has no back-edge from agent {neighbour} to {ii}")
            slots.append(w_out_soft[neighbour][:, kk_back:kk_back + 1])
        w_in_list.append(torch.cat(slots, dim=1))
    return w_in_list
