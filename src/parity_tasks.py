import torch
import torch.nn.functional as F
import numpy as np

def generate_binary_sequence(M, balanced=False):
    if balanced:
        # for dms if the input sequence is correlated it'll make one class very likely
        return (torch.rand(M) < 0.5) * 1.
    else:
        # doesn't seem to have the same effect for nbit-parity
        return (torch.rand(M) < torch.rand(1)) * 1.

# Experiments (not included in paper) with sparse sequences
def generate_sparse_binary_sequence(M, sparsity=0.9):
    s = torch.rand(M) * 2 - 1
    s = torch.where(torch.abs(s) > sparsity, torch.sign(s), 0 * s)
    return s * 1.



############ N_PARITY TASKS ##############

def get_parity(vec, N):
    return (vec[-N:].sum() % 2).long()

def make_batch_Nbit_pair_parity(Ns, bs, duplicate=1, classify_in_time=False):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]
        if classify_in_time:
            if duplicate != 1:
                raise NotImplementedError
            labels = [torch.stack([get_parity_in_time(s, N) for s in sequences]) for N in Ns]
        else:
            labels = [torch.stack([get_parity(s, N) for s in sequences]) for N in Ns]
        # in each sequence of length M, duplicate each bit (duplicate) times
        sequences = [torch.repeat_interleave(s, duplicate, dim=0) for s in sequences]
        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2)
    return sequences, labels


def get_parity_in_time(vec, N):
    labels = []
    for idx in np.arange(N, len(vec)):
        vec_t = vec[idx-N:idx]
        l = get_parity(vec_t, N)
        labels.append(l)
        # labels.append(((vec_t + 1) / 2)[-N:].sum() % 2)

    labels = torch.stack(labels).long()

    return labels

############ DMS TASKS ##################

def get_match(vec, N):
    return (vec[-N] == vec[-1]).long()

def make_batch_multihead_dms(Ns, bs):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M, balanced=True).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_match(s, N) for s in sequences]).squeeze() for N in Ns]

        sequences = torch.stack(sequences)
        sequences = sequences.permute(1, 0, 2)

    return sequences, labels