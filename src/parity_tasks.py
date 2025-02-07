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

def make_batch_Nbit_pair_parity(Ns, batch_size, duplicate=1, classify_in_time=False, device="cpu"):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = (torch.rand(batch_size, M) < 0.5) * 1.
        if classify_in_time:
            if duplicate != 1:
                raise NotImplementedError
            labels = [get_parity_in_time(sequences, N).to(device) for N in Ns]
        else:
            labels = [get_parity_in_time(sequences, N)[:-1].to(device) for N in Ns]
        # in each sequence of length M, duplicate each bit (duplicate) times
        sequences = torch.repeat_interleave(sequences, duplicate, dim=1).unsqueeze(-1).to(device)
    return sequences, labels


def make_batch_Nbit_pair_Nsum(Ns, batch_size, duplicate=1, classify_in_time=False, device="cpu"):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = (torch.rand(batch_size, M) < 0.5) * 1.
        if classify_in_time:
            if duplicate != 1:
                raise NotImplementedError
            labels = [get_parity_in_time(sequences, N).to(device) for N in Ns]
        else:
            labels = [get_parity_in_time(sequences, N)[:-1].to(device) for N in Ns]
        # in each sequence of length M, duplicate each bit (duplicate) times
        sequences = torch.repeat_interleave(sequences, duplicate, dim=1).unsqueeze(-1).to(device)
    return sequences, labels

def get_parity_in_time(sequences, N):
    cumsum = torch.cumsum(sequences, dim=1)
    cumsum_ = torch.cat((torch.zeros((sequences.shape[0], 1)), cumsum[:, :-N]), dim=1)
    labels = (cumsum[:, N-1:] - cumsum_) % 2

    return labels.long()


def make_batch_Nbit_pair_paritysum(Ns, batch_size, duplicate=1, classify_in_time=False, device="cpu", delay=0):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max) + delay
    with torch.no_grad():
        sequences = (torch.rand(batch_size, M) < 0.5) * 1.
        labels = []
        for N in Ns:
            parity, Nsum = get_paritysum_in_time(sequences, N)
            parity = parity[:, :parity.shape[1]-delay]
            labels.append((parity.to(device), Nsum.to(device)))
        # in each sequence of length M, duplicate each bit (duplicate) times
        sequences = torch.repeat_interleave(sequences, duplicate, dim=1).unsqueeze(-1).to(device)
    return sequences, labels

def get_paritysum_in_time(sequences, N):
    cumsum = torch.cumsum(sequences, dim=1)
    cumsum_ = torch.cat((torch.zeros((sequences.shape[0], 1)), cumsum[:, :-N]), dim=1)
    last_N_sum = cumsum[:, N-1:] - cumsum_
    labels = (last_N_sum) % 2

    return labels.long(), last_N_sum

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