import math
from typing import List, Optional

import torch
import torch.nn as nn



class LIF(torch.nn.Module):

    def __init__(
        self,
        v_rest: float,
        v_reset: float,
        v_threshold: float = 1.27,
        tau: float = 50.0,
        spike_duration:int = 1,
        learn_tau = False,
        delta_t: float = 1.0,
    ):
        super(LIF, self).__init__()
        self.v_threshold = v_threshold
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.tau = tau
        self.spike_duration = spike_duration #for neuronio, spikes are a platform
        self.spike_pos = spike_duration+1-spike_duration//2
        self.learn_tau = learn_tau
        if self.tau<0:
            decay = torch.tensor(0)
        else:
            decay = torch.tensor(1/self.tau)
        self.decay = nn.parameter.Parameter(decay, requires_grad=self.learn_tau)
        self.spiking_func = SurrGradSpike()
        
    def forward(self, ode, start_v=None):
        ode = ode.squeeze(-1)
        dv = ode
        v_rec = []
        s_rec = []
        v = torch.zeros(ode.shape[0], device=ode.device)
        v += start_v if not (start_v is None) else self.v_rest
        counter = torch.zeros(ode.shape[0], device=ode.device)
        
        for t in range(ode.shape[1]):            
            #integrate and leak
            v = v - self.decay*(v - self.v_rest)
            v[counter==1] = self.v_reset        ##allow compensate reset
            v += ode[:, t]
            v[counter>1] = self.v_threshold    ##hard set
            
            #spike
            s = self.spiking_func.apply(v - self.v_threshold)
            counter += (self.spike_duration+1)*s.detach()
            #spikes[counter==self.spike_pos] = 1 #should spike immediately? 
            v[counter>1] = self.v_threshold
            dv[s.detach()>0, t-1: t+6] = 0.
            counter -= 1
            counter = torch.clamp(counter, min=0)

            v_rec.append(v)
            s_rec.append(s)

        v_rec = torch.stack(v_rec,dim=1)
        s_rec = torch.stack(s_rec,dim=1)
            
        return v_rec, s_rec, dv


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad