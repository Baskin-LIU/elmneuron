import math
from typing import List, Optional
from .neuronio.neuronio_data_utils import DEFAULT_Y_TRAIN_SOMA_SCALE

import torch
import torch.nn as nn

class self_inhibit(nn.Module):

    def __init__(
        self,
        elm,
        v_threshold: float = 1.27,
        tau: float = 50.0,
        inh_strength: float = 1,
        learn_tau = False,
        learn_sigmoid = False, 
        delta_t: float = 1.0,
    ):
        super(self_inhibit, self).__init__()
        self.elm=elm
        self.v_threshold = v_threshold
        self.tau = tau
        self.inh_strength = inh_strength
        self.learn_tau = learn_tau
        self.learn_sigmoid = learn_sigmoid
        
        self.decay = nn.parameter.Parameter(torch.tensor(1/self.tau), requires_grad=self.learn_tau)
        self.sigmoid_scale = nn.parameter.Parameter(torch.tensor(10.0), requires_grad=self.learn_sigmoid)
        self.b = nn.parameter.Parameter(torch.tensor(-1.0), requires_grad=self.learn_sigmoid)
        #self.spiking_func = nn.LeakyReLU(negative_slope=1e2)#
        self.spiking_func = nn.ReLU()
        
    def forward(self, inputs):
        elm_output = self.elm(inputs)
        x = elm_output.squeeze(-1)
        v_rec = []
        s_rec = []
        inh_rec = []    
        inh = torch.zeros(x.shape[0], device=x.device)
        for t in range(x.shape[1]):
            v = x[:, t] - inh         
            #s = self.spiking_func(v-self.v_threshold)
            s = torch.sigmoid(self.sigmoid_scale*(v-self.v_threshold) + self.b)
            inh = self.decay*inh + self.inh_strength * torch.relu(v-self.v_threshold)
            
            v_rec.append(v)
            s_rec.append(s)
            inh_rec.append(inh)        
            
        v_rec = torch.stack(v_rec,dim=1)
        s_rec = torch.stack(s_rec,dim=1)
        inh_rec = torch.stack(inh_rec,dim=1)
        
        #v_rec_clamp = torch.clamp(v_rec, max=self.v_threshold)
        v_rec_clamp = v_rec - s_rec
        #s_rec = torch.sigmoid(self.sigmoid_scale*s_rec - 2.) - self.b
            
        return v_rec, s_rec, inh_rec, v_rec_clamp, x

    def neuronio_eval_forward(
        self, X, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        v_rec, s_rec, inh_rec, v_rec_clamp, x = self.forward(X)

        # apply sigmoid to spike (probability) prediction
        spike_pred = s_rec
        # apply soma scale to soma prediction
        soma_pred = 1 / y_train_soma_scale * v_rec_clamp

        return torch.stack([spike_pred, soma_pred], dim=-1)
        

class LIF(torch.nn.Module):

    def __init__(
        self,
        elm,
        v_rest: float,
        v_reset: float,
        v_threshold: float = 1.27,
        tau: float = 50.0,
        spike_duration:int = 1,
        learn_tau = False,
        delta_t: float = 1.0,
    ):
        super(LIF, self).__init__()
        self.elm = elm
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
        
    def forward(self, inputs, start_v=None):
        elm_output = self.elm(inputs)
        ode = elm_output.squeeze(-1)
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
            #v[counter>1] = self.v_threshold    ##hard set
            
            #spike
            s = self.spiking_func.apply(v - self.v_threshold)
            counter += (self.spike_duration+1)*s.detach()
            dv[s.detach()>0, t-1: t+6] = 0.
            counter -= 1
            counter = torch.clamp(counter, min=0)

            v_rec.append(v)
            s_rec.append(s)

        v_rec = torch.stack(v_rec,dim=1)
        s_rec = torch.stack(s_rec,dim=1)
            
        return v_rec, s_rec, dv, ode

    def neuronio_eval_forward(
        self, X, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        v_rec, s_rec, dv, ode = self.forward(X)

        # apply sigmoid to spike (probability) prediction
        spike_pred = torch.sigmoid(s_rec)
        # apply soma scale to soma prediction
        soma_pred = 1 / y_train_soma_scale * v_rec_clamp

        return torch.stack([spike_pred, soma_pred], dim=-1)


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
        grad = grad_input* (1 / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2 + 0.01)
        return grad