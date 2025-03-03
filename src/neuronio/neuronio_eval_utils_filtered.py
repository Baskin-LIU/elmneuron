import time
from typing import List, Union
import pickle
import numpy as np
import torch
from sklearn.metrics import auc, explained_variance_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_curve

from .neuronio_data_utils import (
    DEFAULT_Y_SOMA_THRESHOLD,
    DEFAULT_Y_TRAIN_SOMA_BIAS,
    DEFAULT_Y_TRAIN_SOMA_SCALE,
    create_neuronio_input_type,
    parse_sim_experiment_file,
)

START_SAVE_PATH = '../data_processed/NeuronIOstartpoint/'

def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NeuronioEvaluator:
    def __init__(
        self,
        test_file,
        burn_in_time: Union[int, np.ndarray],
        input_window_size: int = 500,
        ignore_time_at_start_ms: int = 500,
        rest_start: bool = False,
        desired_FP_list: List[float] = [],
        verbose: bool = False,
        encoding: str = None,
        device="cpu",
    ):
        self.device = device
        self.test_file = test_file
        self.burn_in_time = burn_in_time
        self.input_window_size = input_window_size
        self.ignore_time_at_start_ms = ignore_time_at_start_ms
        self.desired_FP_list = desired_FP_list
        self.verbose = verbose
        self.encoding = encoding
        self.rest_start = rest_start

        # some sanity checks
        assert self.burn_in_time < self.input_window_size
        assert self.burn_in_time <= self.ignore_time_at_start_ms

        X_test, y_spike_test, y_soma_test = parse_sim_experiment_file(
            test_file, verbose=verbose, encoding=encoding
        )
        self.X_test = X_test
        self.y_spike_test = y_spike_test
        self.y_soma_test = y_soma_test
        self.recover_points = None
        if rest_start:
            with open(START_SAVE_PATH + test_file[-92:-2]+'_recover.pkl', 'rb') as fp:
                recover_points = pickle.load(fp)
            self.recover_points = recover_points[burn_in_time]

    def evaluate(self, neuron):
        test_predictions = compute_test_predictions(
            neuron=neuron,
            X_test=self.X_test,
            y_spike_test=self.y_spike_test,
            y_soma_test=self.y_soma_test,
            burn_in_time=self.burn_in_time,
            input_window_size=self.input_window_size,
            rest_start = self.rest_start,
            recover_points=self.recover_points,
            device=self.device,
        )
        core_results = filter_and_extract_core_results(
            *test_predictions,
            desired_FP_list=self.desired_FP_list,
            ignore_time_at_start_ms=self.ignore_time_at_start_ms,
            verbose=self.verbose,
        )

        return core_results


# TODO: could improve by preallocating arrays
def compute_test_predictions_multiple_sim_files(
    neuron,
    test_files,
    burn_in_time: int = 0,
    input_window_size: int = 500,
    rest_start: bool = False,
    verbose=False,
    encoding=None,
    entire=False,
    device="cpu",
):
    assert len(test_files) > 0, "need at least one file to parse"

    y_spikes_GT_all, y_spikes_hat_all, y_soma_GT_all, y_soma_hat_all = [], [], [], []

    for test_file in test_files:
        X_test, y_spike_test, y_soma_test = parse_sim_experiment_file(
            test_file, verbose=verbose, encoding=encoding
        )

        if entire:
            y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat = compute_test_predictions_entire(
                neuron=neuron,
                X_test=X_test,
                y_spike_test=y_spike_test,
                y_soma_test=y_soma_test,
                device=device,
            )
        else:    
            recover_points = None
            if rest_start:
                with open(START_SAVE_PATH + test_file[-92:-2]+'_recover.pkl', 'rb') as fp:
                    recover_points = pickle.load(fp)
                recover_points = recover_points[burn_in_time]
    
            y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat = compute_test_predictions(
                neuron=neuron,
                X_test=X_test,
                y_spike_test=y_spike_test,
                y_soma_test=y_soma_test,
                burn_in_time=burn_in_time,
                input_window_size=input_window_size,
                rest_start=rest_start,
                recover_points=recover_points,
                device=device,
            )
        y_spikes_GT_all.append(y_spikes_GT)
        y_soma_GT_all.append(y_soma_GT)
        y_spikes_hat_all.append(y_spikes_hat)
        y_soma_hat_all.append(y_soma_hat)

    y_spikes_GT_all = np.vstack(y_spikes_GT_all)
    y_spikes_hat_all = np.vstack(y_spikes_hat_all)
    y_soma_GT_all = np.vstack(y_soma_GT_all)
    y_soma_hat_all = np.vstack(y_soma_hat_all)

    return y_spikes_GT_all, y_spikes_hat_all, y_soma_GT_all, y_soma_hat_all


"""
Most of the following code was written by David Beniaguev and Oren Amsalem and originates
from https://github.com/SelfishGene/neuron_as_deep_net/blob/master/evaluate_CNN_test.py.
Main changes includes wrapping test prediciton code in a functions, as well as
integrating own model prediction computations.
"""


def calc_AUC_at_desired_FP(y_test, y_test_hat, desired_false_positive_rate=0.01):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())

    linear_spaced_FPR = np.linspace(0, 1, num=20000)
    linear_spaced_TPR = np.interp(linear_spaced_FPR, fpr, tpr)

    desired_fp_ind = min(
        max(1, np.argmin(abs(linear_spaced_FPR - desired_false_positive_rate))),
        linear_spaced_TPR.shape[0] - 1,
    )

    return linear_spaced_TPR[:desired_fp_ind].mean()


def calc_TP_at_desired_FP(y_test, y_test_hat, desired_false_positive_rate=0.0025):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_test_hat.ravel())

    desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
    if desired_fp_ind == 0:
        desired_fp_ind = 1

    return tpr[desired_fp_ind]


def extract_core_results(
    y_spikes_GT,
    y_spikes_hat,
    y_soma_GT,
    y_soma_hat,
    burn_in_times = None,
    desired_FP_list=[0.0025, 0.0100],
    verbose=False,
):
    # evaluate the model and save the results
    if verbose:
        print(
            "----------------------------------------------------------------------------------------"
        )
        print("calculating core results...")

    evaluation_start_time = time.time()

    # store results in the hyper param dict and return it
    evaluations_results_dict = {}       
    for desired_FP in desired_FP_list:
        TP_at_desired_FP = calc_TP_at_desired_FP(
            y_spikes_GT, y_spikes_hat, desired_false_positive_rate=desired_FP
        )
        AUC_at_desired_FP = calc_AUC_at_desired_FP(
            y_spikes_GT, y_spikes_hat, desired_false_positive_rate=desired_FP
        )
        if verbose:
            print("-----------------------------------")
            print("TP  at %.4f FP rate = %.4f" % (desired_FP, TP_at_desired_FP))
            print("AUC at %.4f FP rate = %.4f" % (desired_FP, AUC_at_desired_FP))
        TP_key_string = "TP @ %.4f FP" % (desired_FP)
        evaluations_results_dict[TP_key_string] = TP_at_desired_FP

        AUC_key_string = "AUC @ %.4f FP" % (desired_FP)
        evaluations_results_dict[AUC_key_string] = AUC_at_desired_FP

    if verbose:
        print("--------------------------------------------------")
    fpr, tpr, thresholds = roc_curve(y_spikes_GT.ravel(), y_spikes_hat.ravel())
    AUC_score = auc(fpr, tpr)
    if verbose:
        print("AUC = %.4f" % (AUC_score))
        print("--------------------------------------------------")

    soma_explained_variance_percent = 100.0 * explained_variance_score(
        y_soma_GT.ravel(), y_soma_hat.ravel()
    )
    soma_RMSE = np.sqrt(MSE(y_soma_GT.ravel(), y_soma_hat.ravel()))
    soma_MAE = MAE(y_soma_GT.ravel(), y_soma_hat.ravel())

    if verbose:
        print("--------------------------------------------------")
        print(
            "soma explained_variance percent = %.2f%s"
            % (soma_explained_variance_percent, "%")
        )
        print("soma RMSE = %.3f [mV]" % (soma_RMSE))
        print("soma MAE = %.3f [mV]" % (soma_MAE))
        print("--------------------------------------------------")

    evaluations_results_dict["AUC"] = AUC_score
    evaluations_results_dict[
        "soma_explained_variance_percent"
    ] = soma_explained_variance_percent
    evaluations_results_dict["soma_RMSE"] = soma_RMSE
    evaluations_results_dict["soma_MAE"] = soma_MAE

    evaluation_duration_min = (time.time() - evaluation_start_time) / 60
    if verbose:
        print(
            "finished evaluation. time took to evaluate results is %.2f minutes"
            % (evaluation_duration_min)
        )
        print(
            "----------------------------------------------------------------------------------------"
        )

    return evaluations_results_dict


def filter_and_extract_core_results(
    y_spikes_GT,
    y_spikes_hat,
    y_soma_GT,
    y_soma_hat,
    burn_in_times = None,
    desired_FP_list=[0.0025, 0.0100],
    ignore_time_at_start_ms=500,
    num_spikes_per_sim=[0, 24],
    verbose=False,
):
    simulations_to_eval = np.logical_and(
        (y_spikes_GT.sum(axis=1) >= num_spikes_per_sim[0]),
        (y_spikes_GT.sum(axis=1) <= num_spikes_per_sim[1]),
    )
    fraction_simulation_kept = simulations_to_eval.mean()

    if verbose:
        print("total amount of simualtions is %d" % (y_spikes_GT.shape[0]))
        print(
            "percent of simulations kept = %.2f%s"
            % (100 * fraction_simulation_kept, "%")
        )

    y_spikes_GT_to_eval = y_spikes_GT[simulations_to_eval, ignore_time_at_start_ms:]
    y_spikes_hat_to_eval = y_spikes_hat[simulations_to_eval, ignore_time_at_start_ms:]
    y_soma_GT_to_eval = y_soma_GT[simulations_to_eval, ignore_time_at_start_ms:]
    y_soma_hat_to_eval = y_soma_hat[simulations_to_eval, ignore_time_at_start_ms:]

    core_results = extract_core_results(
        y_spikes_GT_to_eval,
        y_spikes_hat_to_eval,
        y_soma_GT_to_eval,
        y_soma_hat_to_eval,
        burn_in_times,
        desired_FP_list=desired_FP_list,
        verbose=verbose,
    )

    return core_results


def compute_test_predictions(
    neuron,
    X_test,
    y_spike_test,
    y_soma_test,
    burn_in_time: int = 0,
    input_window_size: int = 500,
    rest_start: bool = False,
    recover_points= None,
    v_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
    ignore_synapse_types: bool = False,
    device="cpu",
):
    synapse_types = torch.tensor(create_neuronio_input_type()).to(device)
    if ignore_synapse_types:
        synapse_types = torch.abs(synapse_types).to(device)
    y_soma_test[y_soma_test > v_threshold] = v_threshold

    X_test = np.transpose(X_test, axes=[2, 1, 0])
    y1_test = y_spike_test.T[:, :, np.newaxis]
    y2_test = (
        y_soma_test.T[:, :, np.newaxis] - y_train_soma_bias
    )  # do not apply scale for evaluation

    y1_test_hat = np.zeros(y1_test.shape)
    y2_test_hat = np.zeros(y2_test.shape)



    if rest_start:
        start_time_inds = recover_points
        num_test_splits = start_time_inds.shape[1]
        num_sim = start_time_inds.shape[0]
    else:
        num_test_splits = int(
            2
            + (X_test.shape[1] - input_window_size)
            / (input_window_size - burn_in_time)
        )
        start_time_inds = np.arange(num_test_splits, dtype=int) * (input_window_size - burn_in_time)

    for k in range(num_test_splits):
        start_time_ind = start_time_inds[:, k]
        end_time_ind = start_time_ind + input_window_size
        if rest_start:
            curr_X_test = []
            for i in range(num_sim):
                section = X_test[i, start_time_ind[i]:end_time_ind[i], :]
                if section.shape[0] < input_window_size:
                    section = np.pad(section, ((0, input_window_size - section.shape[0]), (0,0)), mode='constant', constant_values = 0)
                curr_X_test.append(section)
            curr_X_test = np.stack(curr_X_test, axis=0)
        else:
            curr_X_test = X_test[:, start_time_ind:end_time_ind, :]
        
        # padding added to end to fill up to input_window_size (ok)
        if curr_X_test.shape[1] < input_window_size:
            padding_size = input_window_size - curr_X_test.shape[1]
            X_pad = np.zeros(
                (
                    curr_X_test.shape[0],
                    padding_size,
                    curr_X_test.shape[2],
                )
            )
            curr_X_test  = np.hstack((curr_X_test , X_pad))

        # use model eval forward to get scaled and rectified predictions
        with torch.no_grad():
            spike_input = (
                torch.from_numpy(curr_X_test).float().to(device) * synapse_types
            )
            outputs = neuron.neuronio_eval_forward(spike_input).cpu().numpy()
            curr_y1_test , curr_y2_test  = (
                outputs[..., 0:1],
                outputs[..., 1:2],
            )
        if rest_start:
            if k == 0:
                # prediction for first split (no need to throw away burn in)
                y1_test_hat[:, :end_time_ind[0], :] = curr_y1_test 
                y2_test_hat[:, :end_time_ind[0], :] = curr_y2_test 
            else:
                for i in range(num_sim):
                    if start_time_ind[i]==-1:
                        continue
                    t0 = start_time_ind[i] + burn_in_time
                    if start_time_ind[i] + input_window_size > y1_test_hat.shape[1]:
                        # prediction for last split (throw away burn in, only fill array)
                        duration_to_fill = y1_test_hat.shape[1] - t0
                        y1_test_hat[i, t0:, :] = curr_y1_test[
                            i, burn_in_time : (burn_in_time + duration_to_fill), :
                        ]
                        y2_test_hat[i, t0:, :] = curr_y2_test[
                            i, burn_in_time : (burn_in_time + duration_to_fill), :
                        ]
                    else:
                        # regular prediction for split (throw away burn in)
                        y1_test_hat[i, t0:end_time_ind[i], :] = curr_y1_test[
                            i, burn_in_time:, :
                        ]
                        y2_test_hat[i, t0:end_time_ind[i], :] = curr_y2_test[
                            i, burn_in_time:, :
                        ]
        else:
            if k == 0:
                # prediction for first split (no need to throw away burn in)
                y1_test_hat[:, :end_time_ind, :] = curr_y1_test 
                y2_test_hat[:, :end_time_ind, :] = curr_y2_test 
            elif k == (num_test_splits - 1):
                # prediction for last split (throw away burn in, only fill array)
                t0 = start_time_ind + burn_in_time
                duration_to_fill = y1_test_hat.shape[1] - t0
                y1_test_hat[:, t0:, :] = curr_y1_test[
                    :, burn_in_time : (burn_in_time + duration_to_fill), :
                ]
                y2_test_hat[:, t0:, :] = curr_y2_test[
                    :, burn_in_time : (burn_in_time + duration_to_fill), :
                ]
            else:
                # regular prediction for split (throw away burn in)
                t0 = start_time_ind + burn_in_time
                y1_test_hat[:, t0:end_time_ind, :] = curr_y1_test[
                    :, burn_in_time:, :
                ]
                y2_test_hat[:, t0:end_time_ind, :] = curr_y2_test[
                    :, burn_in_time:, :
                ]

    # NOTE: the following should probably not be done, however,
    # for comparability to previous methods, we leave it in,
    # and due to the large dataset this should be negligible

    # zero score the prediction and align it with the actual test
    s_dst = y2_test.std()
    m_dst = y2_test.mean()

    s_src = y2_test_hat.std()
    m_src = y2_test_hat.mean()

    y2_test_hat = (y2_test_hat - m_src) / s_src
    y2_test_hat = s_dst * y2_test_hat + m_dst

    # convert to simple (num_simulations, num_time_points) format
    y_spikes_GT = y1_test[:, :, 0]
    y_spikes_hat = y1_test_hat[:, :, 0]
    y_soma_GT = y2_test[:, :, 0]
    y_soma_hat = y2_test_hat[:, :, 0]

    return y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat

def compute_test_predictions_entire(
    neuron,
    X_test,
    y_spike_test,
    y_soma_test,
    burn_in_time: int = 0,
    input_window_size: int = 500,
    rest_start: bool = False,
    recover_points= None,
    v_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
    ignore_synapse_types: bool = False,
    device="cpu",
):
    synapse_types = torch.tensor(create_neuronio_input_type()).to(device)
    if ignore_synapse_types:
        synapse_types = torch.abs(synapse_types).to(device)
    y_soma_test[y_soma_test > v_threshold] = v_threshold

    X_test = np.transpose(X_test, axes=[2, 1, 0])
    y1_test = y_spike_test.T[:, :, np.newaxis]
    y2_test = (
        y_soma_test.T[:, :, np.newaxis] - y_train_soma_bias
    )  # do not apply scale for evaluation

    y1_test_hat = np.zeros(y1_test.shape)
    y2_test_hat = np.zeros(y2_test.shape)

    with torch.no_grad():
        for i in range(8):
            X = X_test[i*16: i*16+16]
            spike_input = (
                torch.from_numpy(X).float().to(device) * synapse_types
            )
            outputs = neuron.neuronio_eval_forward(spike_input).cpu().numpy()
            y1_test_hat[i*16: i*16+16] = outputs[..., 0:1]
            y2_test_hat[i*16: i*16+16] = outputs[..., 1:2]

    # NOTE: the following should probably not be done, however,
    # for comparability to previous methods, we leave it in,
    # and due to the large dataset this should be negligible

    # zero score the prediction and align it with the actual test
    s_dst = y2_test.std()
    m_dst = y2_test.mean()

    s_src = y2_test_hat.std()
    m_src = y2_test_hat.mean()

    y2_test_hat = (y2_test_hat - m_src) / s_src
    y2_test_hat = s_dst * y2_test_hat + m_dst

    # convert to simple (num_simulations, num_time_points) format
    y_spikes_GT = y1_test[:, :, 0]
    y_spikes_hat = y1_test_hat[:, :, 0]
    y_soma_GT = y2_test[:, :, 0]
    y_soma_hat = y2_test_hat[:, :, 0]

    return y_spikes_GT, y_spikes_hat, y_soma_GT, y_soma_hat
