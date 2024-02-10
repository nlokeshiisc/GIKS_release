# Copyright (c) 2020, Ioana Bica

import numpy as np
from scipy.integrate import romb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torch
import sys
from continuous.dynamic_net import Vcnet, Drnet
from utils import common_utils as cu
import constants as constants


def get_patient_outcome(x, v, treatment, dosage, scaling_parameter=10):
    v = torch.FloatTensor(v).to(cu.get_device(), dtype=torch.float64)
    if (treatment == 0):
        #y = float(scaling_parameter) * (np.dot(x, v[treatment][0]) + 12.0 * dosage * (dosage - ( np.dot(x, v[treatment][1]) / np.dot(x, v[treatment][2]))) ** 2)
        y = float(scaling_parameter) * (torch.dot(x, v[treatment][0]) + 12.0 * dosage * (
            dosage - 0.75 * (torch.dot(x, v[treatment][1]) / torch.dot(x, v[treatment][2]))) ** 2)
    elif (treatment == 1):
        y = float(scaling_parameter) * (torch.dot(x, v[treatment][0]) + torch.sin(
            torch.pi * (torch.dot(x, v[treatment][1]) / torch.dot(x, v[treatment][2])) * dosage))
    elif (treatment == 2):
        y = float(scaling_parameter) * (torch.dot(x, v[treatment][0]) + 12.0 * (torch.dot(x, v[treatment][
            1]) * dosage - torch.dot(x, v[treatment][2]) * dosage ** 2))

    return y


def plt_adrf(x, y_t, y=None):
    c1 = 'gold'
    c2 = 'grey'
    c3 = '#d7191c'
    c4 = 'red'
    c0 = '#2b83ba'
    #plt.plot(x, y_t, marker='', ls='-', label='Truth', linewidth=4, color=c1)
    plt.scatter(x, y_t, marker='*', label='Truth',
                alpha=0.9, zorder=3, color=c1, s=15)
    if y is not None:
        plt.scatter(x, y, marker='+', label='TransTTE',
                    alpha=0.9, zorder=3, color='#d7191c', s=15)
    plt.grid()
    plt.legend()
    plt.xlabel('Treatment')
    plt.ylabel('Response')
    plt.savefig("transtee.pdf", bbox_inches='tight')
    plt.close()


def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(
        0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples


def get_model_predictions(num_treatments, test_data, model):
    x = test_data['x']
    t = test_data['t']
    d = test_data['d']
    I_logits = model.forward(dosage=d, t=t, x=x)[1]
    return I_logits.cpu().detach()


def compute_eval_metrics(dataset_name, dataset, test_patients, num_treatments, model, train=False):
    mises = []
    ites = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []

    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(
        np.finfo(float).eps, 1, num_integration_samples)
    treatment_strengths = torch.FloatTensor(treatment_strengths).to(
        cu.get_device(), dtype=torch.float64)

    for patient in test_patients:
        if train and len(pred_best) > 10:
            return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)

        assert num_treatments == 1, "We deal wil univariate treatment only"

        for treatment_idx in range(num_treatments):
            test_data = dict()
            test_data['x'] = torch.repeat_interleave(
                patient.view(1, -1), num_integration_samples, dim=0)
            test_data['t'] = torch.ones(num_integration_samples).to(
                cu.get_device(), dtype=torch.float64) * treatment_idx
            test_data['d'] = treatment_strengths

            pred_dose_response = get_model_predictions(
                num_treatments=num_treatments, test_data=test_data, model=model)

            if dataset_name == constants.TCGA_SINGLE_0:
                true_outcomes = [get_patient_outcome(patient, dataset['metadata']['v'], treatment=0, dosage=d) for d in
                                 treatment_strengths]
            if dataset_name == constants.TCGA_SINGLE_1:
                true_outcomes = [get_patient_outcome(patient, dataset['metadata']['v'], treatment=1, dosage=d) for d in
                                 treatment_strengths]
            if dataset_name == constants.TCGA_SINGLE_2:
                true_outcomes = [get_patient_outcome(patient, dataset['metadata']['v'], treatment=2, dosage=d) for d in
                                 treatment_strengths]

            true_outcomes = torch.FloatTensor(true_outcomes)

            mise = romb(torch.square(true_outcomes.squeeze() -
                        pred_dose_response.squeeze()), dx=step_size)
            inter_r = true_outcomes.squeeze() - pred_dose_response.squeeze()
            ite = torch.mean(inter_r ** 2)
            mises.append(mise)
            ites.append(ite)

            max_dosage_pred, max_dosage = treatment_strengths[torch.argmax(
                pred_dose_response)],  treatment_strengths[torch.argmax(true_outcomes)]
            max_y_pred, max_y = true_outcomes[torch.argmax(
                pred_dose_response)], torch.max(true_outcomes)

            dosage_policy_error = (max_y - max_y_pred) ** 2
            dosage_policy_errors.append(dosage_policy_error.item())

    # For 1 treatment case, both dpe and pe should be the same
    return np.mean(np.sqrt(mises)), np.mean(np.sqrt(dosage_policy_errors)), np.mean(ites)
