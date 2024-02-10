import utils.common_utils as cu
import pickle as pkl
import argparse
import constants
import torch
import numpy as np
import os
import torch.nn as nn
import utils.torch_utils as tu
from continuous.data_helper import t_x_y, t_x_y_vector
from pathlib import Path

this_dir = Path(".").absolute()
os.environ["QT_QPA_PLATFORM"] = "offscreen"

dataset_num = 1


def save_checkpoint(state, checkpoint_dir, model_name):
    filename = os.path.join(checkpoint_dir, model_name + "_ckpt.pth.tar")
    print("=> Saving checkpoint to {}".format(filename))
    torch.save(state, filename)


# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return (
        (out[1].squeeze() - y.squeeze()) ** 2
    ).mean()  # - alpha * torch.log(out[0] + epsilon).mean()


def criterion_nored(out, y, alpha=0.5, epsilon=1e-6):
    # TODO
    # This is wrong. The Likelihood should be product of Beta likelihoods which I believe is intractible.
    # Instead we can use product of Normal Random Variables
    return (
        out[1].squeeze() - y.squeeze()
    ) ** 2  # - alpha * torch.log(out[0] + epsilon)


def criterion_TR(out, trg, y, beta=1.0, epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return (
        beta
        * (
            (
                y.squeeze()
                - trg.squeeze() / (out[0].squeeze() + epsilon)
                - out[1].squeeze()
            )
            ** 2
        ).mean()
    )


def criterion_Xent_nored(out, y):
    """Returns the cross-entropy loss for the given output and target."""
    return nn.CrossEntropyLoss(reduction="none")(out, y)
