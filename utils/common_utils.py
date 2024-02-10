import random
import torch
import numpy as np
from copy import deepcopy
import json
import constants
import pickle as pkl
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

this_dir = Path(".").absolute()


def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cuda:0"


def set_cuda_device(gpu_num: int):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


def insert_kwargs(kwargs: dict, new_args: dict):
    assert type(new_args) == type(kwargs), "Please pass two dictionaries"
    merged_args = kwargs.copy()
    merged_args.update(new_args)
    return merged_args


def dict_print(d: dict):
    d_new = deepcopy(d)

    def cast_str(d_new: dict):
        for k, v in d_new.items():
            if isinstance(v, dict):
                d_new[k] = cast_str(v)
            d_new[k] = str(v)
        return d_new

    d_new = cast_str(d_new)

    pretty_str = json.dumps(d_new, sort_keys=False, indent=4)
    return pretty_str


def set_dump_path(config):
    base_args = config[constants.BASELINE_ARGS]
    gi_args = config[constants.GIKS_ARGS]
    dataset_name = config[constants.DATASET]
    if config[constants.ENFORCE_BASELINE] == True:
        save_path: Path = (
            this_dir
            / "continuous"
            / "results"
            / dataset_name
            / "baseline"
            / base_args[constants.RUN_ALGOS][0]
        )
    else:
        save_path: Path = (
            this_dir
            / "continuous"
            / "results"
            / dataset_name
            / f"GI"
            / gi_args[constants.MODEL_TYPE]
        )

    constants.DUMP_PATH = str(save_path.absolute())
    return True


def get_dump_path() -> Path:
    # with open(this_dir / "continuous" / "results" / "dump_path.txt", "r") as file:
    #     save_path = file.readlines()[0].strip()
    return Path(constants.DUMP_PATH)


class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def plot_ybeta_hist(y, beta, save_name, title, dh):
    betaids = torch.LongTensor([dh._test.beta_to_idx[str(bb)] for bb in beta])
    accs = torch.zeros(dh._test._num_classes, dh._test._num_unqbeta)
    for cls in range(dh._test._num_classes):
        idx = torch.where(y == cls)[0]
        for bid in range(dh._test._num_unqbeta):
            byidx = idx[torch.where(betaids[idx] == bid)[0]]
            accs[cls, bid] = len(byidx)

    df = {}
    for idx in range(dh._test._num_unqbeta):
        df[str(dh._test.idx_to_beta[idx].tolist())] = accs[:, idx]
    df = pd.DataFrame(df)

    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"ctr-y beta - {title}")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)

    plt.close()
    plt.cla()
    plt.clf()


def plot_1d_bar(x_pos: np.array, y_pos: np.array = None, save_path=None):
    """This simply plots the treatment of the dataset
    If y_pos is none, we assume that y_pos=1 for all the x positions

    Args:
        x_pos (_type_): _description_
        y_pos (_type_, optional): _description_. Defaults to None.
    """
    plt.clf()
    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(3)
    if y_pos is None:
        y_pos = np.ones_like(x_pos)
    plt.bar(x=x_pos, height=y_pos, width=0.001)
    save_path = save_path if save_path is not None else "dummy.png"
    plt.savefig(save_path)
