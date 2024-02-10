# %%
import pandas as pd
import pickle as pkl
from pathlib import Path
import torch
import numpy as np
import os
import continuous.dynamic_net as DNet
import continuous.data_helper as cont_data
import utils.common_utils as cu
import constants as constants
import logging

logging.basicConfig(
    filename="news_out.log",
    format="%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s",
    filemode="a+",
)
logger = logging.getLogger(name="continuous_ite")
logger.setLevel(logging.DEBUG)


os.environ["QT_QPA_PLATFORM"] = "offscreen"


def evaluate_seed(dir, pattern, seeds, method):
    """
    Returns the evaluation on both the best and the last model for each pof the experiment.
    """
    t_samples = torch.arange(0.01, 1, 1 / 64).to(cu.get_device())

    with open(
        constants.fcf_dir / "dataset/news/tr_h_1.0_te_h_1.0/news_response.pkl",
        "rb",
    ) as file:
        news_response = pkl.load(file)
    news_response = torch.Tensor(news_response).to(cu.get_device())

    rslts = []
    for seed in seeds:
        cu.set_seed(seed * 100)
        train_matrix, test_matrix, t_grid, indim, test_idxs = cont_data.load_dataset(
            constants.NEWS_CONT, dataset_num=seed
        )
        all_dosage_test, all_x_test, all_y_test = (
            test_matrix[:, 0].to(cu.get_device(), dtype=torch.float64),
            test_matrix[:, 1:-1].to(cu.get_device(), dtype=torch.float64),
            test_matrix[:, -1].to(cu.get_device(), dtype=torch.float64),
        )

        all_x_test = all_x_test.to(cu.get_device())
        test_idxs = torch.LongTensor(test_idxs).to(cu.get_device())

        if method == constants.VCNET or method == constants.VCNET_TR:
            cfg_density = [(indim, 50, 1, "relu"), (50, 50, 1, "relu")]
            num_grid = 10
            cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
            degree = 2
            knots = [0.33, 0.66]
            model = DNet.Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif method == constants.DRNET or method == constants.DRNET_TR:
            cfg_density = [(indim, 50, 1, "relu"), (50, 50, 1, "relu")]
            num_grid = 10
            cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
            isenhance = 1
            model = DNet.Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        elif method == constants.TARNET or method == constants.TARNET_TR:
            cfg_density = [(indim, 50, 1, "relu"), (50, 50, 1, "relu")]
            num_grid = 10
            cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
            isenhance = 0
            model = DNet.Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        else:
            print(method, " not supported")

        model.to(cu.get_device(), dtype=torch.float64)

        model_file = str(pattern).replace("?", str(seed))
        model_path = dir / model_file
        model.load_state_dict(torch.load(
            model_path, map_location=cu.get_device()))
        print(f"Loaded last model {model_path}")

        model.eval()
        mise = None
        with torch.no_grad():
            mise_x = torch.repeat_interleave(all_x_test, 64, dim=0)
            mise_t = t_samples.repeat(all_x_test.shape[0])
            response_preds = model.forward(dosage=mise_t, x=mise_x)[1]
            response_preds = response_preds.view(-1, 64)

        response_tgts = news_response[test_idxs]
        mise = torch.mean(torch.square(response_preds - response_tgts))

        rslts.append(round(torch.sqrt(mise).item(), 3))
    return rslts


if __name__ == "__main__":
    rslts = evaluate_seed(
        dir=Path(
            constants.fcf_dir / "continuous/results/news/GI/Vcnet_tr/models/giks-model"
        ),
        pattern="best_val_model-?.pt",
        seeds=np.arange(20),
        method=constants.VCNET_TR,
    )
    logger.info(rslts)
    logger.info(f"{np.mean(rslts)} $\pm$ {np.std(rslts)}")
    print(f"{np.mean(rslts)} $\pm$ {np.std(rslts)}")
