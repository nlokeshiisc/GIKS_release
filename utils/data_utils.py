import constants as constants
import continuous.data_helper as cont_data
import torch
from utils import common_utils as cu
from utils import early_stopping as ES
from utils import torch_utils as tu
import torch.nn as nn
import continuous.dynamic_net as DNet
from logging import Logger


def get_dataset_seed(dataset_name, dataset_num, num_epoch, suffix, results_path):
    # Load the dataset
    if dataset_name not in [
        constants.TCGA_SINGLE_0,
        constants.TCGA_SINGLE_1,
        constants.TCGA_SINGLE_2,
    ]:
        all_matrix, test_matrix, t_grid, indim, _ = cont_data.load_dataset(
            dataset_name, dataset_num=dataset_num
        )
        data_class = None
    else:
        (
            all_matrix,
            test_matrix,
            t_grid,
            indim,
            data_class,
        ) = cont_data.load_dataset(dataset_name, dataset_num=dataset_num)

    # instantiate early stopping here
    es_args = {
        constants.FWD_FNC: lambda model, x, t, dosage: model.forward(
            dosage=dosage, x=x
        )[1].squeeze(),
        constants.LOGVAL_EPOCHS: 5,
        constants.CHECKPOINTS: [],
    }

    all_dosage, all_x, all_y = constants.matrix_to_txy(all_matrix, cu.get_device())

    es = ES.EarlyStopping(
        all_x=all_x,
        all_y=all_y,
        all_t=None,
        all_d=all_dosage,
        val_pc=0.3,
        results_dir=results_path,
        suffix=suffix,
        num_epochs=num_epoch,
        seed=dataset_num,
        **es_args,
    )

    return (
        all_matrix,
        test_matrix,
        t_grid,
        indim,
        data_class,
        es,
    )


def load_model(*, dataset_name, dataset_num, indim, results_path, **kwargs):
    model_type = kwargs[constants.MODEL_TYPE]
    if dataset_name not in [
        constants.TCGA_SINGLE_0,
        constants.TCGA_SINGLE_1,
        constants.TCGA_SINGLE_2,
    ]:
        cfg_density = [(indim, 50, 1, "relu"), (50, 50, 1, "relu")]
        num_grid = 10
        cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
        degree = 2
        knots = [0.33, 0.66]
        emb_dim = 50
    else:
        cfg_density = [(4000, 100, 1, "relu"), (100, 64, 1, "relu")]
        num_grid = 10
        cfg = [(64, 64, 1, "relu"), (64, 1, 1, "id")]
        degree = 2
        knots = [0.33, 0.66]
        model = DNet.Vcnet(cfg_density, num_grid, cfg, degree, knots)
        model._initialize_weights()
        emb_dim = 64

    model: nn.Module = None  # For compiler auto-complete hints
    if model_type == constants.VCNET or model_type == constants.VCNET_TR:
        model = DNet.Vcnet(cfg_density, num_grid, cfg, degree, knots)
        model._initialize_weights()

    elif model_type == constants.DRNET or model_type == constants.DRNET_TR:
        isenhance = 1
        model = DNet.Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
        model._initialize_weights()

    elif model_type == constants.TARNET or model_type == constants.TARNET_TR:
        isenhance = 0
        model = DNet.Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
        model._initialize_weights()

        # Load the pre-trained model
    model.to(cu.get_device(), dtype=torch.float64)
    if kwargs[constants.LOAD_MODEL] is not None:
        model_path = (
            results_path
            / "models"
            / kwargs[constants.LOAD_MODEL].replace("?", str(dataset_num))
        )
        model.load_state_dict(torch.load(model_path, map_location=cu.get_device()))
        constants.logger.info(f"Loaded pretrained model from: {model_path}")

    return model


def extract_reps(loader, model):
    """Extracts the representations for the given matrix

    Args:
        train_matrix (_type_): _description_
        model (_type_): _description_
    """
    all_reps = []
    model.eval()
    with torch.no_grad():
        for batch_inp, batch_y, batch_ids in loader:
            batch_x = batch_inp[:, 1:].to(cu.get_device(), dtype=torch.float64)
            batch_dosage = batch_inp[:, 0].to(cu.get_device(), dtype=torch.float64)

            factual_args = {constants.RETURN_EMB: True}
            batch_out = model.forward(
                dosage=batch_dosage, x=batch_x, **factual_args
            )  # pass gradient enabled T to model and make the predictions
            all_reps.append(batch_out[2].cpu())
    return torch.cat(all_reps)
