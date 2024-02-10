import pickle as pkl
import torch
import numpy as np
import os
import continuous.dynamic_net as DNet
import continuous.data_helper as cont_data
import continuous.eval as cont_eval
import continuous.models_helper as models_helper
import utils.torch_utils as tu
import utils.common_utils as cu
import constants as constants
import logging
from collections import defaultdict
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from continuous.tcga.tcga_eval import compute_eval_metrics
from utils import early_stopping as ES
from utils import data_utils as du


from pathlib import Path
import torch.nn.functional as F

fcf_dir = Path(".").absolute()
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def run_baselines(
    *,
    dataset_name,
    dataset_nums: list,
    num_epoch,
    logger: logging.Logger,
    suffix="",
    **kwargs,
):
    """This method runs the baselines -- DRNet, VCNet and TARNet
    Args:
        dataset_name (_type_): _description_
        dataset_nums (list): _description_
        method_list (list): _description_
        num_epoch (_type_): _description_
        verbose (_type_): _description_
        logger (logging.Logger): _description_
    """
    save_path = cu.get_dump_path()
    results_dict = {}

    mse_loss = nn.MSELoss()
    wd = 5e-3
    tr_wd = 5e-3

    batch_size = kwargs[constants.BATCH_SIZE]
    method_list = kwargs[constants.RUN_ALGOS]
    results_path: Path = kwargs[constants.RESULTS_PATH]
    tr_lambda = kwargs[constants.TR_LAMBDA]

    assert len(method_list) == 1, "We will run for only one method."

    if dataset_name == "syn":
        assert (
            batch_size == 500
        ), "In the past, a smaller batch size did not work for synthetic dataset"

    for dataset_num in dataset_nums:
        results_dict[dataset_num] = defaultdict(list)

        if dataset_name in [
            constants.TCGA_SINGLE_0,
            constants.TCGA_SINGLE_1,
            constants.TCGA_SINGLE_2,
        ]:
            (
                train_matrix,
                test_matrix,
                t_grid,
                indim,
                data_class,
            ) = cont_data.load_dataset(dataset_name, dataset_num=dataset_num)
        else:
            train_matrix, test_matrix, t_grid, indim, _ = cont_data.load_dataset(
                dataset_name, dataset_num=dataset_num
            )

        all_dosage, all_x, all_y = (
            train_matrix[:, 0].to(cu.get_device(), dtype=torch.float64),
            train_matrix[:, 1:-1].to(cu.get_device(), dtype=torch.float64),
            train_matrix[:, -1].to(cu.get_device(), dtype=torch.float64),
        )
        num_data = len(all_x)
        kwargs["num_data"] = num_data

        # instantiate early stopping here
        """Force idxs is not required. I verified the train and val idxs are consistent by means of us fixing the seed.
        """
        es_args = {
            constants.FWD_FNC: lambda model, x, t, dosage: model.forward(
                dosage=dosage, x=x
            )[1].squeeze(),
            constants.LOGVAL_EPOCHS: 5,
            constants.CHECKPOINTS: [],
        }
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

        # Subsample the dataset into train and val
        trn_idxs, val_idxs = es.train_idxs, es.val_idxs
        all_dosage, all_x, all_y, train_matrix = (
            all_dosage[trn_idxs],
            all_x[trn_idxs],
            all_y[trn_idxs],
            train_matrix[trn_idxs],
        )

        train_loader = cont_data.get_iter(
            train_matrix, batch_size=batch_size, shuffle=True
        )

        all_x_test = test_matrix[:, 1:-1].to(cu.get_device(), dtype=torch.float64)

        model_name = method_list[0]

        cfg_density = [(indim, 50, 1, "relu"), (50, 50, 1, "relu")]
        cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
        if dataset_name in [
            constants.TCGA_SINGLE_0,
            constants.TCGA_SINGLE_1,
            constants.TCGA_SINGLE_2,
        ]:
            cfg_density = [(4000, 100, 1, "relu"), (100, 64, 1, "relu")]
            cfg = [(64, 64, 1, "relu"), (64, 1, 1, "id")]

        model: nn.Module = None  # For compiler auto-complete hints
        if (
            model_name == constants.VCNET
            or model_name == constants.VCNET_TR
            or model_name == constants.VCNET_HSIC
        ):
            num_grid = 10
            degree = 2
            knots = [0.33, 0.66]
            model = DNet.Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == constants.DRNET or model_name == constants.DRNET_TR:
            num_grid = 10
            isenhance = 1
            model = DNet.Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        elif model_name == constants.TARNET or model_name == constants.TARNET_TR:
            num_grid = 10
            isenhance = 0
            model = DNet.Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        model.to(cu.get_device(), dtype=torch.float64)

        hsic_reg = None
        # use Target Regularization?
        if model_name in [
            constants.VCNET_TR,
            constants.DRNET_TR,
            constants.TARNET_TR,
        ]:
            isTargetReg = 1
        else:
            isTargetReg = 0
            if model_name == constants.VCNET_HSIC:
                hsic_reg = DNet.RbfHSIC(sigma_x=1)

        if isTargetReg:
            tr_knots = list(np.arange(0.1, 1, 0.1))
            tr_degree = 2
            TargetReg = DNet.TR(tr_degree, tr_knots)
            TargetReg._initialize_weights()
            TargetReg.to(cu.get_device(), dtype=torch.float64)

        # best cfg for each model
        if model_name == constants.TARNET:
            init_lr = 1e-2
        elif model_name == constants.TARNET_TR:
            init_lr = 1e-2
            tr_init_lr = 0.001
            beta = 1.0
        elif model_name == constants.DRNET:
            init_lr = 0.05
        elif model_name in [constants.DRNET_TR]:
            init_lr = 0.0001
            tr_init_lr = 0.001
            beta = 1.0
        elif model_name == constants.VCNET or model_name == constants.VCNET_HSIC:
            init_lr = 1e-2
        elif model_name == constants.VCNET_TR:
            init_lr = 1e-2
            tr_init_lr = 0.001
            beta = 1.0

        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)
        # Adam optimizer gave much better results on the baselines
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=wd)

        if isTargetReg:
            tr_optimizer = torch.optim.SGD(
                TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd
            )

        print("model = ", model_name)
        logger.info(f"$$$$$$ Model: {model_name}, dataset_num: {dataset_num} $$$$$$$")

        tq_epochs = tqdm(total=num_epoch, desc="loss", position=0, leave=False)
        for epoch in range(num_epoch):
            tq_loader = tqdm(
                total=len(train_loader), desc="floss", position=1, leave=False
            )
            epoch_floss = []
            for idx, (batch_inp, batch_y, batch_ids) in enumerate(train_loader):
                optimizer.zero_grad()
                model.train()
                """This is very important for DANN to work. This starts from 0 and slowly goes to 1"""

                batch_dosage = batch_inp[:, 0].to(cu.get_device(), dtype=torch.float64)
                batch_x = batch_inp[:, 1:].to(cu.get_device(), dtype=torch.float64)
                batch_y = batch_y.to(cu.get_device(), dtype=torch.float64)
                batch_ids = batch_ids.to(cu.get_device(), dtype=torch.int64)

                factual_fwd_args = cu.insert_kwargs(
                    kwargs,
                    {
                        constants.RETURN_EMB: True,
                    },
                )
                out = model.forward(dosage=batch_dosage, x=batch_x, **factual_fwd_args)
                loss_nored = models_helper.criterion_nored(out, batch_y)
                floss = torch.mean(loss_nored)
                epoch_floss.append(floss.item())

                if isTargetReg:
                    trg = TargetReg(batch_dosage)
                    floss = floss + models_helper.criterion_TR(
                        out, trg, batch_y, beta=beta
                    )

                if model_name == constants.VCNET_HSIC:
                    hsic_loss = hsic_reg.forward(input1=batch_x, input2=batch_dosage)
                    floss = floss + hsic_loss

                # floss_epoch.append(loss.item()); tloss_epoch.append(floss.item())
                floss.backward()
                optimizer.step()

                tq_loader.set_description(f"{floss.item()}")
                tq_loader.update(1)

                if isTargetReg:
                    tr_optimizer.zero_grad()
                    out = model.forward(dosage=batch_dosage, x=batch_x, alpha=0)
                    trg = TargetReg(batch_dosage)
                    tr_loss = tr_lambda * models_helper.criterion_TR(
                        out, trg, batch_y, beta=beta
                    )
                    tr_loss.backward()
                    tr_optimizer.step()

            tq_epochs.set_description(f"loss:{round(np.mean(epoch_floss), 4)}")
            tq_epochs.update(1)

            es.log_val_metrics(
                model=model,
                train_floss=floss,
                train_tloss=floss,
                epoch=epoch,
                logger=logger,
            )

        def log_all_test(last_model, perf_model):
            """Logs the test metric on the entire test dataset

            Args:
                last_model True means the lat model and False means the best model
                perf_model This is the model used to evaluate the test metrics
            """
            if dataset_name not in [
                constants.TCGA_SINGLE_0,
                constants.TCGA_SINGLE_1,
                constants.TCGA_SINGLE_2,
            ]:
                test_mse = cont_eval.predictive_mse(
                    perf_model, test_matrix, targetreg=None
                )
                logger.info(f"Last model {last_model} \t test_mse: {test_mse}")
                print(f"Last model {last_model} \t test_mse: {test_mse}")
            else:
                mise, dpe, ite = compute_eval_metrics(
                    dataset_name=dataset_name,
                    dataset=data_class.dataset,
                    test_patients=all_x_test,
                    num_treatments=1,
                    model=perf_model,
                    train=False,
                )
                test_mse = cont_eval.predictive_mse(
                    perf_model, test_matrix, targetreg=None
                )
                logger.info(
                    f"Last model {last_model} -- mise :: {mise} -- dpe :: {dpe} -- ite :: {ite} -- test mse :: {test_mse}"
                )
                print(
                    f"Last model {last_model} \t mise :: {mise}\t dpe :: {dpe}\t ite :: {ite}"
                )
                pass

        logger.info(f"Best validation epoch -- {es._best_epoch}")
        model.load_state_dict(
            torch.load(
                es.models_dir / f"best_val_model-{dataset_num}.pt",
                map_location=cu.get_device(),
            )
        )
        log_all_test(last_model=False, perf_model=model)

    with open(save_path / f"{suffix}.pkl", "wb") as file:
        pkl.dump(results_dict, file)


def clamp(entry):
    return min(max(0, entry), 1)


def sample_linear_delta(*, dosage, num_samples, linear_delta, subtract_delta):
    """Samples delta to be used by performing Taylor's expansion on the dosage"""
    delta_samples = (
        torch.FloatTensor(len(dosage), num_samples)
        .uniform_(-linear_delta, linear_delta)
        .to(cu.get_device(), dtype=torch.float64)
    )
    if subtract_delta == True:
        dosage_delta: Tensor = dosage.view(-1, 1) - delta_samples
    else:
        dosage_delta: Tensor = dosage.view(-1, 1) + delta_samples

    # clamp the altered dosages to be between 0 and 1
    delta_samples[dosage_delta < 0] = 0
    delta_samples[dosage_delta > 1] = 1
    return delta_samples


def sample_far_dosages(*, dosage, linear_delta, num_samples=1, **kwargs):
    """Samples far away dosages to apply kernel smoothing"""
    ctr_sampling_dist = constants.UNIFORM
    if constants.CTR_SAMPLING_DIST in kwargs.keys():
        ctr_sampling_dist = kwargs[constants.CTR_SAMPLING_DIST]

    if ctr_sampling_dist == constants.INV_PROP:
        ds_name = kwargs[constants.DATASET]
        dsnum = kwargs[constants.DATASET_NUM]
        global_trn_idxs = (
            torch.LongTensor(kwargs["global_trn_ids"]).view(-1).to(cu.get_device())
        )
        batch_idxs = kwargs["GP_batch_idxs"]
        dids = global_trn_idxs[batch_idxs]

        with open(
            constants.fcf_dir
            / f"continuous/results/{ds_name}/prop_models/train_prop_{dsnum}.pkl",
            "rb",
        ) as f:
            prop_scores = torch.FloatTensor(pkl.load(f)).to(
                cu.get_device(), dtype=torch.float64
            )[dids]
        prop_scores = 1 / prop_scores
        prop_scores = prop_scores / torch.sum(prop_scores, dim=1).view(-1, 1)

    sampled_betas = []
    for d_idx, d in enumerate(dosage):
        rej_wdow = [clamp(d - linear_delta), clamp(d + linear_delta)]

        if ctr_sampling_dist == constants.UNIFORM:
            delta_samples = (
                torch.FloatTensor(1, num_samples)
                .uniform_(0, 1 - (rej_wdow[1] - rej_wdow[0]))
                .to(cu.get_device())
            )
            delta_samples[delta_samples > rej_wdow[1]] += rej_wdow[1] - rej_wdow[0]

        elif ctr_sampling_dist == constants.INV_PROP:
            idx_prop_scores = prop_scores[d_idx]
            tt_bin = torch.multinomial(
                idx_prop_scores, num_samples, replacement=True
            ) * (1.0 / prop_scores.shape[1])
            delta_samples = tt_bin + torch.FloatTensor(1, num_samples).uniform_(
                0, 1 / prop_scores.shape[1]
            ).to(cu.get_device()).view(-1, 1)

        elif ctr_sampling_dist == constants.MARGINAL_T_DIST:
            all_dosage = kwargs["all_trn_dosage"]
            perm = torch.randperm(all_dosage.size(0))
            idx = perm[:num_samples]
            delta_samples = (
                all_dosage[idx]
                + torch.FloatTensor(1, num_samples)
                .uniform_(0, 0.05)
                .to(cu.get_device())
            ).view(-1, 1)
            delta_samples = torch.clamp(delta_samples, min=0, max=1)

        else:
            assert False, "Unknown sampling distribution"

        sampled_betas.append(delta_samples)
    return torch.cat(sampled_betas, dim=0)


def find_nnT_tgts(
    *, batch_dosage_delta, linear_delta, all_t, all_y, method="GI", **kwargs
):
    """
    If method if GI, the targets labels are just the factual labels of the neighbor. So return the labels directly
    If method is GP, then we have to return ths ids NND
    Note that number of samples in each neighborhood may be totally different.
    """
    batch_d_colview = batch_dosage_delta.view(-1, 1)
    all_d_rowview = all_t.view(1, -1)

    ctr_fct_beta_diff = torch.abs(batch_d_colview - all_d_rowview)

    if method == "GI":
        cand_ids = torch.argmin(ctr_fct_beta_diff, dim=1)

    elif method == "GP":
        candids = [torch.where(row < linear_delta)[0] for row in ctr_fct_beta_diff]
        assert len(candids) == len(batch_dosage_delta)
        return candids
    else:
        assert False
    return all_y[cand_ids]


def GI_reg_valloss(
    *,
    model: DNet.Vcnet,
    val_dosages: torch.Tensor,
    val_xemb: torch.Tensor,
    val_y: torch.Tensor,
    linear_delta,
    **kwargs,
):
    """This is the gi mode for getting losses while tuning on"""
    delta_samples = sample_linear_delta(
        dosage=val_dosages,
        num_samples=1,
        linear_delta=linear_delta,
        subtract_delta=True,
    ).view(-1)

    val_dosages_grad = val_dosages.clone().detach().requires_grad_(True)

    val_dosages_delta = (
        val_dosages_grad - delta_samples
    )  # As we are not passing to nearest neighbor, no need to do this

    gi_ypreds = model.forward_with_emb(dosage=val_dosages_delta, x_emb=val_xemb)[1]
    gi_grads = torch.autograd.grad(
        gi_ypreds,
        val_dosages_delta,
        grad_outputs=torch.ones_like(gi_ypreds),
        create_graph=True,
    )[0]

    ypreds_taylor = gi_ypreds.squeeze() + delta_samples * gi_grads
    gi_loss = nn.MSELoss(reduction="none")(val_y.squeeze(), ypreds_taylor.squeeze())
    return gi_loss


def GP_unf_valloss(
    *,
    model: DNet.Vcnet,
    val_dosages,
    GI_linear_delta,
    GP_linear_delta,
    val_xemb: Tensor,
    val_ys,
    trn_dosages,
    trn_ys,
    trn_embs,
    btmk_var,
    gp_kernel,
    **kwargs,
):
    batch_ids = torch.arange(len(val_dosages)).to(cu.get_device())
    batch_dosage_CF = torch.rand(len(batch_ids)).to(cu.get_device())

    near_idxs = torch.where(torch.abs(val_dosages - batch_dosage_CF) < GI_linear_delta)[
        0
    ]
    far_idxs = torch.where(torch.abs(val_dosages - batch_dosage_CF) >= GI_linear_delta)[
        0
    ]

    factual_losses = None
    model.eval()
    with torch.no_grad():
        fct_preds = model.forward_with_emb(dosage=val_dosages, x_emb=val_xemb)[1]
        factual_losses = nn.MSELoss(reduction="none")(
            val_ys.squeeze(), fct_preds.squeeze()
        )

    gi_losses, gp_losses = None, None
    if len(near_idxs) > 0:
        gi_losses = GI_reg_valloss(
            model=model,
            val_dosages=val_dosages[near_idxs],
            val_xemb=val_xemb[near_idxs],
            val_y=val_ys[near_idxs],
            linear_delta=GI_linear_delta,
        )
    if len(far_idxs) > 0:
        kwargs[constants.FORCE_FAR_DOSAGES] = val_dosages[far_idxs].view(-1, 1)
        kwargs[constants.FORCE_GP_SUPERVISION] = val_ys[far_idxs]
        val_means, _, gp_losses = GP_loss(
            model=model,
            batch_dosage_f=val_dosages[far_idxs],
            GI_linear_delta=GI_linear_delta,
            GP_linear_delta=GP_linear_delta,
            batch_ids=torch.arange(
                len(far_idxs), device=cu.get_device()
            ),  # this is a dummy
            batch_emb=val_xemb[far_idxs],
            trn_dosages=trn_dosages,
            trn_ys=trn_ys,
            trn_embs=trn_embs,
            btmk_var=btmk_var,
            gp_kernel=gp_kernel,
            **kwargs,
        )
        gp_losses = nn.MSELoss(reduction="none")(val_ys[far_idxs], val_means)

    return factual_losses, gi_losses, gp_losses


def GI_reg_loss(
    *,
    model: DNet.Vcnet,
    batch_dosage,
    batch_xemb: torch.Tensor,
    batch_y: torch.Tensor,
    batch_dosage_grad,
    linear_delta,
    GI_num_explore,
    **kwargs,
):
    """This is regularization version of the GI Loss.
    As we vary \beta, we dont want the network to change much.
    This I feel is not very impactful, because this seems to desensitivize the network to changes in \beta
    Perhaps a better alternative according to me is to desensitivize the confounders x
    """
    delta_samples = sample_linear_delta(
        dosage=batch_dosage,
        num_samples=GI_num_explore,
        linear_delta=linear_delta,
        subtract_delta=True,
    ).view(-1)

    batch_dosage_gi = torch.repeat_interleave(batch_dosage_grad, repeats=GI_num_explore)
    batch_dosage_delta = (
        batch_dosage_gi - delta_samples
    )  # As we are not passing to nearest neighbor, no need to do this

    # Tracking the newly augmented treatments
    # aug_dict = kwargs["aug_dict"]
    # batch_idxs = kwargs["batch_idxs"]
    # for _, bid in enumerate(batch_idxs):
    #     aug_dict[bid].extend(
    #         batch_dosage_delta.view(-1, GI_num_explore)[_].cpu().tolist()
    #     )

    x_emb = torch.repeat_interleave(batch_xemb, repeats=GI_num_explore, dim=0)
    gi_tgt_y = torch.repeat_interleave(batch_y, repeats=GI_num_explore)
    # x_emb_gi = torch.repeat_interleave(x_emb, repeats=num_lindelta)
    gi_ypreds = model.forward_with_emb(dosage=batch_dosage_delta, x_emb=x_emb)[1]
    gi_grads = torch.autograd.grad(
        gi_ypreds,
        batch_dosage_delta,
        grad_outputs=torch.ones_like(gi_ypreds),
        create_graph=True,
    )[0]
    # gi_grads = torch.clamp(gi_grads, min=-2, max=2)

    ypreds_taylor = gi_ypreds.squeeze() + delta_samples * gi_grads

    gi_loss = nn.MSELoss()(gi_tgt_y, ypreds_taylor)
    return gi_loss


def GI_reg_loss(
    *,
    model: DNet.Vcnet,
    batch_dosage,
    batch_xemb: torch.Tensor,
    batch_y: torch.Tensor,
    batch_dosage_grad,
    linear_delta,
    GI_num_explore,
    **kwargs,
):
    """This is regularization version of the GI Loss.
    As we vary \beta, we dont want the network to change much.
    This I feel is not very impactful, because this seems to desensitivize the network to changes in \beta
    Perhaps a better alternative according to me is to desensitivize the confounders x
    """
    delta_samples = sample_linear_delta(
        dosage=batch_dosage,
        num_samples=GI_num_explore,
        linear_delta=linear_delta,
        subtract_delta=True,
    ).view(-1)

    batch_dosage_gi = torch.repeat_interleave(batch_dosage_grad, repeats=GI_num_explore)
    batch_dosage_delta = (
        batch_dosage_gi - delta_samples
    )  # As we are not passing to nearest neighbor, no need to do this

    x_emb = torch.repeat_interleave(batch_xemb, repeats=GI_num_explore, dim=0)
    gi_tgt_y = torch.repeat_interleave(batch_y, repeats=GI_num_explore)
    # x_emb_gi = torch.repeat_interleave(x_emb, repeats=num_lindelta)
    gi_ypreds = model.forward_with_emb(dosage=batch_dosage_delta, x_emb=x_emb)[1]
    gi_grads = torch.autograd.grad(
        gi_ypreds,
        batch_dosage_delta,
        grad_outputs=torch.ones_like(gi_ypreds),
        create_graph=True,
    )[0]
    # gi_grads = torch.clamp(gi_grads, min=-2, max=2)

    ypreds_taylor = gi_ypreds.squeeze() + delta_samples * gi_grads

    gi_loss = nn.MSELoss()(gi_tgt_y, ypreds_taylor)
    return gi_loss


def gi_interpolate(
    *,
    gp_NND_dosages: torch.Tensor,
    far_CF_dosage,
    all_t_grads,
    NND_ids,
    all_y,
):
    """This function extrapolates the targets for fat counyterfactual betas using GI loss imposed
    on betas nearby counterfactual beta

    We use GI to interpolate target at \beta^CF from \beta using first order Taylor's expansion
    i.e. y(\beta^CF) = y(\beta) + \nabla_{\beta} y(\beta) (\beta^CF - \beta)

    Args:
        model (_type_): _description_
        gp_X (_type_): _description_
        target_dosage (_type_): _description_
        factual_dosage (_type_): _description_
    """
    NND_dosages_grad = gp_NND_dosages.requires_grad_(True)
    y_beta_factual = all_y[NND_ids]
    gi_NND_grads = all_t_grads[NND_ids]

    delta = far_CF_dosage - NND_dosages_grad
    y_beta_CF: torch.Tensor = y_beta_factual.squeeze() + (
        delta.squeeze() * gi_NND_grads
    )
    return y_beta_CF.detach()


def GP_loss(
    *,
    model: DNet.Vcnet,
    batch_dosage_f,
    GI_linear_delta,
    GP_linear_delta,
    batch_ids,
    batch_emb: Tensor,
    trn_dosages,
    trn_ys,
    trn_embs,
    btmk_var,
    gp_kernel,
    **kwargs,
):
    """Given (xi, \beta_i, y_i), we sample random \betas away from \beta_i +/- \Delta
    We fit a GP amongst all samples that fall in the \beta window.
    Finally we get the variance from the GP and select bottom-k targets with least variance.
    Finally minimize the bottom-k losses.
    """
    kwargs["GP_batch_idxs"] = batch_ids

    if constants.FORCE_FAR_DOSAGES not in kwargs.keys():
        far_dosage_CF = sample_far_dosages(
            dosage=batch_dosage_f, linear_delta=GI_linear_delta, **kwargs
        )  # There is no need to clamp the dosage_samples like we did for gradient interpolation
    else:
        far_dosage_CF = kwargs[constants.FORCE_FAR_DOSAGES]

    force_gp_sup = None
    if constants.FORCE_GP_SUPERVISION in kwargs.keys():
        force_gp_sup = kwargs[constants.FORCE_GP_SUPERVISION]

    gp_NND_CF: list = find_nnT_tgts(
        batch_dosage_delta=far_dosage_CF,
        linear_delta=GP_linear_delta,
        all_t=trn_dosages,
        all_y=trn_ys,
        method="GP",
    )

    batch_means_CF, batch_vars_CF = [], []
    # for fct_i in range(len(batch_ids)):
    for loop_idx, fct_id in enumerate(batch_ids):
        with torch.no_grad():
            gp_NND_ids = gp_NND_CF[loop_idx]
            gp_NND_emb = trn_embs[gp_NND_ids]

            gp_NND_y = trn_ys[gp_NND_ids]
            gp_NND_dosages = trn_dosages[gp_NND_ids]

            gp_NND_ymean = torch.mean(gp_NND_y)
            gp_NND_ystd = gp_NND_y - gp_NND_ymean

            GP_model = DNet.GP_NN()
            # gp_args = {constants.GP_KERNEL: gp_kernel, "model" :model, "T_f":gp_t, "X_f": gp_actual_x}
            gp_args = {constants.GP_KERNEL: gp_kernel}
            GP_model.forward(Z_f=gp_NND_emb, **gp_args)

            mean_w = GP_model.mean_w(gp_NND_ystd)  # [50, 471]
            factual_x_emb = batch_emb[loop_idx].detach().clone()
            factual_x_emb = F.normalize(factual_x_emb, p=2, dim=0)
            mean_CF = (
                torch.sum(mean_w.T * factual_x_emb, axis=-1).squeeze() + gp_NND_ymean
            )
            var_CF = (
                factual_x_emb.view(1, -1) @ GP_model.ker_inv @ factual_x_emb.view(-1, 1)
            )

            batch_means_CF.append(mean_CF)
            batch_vars_CF.append(var_CF)

    batch_means_CF = torch.stack(batch_means_CF).squeeze()
    batch_vars_CF = torch.stack(batch_vars_CF).squeeze()

    if btmk_var == constants.GNLL:
        if force_gp_sup is None:
            # In this case, we impose the Gaussian Negative Log Likelihood loss on the GP predictions. There is no top-k filtering.
            out_gp_cf = model.forward_with_emb(
                dosage=far_dosage_CF.squeeze(), x_emb=batch_emb
            )
            # TODO
            # Check if we need to control insane scaling of loss w.r.t. the least variance in the batch
            gp_loss = constants.GNLL_LOSS(
                input=out_gp_cf[1].squeeze(),
                target=batch_means_CF.squeeze(),
                var=batch_vars_CF.squeeze(),
            )
        else:
            gp_loss = constants.GNLL_LOSS_PEREX(
                input=force_gp_sup.squeeze(),
                target=batch_means_CF.squeeze(),
                var=batch_vars_CF.squeeze(),
            )
            return batch_means_CF, batch_vars_CF, gp_loss

    elif btmk_var == constants.SM:
        sm_temp = kwargs[constants.SM_TEMP]
        if force_gp_sup is None:
            out_gp_cf = model.forward_with_emb(
                dosage=far_dosage_CF.squeeze(), x_emb=batch_emb
            )
            gp_loss = tu.weighted_mse_loss(
                input=out_gp_cf[1].squeeze(),
                target=batch_means_CF,
                weight=F.softmax(
                    -batch_vars_CF / sm_temp, dim=0
                ),  # negate the variances so that low weight ones have high representation in the
            )
        else:
            weight = F.softmax(batch_vars_CF / sm_temp, dim=0)
            perex_loss = tu.mse_loss_perex(
                force_gp_sup.squeeze(), batch_means_CF.squeeze()
            )
            return batch_means_CF, batch_vars_CF, weight * perex_loss
    else:
        if force_gp_sup is None:
            btmk_idxs = tu.gumbel_topk(batch_vars_CF, k=btmk_var, largest=False)[1]
            x_emb_sel, batch_dosage_NND_sel, ycf_NND_sel = (
                batch_emb[btmk_idxs],
                far_dosage_CF[btmk_idxs].squeeze(),
                batch_means_CF[btmk_idxs],
            )
            out_gp_cf = model.forward_with_emb(
                dosage=batch_dosage_NND_sel, x_emb=x_emb_sel
            )
            gp_loss = models_helper.criterion(out=out_gp_cf, y=ycf_NND_sel)

        else:
            perex_loss = tu.mse_loss_perex(
                force_gp_sup.squeeze(), batch_means_CF.squeeze()
            )
            return batch_means_CF, batch_vars_CF, perex_loss

    if (
        constants.RETURN_VARIANCE in kwargs.keys()
        and kwargs[constants.RETURN_VARIANCE] == True
    ):
        return gp_loss, batch_vars_CF

    return gp_loss


def GP_unf_loss(
    *,
    model: DNet.Vcnet,
    batch_dosage_f,
    batch_dosage_grad,
    GI_linear_delta,
    GP_linear_delta,
    batch_ids,
    batch_emb: Tensor,
    trn_dosages,
    trn_ys,
    trn_embs,
    btmk_var,
    gp_kernel,
    **kwargs,
):
    """Given (xi, \beta_i, y_i), we sample random \betas away from U[0, 1]
    We fit a GP amongst all samples that fall outside the \delta window of the factual treatment.
    We fit GI is the \beta^Cf falls within \delta window of the factual treatment
    Finally we get the variance from the GP and select bottom-k targets with least variance.
    Finally minimize the bottom-k losses.
    """

    batch_dosage_CF = torch.rand(len(batch_ids)).to(cu.get_device())

    near_idxs = torch.where(
        torch.abs(batch_dosage_f - batch_dosage_CF) < GI_linear_delta
    )[0]
    far_idxs = torch.where(
        torch.abs(batch_dosage_f - batch_dosage_CF) >= GI_linear_delta
    )[0]

    # Impose GI loss on near_cf
    loss = 0
    if len(near_idxs) > 0:
        gi_loss = GI_reg_loss(
            model=model,
            batch_dosage=batch_dosage_f[near_idxs],
            batch_xemb=batch_emb[near_idxs],
            batch_y=trn_ys[batch_ids][near_idxs],
            batch_dosage_grad=batch_dosage_grad[near_idxs],
            linear_delta=GI_linear_delta,
            GI_num_explore=1,
            **kwargs,
        )
        loss = loss + gi_loss
    if len(far_idxs) > 0:
        gp_loss = GP_loss(
            model=model,
            batch_dosage_f=batch_dosage_f[far_idxs],
            GI_linear_delta=GI_linear_delta,
            GP_linear_delta=GP_linear_delta,
            batch_ids=batch_ids[far_idxs],
            batch_emb=batch_emb[far_idxs],
            trn_dosages=trn_dosages,
            trn_ys=trn_ys,
            trn_embs=trn_embs,
            btmk_var=btmk_var,
            gp_kernel=gp_kernel,
            **kwargs,
        )

        loss = loss + gp_loss
    return loss


def run_GIKS(
    *,
    dataset_name,
    dataset_nums: list,
    num_epoch,
    logger: logging.Logger,
    suffix="",
    **kwargs,
):
    """Run our method that involves gradient interpolation and kernel smoothing

    Steps:
        1. Assume i is index over the samples and j indexes counterfactuals.
        2. i.e. D = {(x_i, \beta_i, y_i)} and D_i^CF = {(x_j, \beta_i, y_j^CF)} where D_i^CF is the ctr factual dataset for \beta_i
        3. Get y_j^CF from vcnet for each \beta_i. We precompute the embeddings
        4. Fit the GP to estimate the factual lklhood
    """
    wd = 5e-3
    lrn_rate = kwargs[constants.LRN_RATE]
    gi_lambda = kwargs[constants.GI_LAMBDA]
    pretrn = kwargs[constants.PRETRN_EPOCHS]
    bsz = kwargs[constants.BATCH_SIZE]

    need_gp = kwargs[constants.NEED_FAR_GP]
    gp_lambda = kwargs[constants.GP_LAMBA]
    far_ctr_epochs = kwargs[constants.TRIGGER_FAR_CTR]
    btmk_var = kwargs[constants.BTM_K_VAR]
    sm_temp = kwargs[constants.SM_TEMP]
    hpm_tuning = kwargs[constants.HPM_TUNING]
    log_val_mse = kwargs[constants.LOG_VAL_MSE]

    gp_kernel = kwargs[constants.GP_KERNEL]

    GI_linear_delta = kwargs[
        constants.GI_LINEAR_DELTA
    ]  # TCGA is dense in treaatments and so 0.05 is adequate # kwargs[constants.LINEAR_DELTA]
    GI_num_explore = kwargs[constants.NUM_SAMPLES_LINDELTA]

    GP_linear_delta = kwargs[constants.GP_LINEAR_DELTA]

    gp_unf_t = kwargs[
        constants.GP_UNF_T
    ]  # Should we sample the counterfactuals uniformly during the GP phase?
    only_far_ctr = kwargs[constants.ONLY_FAR_CTR]
    ctr_sampling_dist = kwargs[constants.CTR_SAMPLING_DIST]

    results_path: Path = kwargs[constants.RESULTS_PATH]
    start_epochs = (
        kwargs[constants.START_EPOCHS]
        if kwargs[constants.START_EPOCHS] is not None
        else 0
    )
    num_epoch = start_epochs + num_epoch

    for dataset_num in dataset_nums:
        END_SEED = False

        logger.info(f"------Start_Seed_{dataset_num}-----")

        # Load the dataset and initiate early stopping with validation split
        (
            all_matrix,
            test_matrix,
            t_grid,
            indim,
            data_class,
            es,
        ) = du.get_dataset_seed(
            dataset_name, dataset_num, num_epoch, suffix, results_path
        )

        # Spli the dataset into train and val
        es: ES.EarlyStopping = es
        trn_idxs, val_idxs = es.train_idxs, es.val_idxs

        train_matrix = all_matrix[trn_idxs]
        val_matrix = all_matrix[val_idxs]
        val_dosages, val_xs, val_ys = constants.matrix_to_txy(
            val_matrix, cu.get_device()
        )

        trn_dosages, trn_xs, trn_ys = constants.matrix_to_txy(
            train_matrix, cu.get_device()
        )
        trn_dosages_grad = trn_dosages.requires_grad_(True)

        train_loader = cont_data.get_iter(train_matrix, batch_size=bsz, shuffle=True)

        test_dosages, test_xs, test_ys = constants.matrix_to_txy(
            test_matrix, cu.get_device()
        )

        num_data = len(trn_xs)
        kwargs["num_data"] = num_data

        # This is to track the augmented treatments, noting to do with the execution
        augmented_new_t = defaultdict(list)

        # import model
        model = du.load_model(
            dataset_name=dataset_name,
            dataset_num=dataset_num,
            indim=indim,
            results_path=results_path,
            **kwargs,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lrn_rate, weight_decay=wd)

        floss_hist = []
        tloss_hist = []  # t stands for total

        """Tuning the HPMs using a model that is converged on factual loss for upto 200 epochs
        """
        GP_param_tuning = False
        if hpm_tuning == True:
            model.train()
            # For each example in the validation dataset, get the GP variance
            factual_args = {constants.RETURN_EMB: True}
            val_out = model.forward(dosage=val_dosages, x=val_xs, **factual_args)
            val_embs = val_out[2]
            val_y_preds = val_out[1]
            GP_param_tuning = kwargs[constants.GP_PARAMS_TUNING]

        tq_epochs = tqdm(total=num_epoch, desc="floss", position=0, leave=False)
        for epoch in range(start_epochs, num_epoch):
            if END_SEED == True:
                break

            tq_loader = tqdm(
                total=len(train_loader), desc="floss", position=1, leave=False
            )
            floss_epoch, tloss_epoch = [], []

            for epoch_step, (batch_inp, batch_y, batch_ids) in enumerate(train_loader):
                batch_dosage_grad: Tensor = (
                    batch_inp[:, 0]
                    .to(cu.get_device(), dtype=torch.float64)
                    .requires_grad_(True)
                )

                batch_dosage = (
                    batch_dosage_grad.detach().clone()
                )  # This is treatment without gradient

                batch_y = batch_y.to(cu.get_device(), dtype=torch.float64)
                batch_ids = batch_ids.to(cu.get_device(), dtype=torch.int64)

                optimizer.zero_grad()
                model.train()

                factual_args = {constants.RETURN_EMB: True}
                trn_out = model.forward(
                    dosage=trn_dosages_grad, x=trn_xs, **factual_args
                )  # pass gradient enabled T to model and make the predictions
                trn_embs = trn_out[2]
                trn_y_preds = trn_out[1]

                batch_emb: Tensor = trn_embs[batch_ids]
                batch_y_preds = trn_y_preds[batch_ids]

                floss = nn.MSELoss()(batch_y_preds.squeeze(), batch_y.squeeze())
                loss = floss
                loss_str = f"fl:{round(floss.item(), 3)}"

                gigp_args = {
                    "aug_dict": augmented_new_t,
                    "batch_idxs": batch_ids.cpu().numpy(),
                    "global_trn_ids": trn_idxs,
                    constants.CTR_SAMPLING_DIST: ctr_sampling_dist,
                    constants.DATASET: dataset_name,
                    constants.DATASET_NUM: dataset_num,
                    "all_trn_dosage": trn_dosages,
                    constants.SM_TEMP: sm_temp,
                }
                gi_loss = torch.tensor(0)

                if hpm_tuning == True and epoch_step == 0 and epoch == start_epochs:
                    gigp_args[constants.FORCE_FAR_DOSAGES] = val_dosages.view(-1, 1)
                    gigp_args[constants.RETURN_VARIANCE] = True
                    gigp_args[constants.SM_TEMP] = sm_temp
                    gigp_args[constants.FORCE_GP_SUPERVISION] = val_ys

                    if kwargs[constants.GI_DELTA_TUNING] == True:
                        # TODO
                        # Add factual loss also to the logs here

                        fct_losses, gi_losses, gp_losses = GP_unf_valloss(
                            model=model,
                            val_dosages=val_dosages,
                            GI_linear_delta=GI_linear_delta,
                            GP_linear_delta=GP_linear_delta,
                            val_xemb=val_embs,
                            val_ys=val_ys,
                            trn_dosages=trn_dosages,
                            trn_ys=trn_ys,
                            trn_embs=trn_embs,
                            btmk_var=btmk_var,
                            gp_kernel=gp_kernel,
                            **gigp_args,
                        )

                        logger.info(
                            f"Seed: {dataset_num} \t GI_linear_delta: {GI_linear_delta} \t fct_losses: {torch.mean(fct_losses).item()} \t  gi_losses: {torch.mean(gi_losses).item()} \t  ks_losses: {torch.mean(gp_losses).item()}"
                        )
                        print(
                            f"Seed: {dataset_num} \t GI_linear_delta: {GI_linear_delta} \t fct_losses: {torch.mean(fct_losses).item()} \t  gi_losses: {torch.mean(gi_losses).item()} \t  ks_losses: {torch.mean(gp_losses).item()}"
                        )

                        END_SEED = True
                        break

                    val_means, val_vars, gp_losses = GP_loss(
                        model=model,
                        batch_dosage_f=val_dosages,
                        GI_linear_delta=GI_linear_delta,
                        GP_linear_delta=GP_linear_delta,
                        batch_ids=torch.arange(
                            len(val_dosages), device=cu.get_device()
                        ),  # this is a dummy
                        batch_emb=val_embs,
                        trn_dosages=trn_dosages,
                        trn_ys=trn_ys,
                        trn_embs=trn_embs,
                        btmk_var=btmk_var,
                        gp_kernel=gp_kernel,
                        **gigp_args,
                    )

                    # normalize val err
                    val_weights = torch.sqrt(val_vars) / torch.sum(torch.sqrt(val_vars))

                    # If we are tuning GP params, we juct compute based on the GP predictions
                    # and no training needs to be done
                    if GP_param_tuning == True:
                        avg_err = round(torch.mean(gp_losses).item(), 3)
                        wtd_err = round(torch.sum(gp_losses * val_weights).item(), 3)

                        logger.info(f"HPM wtd validation MSE: {wtd_err}")
                        logger.info(f"HPM avg validation MSE: {avg_err}")
                        END_SEED = True
                        break

                    # remove the key for getting variance
                    del gigp_args[constants.RETURN_VARIANCE]
                    del gigp_args[constants.FORCE_FAR_DOSAGES]
                    del gigp_args[constants.FORCE_GP_SUPERVISION]

                # Running the Gradient interpolation loss
                if only_far_ctr == False and epoch >= pretrn:
                    # Do we need explicit GI loss during the third phase?
                    if (
                        need_gp == True
                        and (epoch + 1) >= far_ctr_epochs
                        and gp_unf_t == True
                    ):
                        pass

                    else:
                        gi_loss = GI_reg_loss(
                            model=model,
                            batch_dosage=batch_dosage,
                            batch_dosage_grad=batch_dosage_grad,
                            batch_xemb=batch_emb,
                            batch_y=batch_y,
                            GI_num_explore=GI_num_explore,
                            linear_delta=GI_linear_delta,
                            **gigp_args,
                        )

                    loss = loss + gi_lambda * gi_loss
                    loss_str = f"fl:{round(floss.item(), 4)}\t gi:{round((gi_lambda*gi_loss).item(), 4)}"

                if need_gp and (epoch + 1) >= far_ctr_epochs:
                    if gp_unf_t == True:
                        gp_loss = GP_unf_loss(
                            model=model,
                            batch_dosage_f=batch_dosage,
                            batch_dosage_grad=batch_dosage_grad,
                            GI_linear_delta=GI_linear_delta,
                            GP_linear_delta=GP_linear_delta,
                            batch_ids=batch_ids,
                            batch_emb=batch_emb,
                            trn_dosages=trn_dosages,
                            trn_ys=trn_ys,
                            trn_embs=trn_embs,
                            btmk_var=btmk_var,
                            gp_kernel=gp_kernel,
                            **gigp_args,
                        )
                    else:
                        gp_loss = GP_loss(
                            model=model,
                            batch_dosage_f=batch_dosage,
                            GI_linear_delta=GI_linear_delta,
                            GP_linear_delta=GP_linear_delta,
                            batch_ids=batch_ids,
                            batch_emb=batch_emb,
                            trn_dosages=trn_dosages,
                            trn_ys=trn_ys,
                            trn_embs=trn_embs,
                            btmk_var=btmk_var,
                            gp_kernel=gp_kernel,
                            **gigp_args,
                        )

                    loss = loss + gp_lambda * gp_loss
                    if only_far_ctr == False:
                        loss_str = f"fl:{round(floss.item(), 4)}\t gi:{round((gi_lambda*gi_loss).item(), 4)}\t gp:{round((gp_lambda*gp_loss).item(), 4)}"
                    else:
                        loss_str = f"fl:{round(floss.item(), 4)}\t gp:{round((gp_lambda*gp_loss).item(), 4)}"

                floss_epoch.append(floss.item())
                tloss_epoch.append(loss.item())

                loss.backward()
                optimizer.step()

                tq_loader.set_description("  ".join(loss_str.split()))
                tq_loader.update(1)

            floss_hist.extend(floss_epoch)
            tloss_hist.extend(tloss_epoch)
            tq_epochs.set_description(
                f"floss:{round(np.mean(floss_epoch), 4)}, tloss:{round(np.mean(tloss_epoch), 4)}"
            )
            tq_epochs.update(1)

            # Invoke a call to early stopping to dump the best model
            es.log_val_metrics(
                model=model,
                train_floss=floss_epoch,
                train_tloss=tloss_epoch,
                epoch=epoch,
                logger=logger,
            )

        logger.info(f"---------End_Epoch{dataset_num}------")

        # Load the best model now
        logger.info(f"Best validation epoch -- {es._best_epoch}")
        model.load_state_dict(
            torch.load(
                es.models_dir / f"best_val_model-{dataset_num}.pt",
                map_location=cu.get_device(),
            )
        )

        if log_val_mse == True:
            best_val_mse = es.get_val_perf(model=model, weights=val_weights)
            logger.info(f"HPM wtd validation MSE: {best_val_mse}")
            best_val_mse = es.get_val_perf(model=model, weights=None)
            logger.info(f"HPM avg validation MSE: {best_val_mse}")

        if dataset_name in [constants.IHDP_CONT, constants.NEWS_CONT]:
            if kwargs[constants.HPM_TUNING] == False:
                _, pehe = cont_eval.curve(model, test_matrix, t_grid, targetreg=None)
                test_mse = cont_eval.predictive_mse(model, test_matrix, targetreg=None)
                logger.info(f"Last model False \t pehe: {pehe}\t test_mse: {test_mse}")
                print(f"Last model False \t pehe: {pehe}\t test_mse: {test_mse}")
            else:
                test_mse = cont_eval.predictive_mse(model, test_matrix, targetreg=None)
                logger.info(f"Last model False \t test_mse: {test_mse}")
        else:
            if kwargs[constants.HPM_TUNING] == False:
                mise, dpe, ite = compute_eval_metrics(
                    dataset_name=dataset_name,
                    dataset=data_class.dataset,
                    test_patients=test_xs,
                    num_treatments=1,
                    model=model,
                    train=False,
                )
                test_mse = cont_eval.predictive_mse(model, test_matrix, targetreg=None)
                logger.info(
                    f"Last model False -- mise :: {mise} -- dpe :: {dpe} -- ite :: {ite} -- test_mse: {test_mse}"
                )
                print(
                    f"Last model False :: {epoch}\t mise :: {mise}\t dpe :: {dpe}\t ite :: {ite}\t test_mse :: {test_mse}"
                )
            else:
                test_mse = cont_eval.predictive_mse(model, test_matrix, targetreg=None)
                logger.info(f"Last model False \t test_mse: {test_mse}")

        with open(
            results_path / "pkl" / suffix / f"{suffix}-{dataset_num}.pkl", "wb"
        ) as file:
            pkl.dump(augmented_new_t, file)
