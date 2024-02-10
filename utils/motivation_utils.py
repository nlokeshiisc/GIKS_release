import torch
import os

from tqdm import tqdm
import continuous.dynamic_net as DNet
import continuous.data_helper as cont_data
import utils.torch_utils as tu
import utils.common_utils as cu
import constants as constants
from continuous.main_helper import find_nnT_tgts
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind

from pathlib import Path

this_dir = Path(".").absolute()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pickle as pkl
import pandas as pd

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
import numpy as np
from dataset.generate.generate_simu1 import t_x_y, simu_ctr_data, simu_data1


def plot_bins(num_bins, X0, X1, t):
    t_bins = []
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "pink",
        "black",
        "orange",
        "purple",
        "beige",
        "brown",
        "gray",
        "cyan",
        "magenta",
    ]
    for _, bin_entry in enumerate(range(num_bins)):
        bin_a, bin_b = bin_entry / num_bins, (bin_entry + 1) / num_bins
        t_bin = torch.LongTensor(
            [1 if entry >= bin_a and entry <= bin_b else 0 for entry in t]
        )
        t_bins.append(torch.where(t_bin == 1)[0])

        print(bin_a, bin_b, len(t_bins[-1]))
        t_bin_idx = t_bins[-1]

        plt.scatter(X0[t_bin_idx], X1[t_bin_idx], c=colors[_])
        plt.title(f"t in [{bin_a}, {bin_b}]")
        plt.show()


# %%
def plot_train_support(train_matrix, x_coords):
    """Plots the treatment support in the raw training dataset"""
    X0, X1 = train_matrix[:, x_coords[0]], train_matrix[:, x_coords[1]]
    t = train_matrix[:, 0]
    plot_bins(num_bins=5, X0=X0, X1=X1, t=t)


def create_ctr_dataset(dataset_name, train_matrix, fct_matrix, uniform_t, test_idxs, **kwargs):
    
    if dataset_name == constants.SYNTHETIC_CONT:
        if uniform_t == True:
            num_ctr = kwargs["num_ctr"]

        ctr_y, ctr_t = simu_ctr_data(fct_matrix=fct_matrix, uniform_t=uniform_t, **kwargs)
        ctr_y = torch.Tensor(ctr_y)
        ctr_t = torch.Tensor(ctr_t)

        ctr_train_matrix = torch.repeat_interleave(train_matrix, len(ctr_t), dim=0)
        ctr_train_matrix[:, 0] = ctr_t.view(-1, 1).repeat(len(fct_matrix), 1).squeeze()
        ctr_train_matrix[:, -1] = torch.Tensor(ctr_y).view(-1, 1).squeeze()
        ctr_loader = cont_data.get_iter(ctr_train_matrix, batch_size=500, shuffle=False)
        
    elif dataset_name == constants.IHDP_CONT:
        num_integration_samples = 64
        t_samples = torch.arange(0.01, 1, 1 / 64).to(cu.get_device())
        test_idxs = torch.LongTensor(test_idxs).to(cu.get_device())

        with open(
            constants.fcf_dir  / "dataset/ihdp/tr_h_1.0_te_l_0.0_h1.0/ihdp_response.pkl",
            "rb",
        ) as file:
            ihdp_response = pkl.load(file)
        ihdp_response = torch.Tensor(ihdp_response).to(cu.get_device())
        
        all_dosage_test, all_x_test, all_y_test = (
            train_matrix[:, 0].to(cu.get_device(), dtype=torch.float64),
            train_matrix[:, 1:-1].to(cu.get_device(), dtype=torch.float64),
            train_matrix[:, -1].to(cu.get_device(), dtype=torch.float64),
        )
        
        ctr_train_matrix = torch.repeat_interleave(train_matrix, num_integration_samples, dim=0)
        ctr_train_matrix[:, 0] = t_samples.repeat(all_x_test.shape[0])
        ctr_train_matrix[:, -1] = ihdp_response[test_idxs].view(-1)
        ctr_loader = cont_data.get_iter(ctr_train_matrix, batch_size=500, shuffle=False)
        
    return ctr_train_matrix, ctr_loader


# %%
def predict_y(model, loader, targetreg=None):
    pred_y = []
    with torch.no_grad():
        if targetreg is None:
            for idx, (inputs, y, batch_ids) in enumerate(loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]

                t, x, y = (
                    t.to(cu.get_device(), dtype=torch.float64),
                    x.to(cu.get_device(), dtype=torch.float64),
                    y.to(cu.get_device(), dtype=torch.float64),
                )
                out = model.forward(t, x)
                out = out[1].data.squeeze()
                pred_y.append(out.cpu())
        else:
            for idx, (inputs, y, batch_ids) in enumerate(loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                t, x, y = (
                    t.to(cu.get_device(), dtype=torch.float64),
                    x.to(cu.get_device(), dtype=torch.float64),
                    y.to(cu.get_device(), dtype=torch.float64),
                )
                out = model.forward(t, x)
                tr_out = targetreg(t).data
                g = out[0].data.squeeze()
                out = out[1].data.squeeze() + tr_out / (g + 1e-6)
                # mse = ((y.squeeze() - out.squeeze()) ** 2).mean().data
                pred_y.append(out.cpu())

    return pred_y


def predict_emb(model, loader):
    """Predicts the embeddings on the dataset"""
    embs = []
    with torch.no_grad():
        for idx, (inputs, y, batch_ids) in enumerate(loader):
            t = inputs[:, 0]
            x = inputs[:, 1:]

            t, x, y = (
                t.to(cu.get_device(), dtype=torch.float64),
                x.to(cu.get_device(), dtype=torch.float64),
                y.to(cu.get_device(), dtype=torch.float64),
            )
            fwd_args = {constants.RETURN_EMB: True}
            out = model.forward(t, x, **fwd_args)
            out = out[2].data.squeeze()
            embs.append(out.cpu())
    return torch.vstack(embs)


# %% [markdown]
# ### Create the VCNET model

# %%
def create_vcnet(indim):
    cfg_density = [(indim, 50, 1, "relu"), (50, 50, 1, "relu")]
    cfg = [(50, 50, 1, "relu"), (50, 1, 1, "id")]
    num_grid = 10
    degree = 2
    knots = [0.33, 0.66]
    model = DNet.Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model._initialize_weights()
    model.to(cu.get_device(), dtype=torch.float64)
    return model


def GP_preds(
    model: DNet.Vcnet,
    factual_t,
    factual_y,
    factual_x,
    ctr_t,
    linear_delta=0.05,
    gp_kernel=constants.COSINE_KERNEL,
    btmk_var=0.05,
    gihelpsgp=False,
    ctr_loader=None,
    **kwargs,
):

    batch_ids = torch.arange(len(factual_x))
    factual_emb = predict_emb(model=model, loader=ctr_loader)

    # Find the nearest neighbors in \delta window
    gp_NND_CF: list = find_nnT_tgts(
        batch_dosage_delta=ctr_t,
        linear_delta=linear_delta,
        all_t=factual_t,
        all_y=factual_y,
        method="GP",
    )

    batch_means_CF, batch_vars_CF = [], []

    for fct_id in tqdm(batch_ids):
        for ctt_idx, ctt in enumerate(ctr_t):
            # Make prediction for (x[fct_id], ctt)
            with torch.no_grad():
                gp_NND_ids = gp_NND_CF[ctt_idx]
                gp_NND_emb = factual_emb[gp_NND_ids].to(cu.get_device())

                gp_NND_y = factual_y[gp_NND_ids]

                # Mean subtraction for GP
                gp_NND_ymean = torch.mean(gp_NND_y)
                gp_NND_ystd = gp_NND_y - gp_NND_ymean

                GP_model = DNet.GP_NN()
                gp_args = {constants.GP_KERNEL: gp_kernel}
                GP_model.forward(Z_f=gp_NND_emb, **gp_args)

                mean_w = GP_model.mean_w(
                    gp_NND_ystd.to(cu.get_device(), dtype=torch.float64)
                )  # [50, 471]
                factual_x_emb = factual_emb[fct_id].detach().clone()
                factual_x_emb = factual_x_emb / torch.norm(factual_x_emb)
                mean_CF = (
                    torch.sum(
                        mean_w.T * factual_x_emb.to(cu.get_device()), axis=-1
                    ).squeeze()
                    + gp_NND_ymean
                )
                var_CF = (
                    factual_x_emb.view(1, -1).to(cu.get_device())
                    @ GP_model.ker_inv
                    @ factual_x_emb.view(-1, 1).to(cu.get_device())
                )

                batch_means_CF.append(mean_CF)
                batch_vars_CF.append(var_CF)
    return torch.stack(batch_means_CF), torch.stack(batch_vars_CF)


def export_legend(filename="legend.png", fig=None):
    legend = fig.legend(fontsize="15", ncol=2, loc="upper right", bbox_to_anchor=(1, 1))
    fig2  = legend.figure
    fig2.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    fig2.savefig(filename, dpi=300, bbox_inches=bbox, facecolor="w")
    legend.remove()


def plot_pdfs(
    values_1,
    values_2,
    values_3,
    values_4,
    label1=None,
    label2=None,
    xlabel=None,
    ylabel=None,
    save_name=None,
):
    # libraries & dataset
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.cla()
    plt.clf()

    f, axarr = plt.subplots(1, 2, sharey=True)

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="whitegrid")
    
    # c1 = #EA2027, c2 = #0652DD

    fig = sns.kdeplot(
        values_1, shade=True, color="#EA2027", label=label1, linewidth=3, ax=axarr[0]
    )
    fig = sns.kdeplot(
        values_2, shade=True, color="#0652DD", label=label2, linewidth=3, ax=axarr[0]
    )
    
    # plt.legend(fontsize=18, loc ="lower left")
    # export_legend(filename="legend.pdf", fig=plt.gcf())
    
    fig = sns.kdeplot(
        values_3, shade=True, color="#EA2027", linewidth=3, ax=axarr[1]
    )
    fig = sns.kdeplot(
        values_4, shade=True, color="#0652DD", linewidth=3, ax=axarr[1]
    )

    f.text(0.5, 0.04, xlabel, ha="center", fontsize=15)
    axarr[0].set_ylabel(ylabel,fontsize=15)
    
    plt.savefig(save_name, format="pdf", bbox_inches="tight", dpi=300)


def plot_topk_btmk_dosage_bins(
    values_1,
    values_2,
    dosages_1,
    dosages_2,
    label1=None,
    label2=None,
    xlabel=None,
    ylabel=None,
    save_name=None,
):
    # libraries & dataset
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    bins = np.arange(6) * 0.2
    dbins_1 = np.digitize(x=dosages_1, bins=bins, right=True)
    dbins_2 = np.digitize(x=dosages_2, bins=bins, right=True)

    # %% matplotlib code

    v1_mean = [
        np.mean(values_1[np.where(dbins_1 == entry)[0]]) for entry in range(1, 6)
    ]
    v2_mean = [
        np.mean(values_2[np.where(dbins_2 == entry)[0]]) for entry in range(1, 6)
    ]

    print([len(values_1[np.where(dbins_1 == entry)[0]]) for entry in range(1, 6)])
    print([len(values_2[np.where(dbins_2 == entry)[0]]) for entry in range(1, 6)])

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(bins) - 1)
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(
        r1,
        v1_mean,
        # yerr=[[0] * len(v1_std), v1_std],
        color="#EA2027",
        width=barWidth,
        edgecolor="white",
        label=label1,
        # capsize=3,
    )
    plt.bar(
        r2,
        v2_mean,
        # yerr=[[0] * len(v2_std), v2_std],
        color="#0652DD",
        width=barWidth,
        edgecolor="white",
        label=label2,
        # capsize=3,
    )

    plt.xticks(
        [r for r in range(len(bins) - 1)],
        ["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1]"],
    )
    # plt.grid(True)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_name, format="pdf", bbox_inches="tight", dpi=300)

  

def plot_dosage_bins(
    values_1,
    values_2,
    dosages,
    label1=None,
    label2=None,
    xlabel=None,
    ylabel=None,
    save_name=None,
):
    # libraries & dataset
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.clf()
    plt.cla()

    bins = np.arange(6) * 0.2
    d_bins = np.digitize(x=dosages, bins=bins, right=True)

    # %% matplotlib code

    v1_mean = [np.mean(values_1[np.where(d_bins == entry)[0]]) for entry in range(1, 6)]
    v2_mean = [np.mean(values_2[np.where(d_bins == entry)[0]]) for entry in range(1, 6)]

    num_entries = np.array(
        [len(values_1[np.where(d_bins == entry)[0]]) for entry in range(1, 6)]
    )
    num_entries = num_entries / np.sum(num_entries)
    num_entries = num_entries[[0, 1, 3, 2, 4]]

    v1_std = [np.std(values_1[np.where(d_bins == entry)[0]]) for entry in range(1, 6)]
    v2_std = [np.std(values_2[np.where(d_bins == entry)[0]]) for entry in range(1, 6)]

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(bins) - 1)
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(
        r1,
        v1_mean,
        # yerr=[[0] * len(v1_std), v1_std],
        color="red",
        width=barWidth,
        edgecolor="white",
        label=label1,
        alpha=0.7
        # capsize=3,
    )
    plt.bar(
        r2,
        v2_mean,
        # yerr=[[0] * len(v2_std), v2_std],
        color="blue",
        width=barWidth,
        edgecolor="white",
        label=label2,
        alpha=0.7
        # capsize=3,
    )
    plt.plot(
        np.array(r2) - 0.12,
        num_entries,
        color="#00FF7F",
        linewidth=3,
        marker="*",
        markersize=15,
        alpha=1
    )

    plt.xticks(
        [r for r in range(len(bins) - 1)],
        ["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1]"],
    )
    # plt.grid(True)

    # plt.legend(fontsize="15", ncol=2, loc="upper right", bbox_to_anchor=(1, 1))
    # export_legend(filename="legend.pdf", fig=plt.gcf())
    
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.savefig(save_name, format="pdf", bbox_inches="tight", dpi=300)
    pass


def t_test(x, y, alternative="lesser"):
    # Code taken from: https://stackoverflow.com/questions/15984221/how-to-perform-two-sample-one-tailed-t-test-with-numpy-scipy
    if type(x) == torch.Tensor:
        x = x.numpy()
    if type(y) == torch.Tensor:
        y = y.numpy()
    _, double_p = ttest_ind(x, y, equal_var=False)
    if alternative == "both-sided":
        pval = double_p
    elif alternative == "greater":
        if np.mean(x) > np.mean(y):
            pval = double_p / 2.0
        else:
            pval = 1.0 - double_p / 2.0
    elif alternative == "lesser":
        if np.mean(x) < np.mean(y):
            pval = double_p / 2.0
        else:
            pval = 1.0 - double_p / 2.0
    return pval
