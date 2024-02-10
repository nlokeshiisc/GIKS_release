import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from functorch import make_functional, vmap, vjp, jvp, jacrev
import GPUtil


def init_weights(m: nn.Module):

    def set_params(w):
        if isinstance(w, nn.Linear):
            torch.nn.init.xavier_uniform(w.weight)
            try:  # bias may or maynot exist
                w.bias.data.fill_(0.01)
            except:
                pass
    m.apply(set_params)


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer
    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler
    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError(
            "Other learning rate schedulers are not implemented")


def _sel_nzro(self, t, sij):
    def sel_nonzero(t, sij): return torch.squeeze(t[torch.nonzero(sij)])
    res = sel_nonzero(t, sij)
    if res.dim() == t.dim()-1:
        res = torch.unsqueeze(res, 0)
    return res


def _sel_zro(self, t, sij):
    def sel_zero(t, sij): return torch.squeeze(1-t[torch.nonzero(sij)])
    res = sel_zero(t, sij)
    if res.dim() == t.dim()-1:
        res = torch.unsqueeze(res, 0)
    return res


def adjust_learning_rate(optimizer, init_lr, lr_type, epoch, total_epochs):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / total_epochs))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def jacobian(x1, t1, **kwargs):
    # Compute J(x1)
    fnet, params = make_functional(kwargs["model"])

    def fnet_single(params, t, x):
        return fnet(params, t, x, is_NTK=True)[1].squeeze(0)

    jac1 = jacrev(fnet_single)(params, t1[0].unsqueeze(0), x1[0].unsqueeze(0))
    d1 = torch.Tensor([]).to(x1.get_device())
    for idx in range(len(jac1)):
        d1 = torch.cat((d1, jac1[idx].flatten().float()))
    d1 = d1.unsqueeze(0)

    for i in range(1, len(x1)):
        jac1 = jacrev(fnet_single)(
            params, t1[i].unsqueeze(0), x1[i].unsqueeze(0))
        d = torch.Tensor([]).to(x1.get_device())
        for idx in range(len(jac1)):
            d = torch.cat((d, jac1[idx].flatten().float()))
        d = d.unsqueeze(0)
        d1 = torch.cat((d1, d), 0)

    phi = d1.T

    return phi


def gumbel(unf_noise):
    return -torch.log(-torch.log(unf_noise))


def gumbel_topk(a: torch.Tensor, k, largest):
    if k < 1:
        k = int(k * a.shape[0])
    anoise = a + gumbel(torch.rand_like(a))
    return torch.topk(anoise, k, largest=largest)


def partition_idxs(arr: torch.Tensor, test_pc: float):
    """This is a simple torch utility that partitons the indices to train and test

    Args:
        arr (torch.Tensor): _description_
    Returns:
        Train indices and test indices in this order
    """
    num_samples = arr.shape[0]
    all_idxs = torch.arange(num_samples)
    num_subset = int(num_samples * test_pc)
    return all_idxs[num_subset:], all_idxs[:num_subset]


def weighted_mse_loss(input, target, weight):
    # Implements a weighted version of mean squared error as loss function
    return torch.sum(weight * (input - target) ** 2)


def mse_loss_perex(input, target):
    # Implements a weighted version of mean squared error as loss function
    return (input - target) ** 2


def get_available_gpus():
    gpus = GPUtil.getGPUs()
    GPUavailability = GPUtil.getAvailability(
        gpus, maxLoad=0.5, maxMemory=0.8, includeNan=False, excludeID=[], excludeUUID=[])
    # return a ransom sample "1" in GPUavailability
    return np.random.choice(np.where(np.asarray(GPUavailability) == 1)[0], 1)[0]
