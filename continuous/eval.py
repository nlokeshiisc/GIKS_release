import torch
import numpy as np
import json
from continuous.data_helper import get_iter
from utils import common_utils as cu

def curve(model, test_matrix, t_grid, targetreg=None):
    model.eval()
    test_matrix = test_matrix.to(cu.get_device(), dtype=torch.float64)
    t_grid = t_grid.to(cu.get_device(), dtype=torch.float64)

    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test).to(cu.get_device(), dtype=torch.float64)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    with torch.no_grad():
        if targetreg is None:
            for _ in range(n_test):
                for idx, (inputs, y, batch_ids) in enumerate(test_loader):
                    t = inputs[:, 0]
                    t *= 0
                    t += t_grid[0, _]
                    x = inputs[:, 1:]
                    break
                out = model.forward(t, x)
                out = out[1].data.squeeze()
                out = out.mean()
                t_grid_hat[1, _] = out
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
            return t_grid_hat, mse
        else:
            for _ in range(n_test):
                for idx, (inputs, y, batch_ids) in enumerate(test_loader):
                    t = inputs[:, 0]
                    t *= 0
                    t += t_grid[0, _]
                    x = inputs[:, 1:]
                    break
                out = model.forward(t, x)
                tr_out = targetreg(t).data
                g = out[0].data.squeeze()
                out = out[1].data.squeeze() + tr_out / (g + 1e-6)
                out = out.mean()
                t_grid_hat[1, _] = out
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
            return t_grid_hat.cpu().numpy(), mse

def predictive_mse(model, test_matrix, targetreg=None):
    """Computes the predictive mse of the model
    the above function simply evaluates the mse of the average treatment effect

    Args:
        model (_type_): _description_
        test_matrix (_type_): _description_
        targetreg (_type_): _description_
    """
    model.eval()
    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    with torch.no_grad():
        if targetreg is None:
            for idx, (inputs, y, batch_ids) in enumerate(test_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                break
            t, x, y = t.to(cu.get_device(), dtype=torch.float64), x.to(cu.get_device(), dtype=torch.float64), y.to(cu.get_device(), dtype=torch.float64)
            out = model.forward(t, x)
            out = out[1].data.squeeze()
            mse = ((y.squeeze() - out.squeeze()) ** 2).mean().data
            return mse
        else:
            for idx, (inputs, y, batch_ids) in enumerate(test_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                break
            t, x, y = t.to(cu.get_device(), dtype=torch.float64), x.to(cu.get_device(), dtype=torch.float64), y.to(cu.get_device(), dtype=torch.float64)
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            mse = ((y.squeeze() - out.squeeze()) ** 2).mean().data
            return mse