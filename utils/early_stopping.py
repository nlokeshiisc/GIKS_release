from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pickle as pkl
from pathlib import Path
import torch.nn as nn

# from traitlets import default
import constants
import torch


class EarlyStopping:
    def __init__(
        self,
        all_x,
        all_y,
        all_d,
        all_t,
        val_pc,
        results_dir,
        suffix,
        num_epochs,
        seed,
        task="reg",
        **kwargs,
    ):
        """This is the early stopping class that tracks the training progress
        results_dir = root directory where we dump logs, models etc.

        We assume that we create one Early Stopping object for every dataset seed

        IMPORTANT:
            For single treatment case, we consider t to reflect the dosage. So this must be handled \
            while instantiating the class

        Args:
            all_x (_type_): _description_
            all_y (_type_): _description_
            all_d (_type_): _description_
            all_t (_type_): _description_
            val_pc: Percentage of samples to be in the validation dataset
            results_dir (_type_): _description_
        """
        self.all_x = all_x
        self.all_y = all_y
        self.all_d = all_d
        self.all_t = all_t
        if self.all_t is None:
            self.all_t = torch.zeros_like(
                self.all_y
            )  # This is for the Single Treatment case

        self.val_pc = val_pc

        self.task = task

        # Use this if you want to force the validation dataset
        if "force_val_idxs" in kwargs.keys() and kwargs["force_val_idxs"] is not None:
            self.val_idxs = torch.Tensor(kwargs["force_val_idxs"]).to(
                device=all_x.device, dtype=torch.int64
            )
            self.train_idxs = torch.Tensor(kwargs["force_trn_idxs"]).to(
                device=all_x.device, dtype=torch.int64
            )
        else:
            self.train_idxs, self.val_idxs = self.data_split(
                all_x=all_x, all_y=all_y, val_fraction=val_pc
            )

        self.val_x = self.all_x[self.val_idxs]
        self.val_t = self.all_t[self.val_idxs]
        self.val_d = self.all_d[self.val_idxs]
        self.val_y = self.all_y[self.val_idxs]

        self.suffix = suffix
        self.num_epochs = num_epochs

        self.val_metrics = []
        self.epochs = []
        self.train_floss = []
        self.train_tloss = []
        self.results_dir: Path = results_dir
        self.seed = seed
        self._best_epoch = -1

        self.__init_kwargs(**kwargs)

    def __init_kwargs(self, **kwargs):
        # Create the directories where we dump the pkls and models
        if type(self.results_dir) != type(Path(".")):
            self.results_dir = Path(self.results_dir)
        self.models_dir: Path = self.results_dir / "models" / self.suffix
        self.pkl_dir: Path = self.results_dir / "pkl" / self.suffix
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.pkl_dir.mkdir(exist_ok=True, parents=True)

        self.forward_fnc: torch.Tensor = (
            lambda model, x, t, d, **kwargs_dict: model.forward(
                x=x, t=t, d=d, **kwargs_dict
            ).squeeze()
        )
        if constants.FWD_FNC in kwargs.keys():
            self.forward_fnc = kwargs[constants.FWD_FNC]

        self.logval_epochs = 1
        if constants.LOGVAL_EPOCHS in kwargs.keys():
            self.logval_epochs = kwargs[constants.LOGVAL_EPOCHS]

        self.sval_idxs = torch.arange(len(self.val_y))
        # if int(0.2 * len(self.val_y)) >= 200:
        #     _, self.sval_idxs = self.data_split(
        #         all_x=self.val_x, all_y=self.all_y, val_fraction=0.2
        #     )
        self.sval_x, self.sval_t, self.sval_d, self.sval_y = (
            self.val_x[self.sval_idxs],
            self.val_t[self.sval_idxs],
            self.val_d[self.sval_idxs],
            self.val_y[self.sval_idxs],
        )

        self.ckpts = None
        if constants.CHECKPOINTS in kwargs.keys():
            self.ckpts = kwargs[constants.CHECKPOINTS]

    @property
    def val_xtdy(self):
        return self.val_x, self.val_t, self.val_d, self.val_y

    @property
    def sval_xtdy(self):
        return self.sval_x, self.sval_t, self.sval_d, self.sval_y

    @property
    def _epoch(self):
        return self.epochs[-1]

    def data_split(self, all_x, all_y, val_fraction):
        """Splits the dataset into train and test
        We stratify only based on the target y values
        It returns the idxs in sorted order.
        """
        num_val_patients = int(np.floor(all_x.shape[0] * val_fraction))
        all_idxs = torch.randperm(len(all_x))
        train_indices, val_indices = (
            all_idxs[num_val_patients:],
            all_idxs[:num_val_patients],
        )
        return train_indices.sort()[0], val_indices.sort()[0]

    def dump_model(self, model: torch.nn.Module, logger, **kwargs):
        # Dump if last epoch
        if (self._epoch + 1) == self.num_epochs:
            torch.save(
                model.state_dict(), self.models_dir / f"last_model-{self.seed}.pt"
            )
            # Dump the val metrics also
            with open(self.pkl_dir / f"es-{self.seed}.pkl", "wb") as file:
                pkl.dump(
                    {
                        "epoch": self._epoch,
                        "val_rmse": self.val_metrics,
                        "train_factual_loss": self.train_floss,
                        "train_total_loss": self.train_tloss,
                        "best_epoch": self._best_epoch,
                        "train_idxs": self.train_idxs,
                        "val_idxs": self.val_idxs,
                    },
                    file,
                )

        # Dump the best model
        if self.task == "reg":
            prev_best = (
                min(self.val_metrics[:-1]) if len(self.val_metrics) > 1 else 1e10
            )
        else:
            prev_best = (
                max(self.val_metrics[:-1]) if len(self.val_metrics) > 1 else -1e10
            )
        if (
            (self.task == "reg" and self.val_metrics[-1] <= prev_best)
            or (self.task == "cls" and self.val_metrics[-1] >= prev_best)
            or len(self.val_metrics) == 0
        ):  # This is for the first epoch
            torch.save(
                model.state_dict(), self.models_dir / f"best_val_model-{self.seed}.pt"
            )
            self._best_epoch = self._epoch

        # Dump the model at checkpoints
        if self._epoch in self.ckpts:
            torch.save(
                model.state_dict(),
                self.models_dir / f"ckpt-{self._epoch}-{self.seed}.pt",
            )

    def log_val_metrics(self, model, train_floss, train_tloss, epoch, logger, **kwargs):
        """To save time, we log metrics on a sub-sampled validation dataset and log metrics on entire validation
        dataset every 5 epochs. Hope that early stopping is not very sensitie to epochs%5
        """
        if epoch % self.logval_epochs == 0 or epoch == self.num_epochs:
            x, t, d, y = self.val_xtdy
        else:
            x, t, d, y = self.sval_xtdy

        model.eval()
        with torch.no_grad():
            y_pred: torch.Tensor = self.forward_fnc(model, x=x, t=t, dosage=d, **kwargs)

        if self.task == "reg":
            self.val_metrics.append(
                torch.sqrt(((y_pred.squeeze() - y.squeeze()) ** 2).mean()).item()
            )
        elif self.task == "cls":
            assert (
                len(y_pred.shape) == 1
            ), "Pls pass fwd_fnc == forward_labels for classification task"
            self.val_metrics.append(torch.sum(y_pred.squeeze() == y.squeeze()) / len(y))
            logger.info(f"Epoch {epoch}; Validation Acc: {self.val_metrics[-1]}")
        else:
            assert False, f"Early stopping does not support {self.task} task yet!"
        self.epochs.append(epoch)
        self.train_floss.append(train_floss)
        self.train_tloss.append(train_tloss)
        # self.val_loss.append(val_loss) This is not required since val loss is mostly mse (square of valrmse)

        self.dump_model(model, logger, **kwargs)

        return self.val_metrics[-1]
    
    
    def get_val_perf(self, *, model, weights=None, **kwargs) -> float:
        """To save time, we log metrics on a sub-sampled validation dataset and log metrics on entire validation
        dataset every 5 epochs. Hope that early stopping is not very sensitie to epochs%5
        """
        
        x, t, d, y = self.val_xtdy
        model.eval()
        with torch.no_grad():
            y_pred: torch.Tensor = self.forward_fnc(model, x=x, t=t, dosage=d, **kwargs)

        if self.task == "reg":
            mse = nn.MSELoss(reduction="none")(y_pred.squeeze(), y.squeeze())
            if weights is not None:
                weights = weights.squeeze()
                assert len(weights) == len(mse)
                return torch.sum(mse * weights).item()
            else:
                return torch.mean(mse).item()
                        
        elif self.task == "cls":
            raise NotImplementedError()
        else:
            assert False, f"Early stopping does not support {self.task} task yet!"
