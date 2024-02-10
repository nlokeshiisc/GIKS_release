from torch.utils.data.sampler import Sampler
import torch.utils.data as data_utils
import random
import numpy as np
import constants
import torch


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = np.array(labels)
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class +
                                  1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)


class CustomThetaDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, data_ids, X, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        data_id, x, y = self.data_ids[index], self.X[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, y

    def __len__(self):
        return len(self.y)


class CustomXBetaDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, data_ids, X, Beta, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.Beta = Beta
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        data_id, x, beta, y = self.data_ids[index], self.X[index], self.Beta[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, beta, y

    def __len__(self):
        return len(self.y)


class CustomCtrCnfDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    Returns  data_id, x, beta, beta_ids, ycnf, ypreds, ylbl
    """

    def __init__(self, data_ids, X, Beta, Beta_ids, ycnf, ypreds, ylbl, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.Beta = Beta
        self.Beta_ids = Beta_ids
        self.ycnf = ycnf
        self.ypreds = ypreds
        self.ylbl = ylbl
        self.transform = transform

    def __getitem__(self, index):
        data_id, x, beta, beta_ids, ycnf,  ypreds, ylbl = self.data_ids[index], self.X[index], \
            self.Beta[index], self.Beta_ids[index], self.ycnf[index], self.ypreds[index], self.ylbl[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, beta, beta_ids, ycnf, ypreds, ylbl

    def __len__(self):
        return len(self.ylbl)


class CustomGrpXBetaDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, data_ids_grps, Xgrps, Betagrps, ygrps, transform, *args, **kwargs):
        self.data_ids_grps = data_ids_grps
        self.Xgrps = Xgrps
        self.Betagrps = Betagrps
        self.ygrps = ygrps
        self.transform = transform

    def __getitem__(self, index):
        data_idgrp, xgrp, betagrp, ygrp = self.data_ids_grps[
            index], self.Xgrps[index], self.Betagrps[index], self.ygrps[index]
        if self.transform is not None:
            xgrp = torch.stack([self.transform(entry) for entry in xgrp])
        return data_idgrp, xgrp, betagrp, ygrp

    def __len__(self):
        return len(self.ygrps)


class CustomGreedyDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, data_ids, X, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        data_id, x, y = self.data_ids[index], self.X[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, y

    def __len__(self):
        return len(self.y)


class CustomPhiDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, R_ids, X, Beta, tgt_Beta, transform, *args,  **kwargs):
        self.R_ids = R_ids
        self.X = X
        self.Beta = Beta
        self.tgt_Beta = tgt_Beta
        self.transform = transform

    def __getitem__(self, index):
        r_id, x, beta, tgt_beta = self.R_ids[index], self.X[index], self.Beta[index], self.tgt_Beta[index]

        if self.transform is not None:
            x = self.transform(x)
        return r_id, x, beta, tgt_beta

    def __len__(self):
        return len(self.R_ids)


class CustomPhiGenDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, R_ids, X, Beta, Sib_beta, Sij, Sib_losses, transform, *args,  **kwargs):
        self.R_ids = R_ids
        self.X = X
        self.Beta = Beta
        self.Sib_beta = Sib_beta
        self.Sij = Sij
        self.Sib_losses = Sib_losses
        self.transform = transform

    def __getitem__(self, index):
        r_id, x, beta, sib_beta, sij, sib_losses = self.R_ids[index], self.X[index], \
            self.Beta[index], self.Sib_beta[index], self.Sij[index], self.Sib_losses[index]

        if self.transform is not None:
            x = self.transform(x)
        return r_id, x, beta, sib_beta, sij, sib_losses

    def __len__(self):
        return len(self.R_ids)


class CustomPsiDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, data_ids, X, Beta, R_tgts, transform, *args,  **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.Beta = Beta
        self.R_tgts = R_tgts
        self.transform = transform

    def __getitem__(self, index):
        data_id, x, beta, r_tgt = self.data_ids[index], self.X[index], self.Beta[index], self.R_tgts[index]

        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, beta, r_tgt

    def __len__(self):
        return len(self.R_tgts)


def get_loader_subset(loader: data_utils.DataLoader, subset_idxs: list, batch_size=None, shuffle=False):
    """Returns a data loader with the mentioned subset indices
    """
    subset_ds = data_utils.Subset(dataset=loader.dataset, indices=subset_idxs)
    if batch_size is None:
        batch_size = loader.batch_size
    return data_utils.DataLoader(subset_ds, batch_size=batch_size, shuffle=shuffle)


def init_loader(ds: data_utils.Dataset, batch_size, shuffle=False, **kwargs):
    if constants.SAMPLER in kwargs:
        return data_utils.DataLoader(ds, batch_size=batch_size, sampler=kwargs[constants.SAMPLER])
    else:
        return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
