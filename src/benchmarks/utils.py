#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from torch.utils.data.dataset import Dataset
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset


def wrap_with_task_labels(datasets, target_transform=None):
    return [AvalancheDataset(ds, task_labels=idx, target_transform=target_transform) for idx, ds in enumerate(datasets)]


class XYDataset(Dataset):
    """ Template Dataset with Labels """

    def __init__(self, x, y, transform=None, **kwargs):
        self.data, self.targets = x, y
        self.transform = transform
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y