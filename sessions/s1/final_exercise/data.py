import os
import torch
from pathlib import Path


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    path = Path(os.path.realpath(__name__))
    data_dir = path.parent.parent.parent.parent.absolute() / 'data/corruptmnist'

    x_train = [torch.load(f'{data_dir}/train_images_{i}.pt') for i in range(6)]
    y_train = [torch.load(f'{data_dir}/train_target_{i}.pt') for i in range(6)]
    x_train = torch.cat(x_train)
    y_train = torch.cat(y_train)

    x_test = torch.load(f'{data_dir}/test_images.pt')
    y_test = torch.load(f'{data_dir}/test_target.pt')
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)
    # train = (x_train, y_train)
    # test = (x_test, y_test)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)

    return train, test
