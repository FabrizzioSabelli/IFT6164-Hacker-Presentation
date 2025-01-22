import torch

from src.dataloaders import load_cifar10_dataset
from src.utils import set_seed

def load_inspect_cifar10(root=None, download=False):
    set_seed()
    mean = 0
    div = 0
    train_loader, val_loader, test_loader, final_loader, aug_dataset = load_cifar10_dataset()
    for (data, target) in val_loader:
        print(data.shape, data.max(), data.min())
        mean += torch.mean(data)
        div += 1
    print(f"Mean: {mean/div}")
if __name__ == "__main__":
    load_inspect_cifar10()