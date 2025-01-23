import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import os
import pandas as pd
import shutil
from .augmented_dataset import AugmentedDataset
from .utils import check_seed_setted
import numpy as np

# MNIST
def load_mnist_dataset(
    batch_size=64,
    validation_split=0.1,
    sub_set_size=150,
    download=False,
    save_substitute_path=None,
):
    # check if seed was set
    check_seed_setted()

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    full_train_dataset = datasets.MNIST(
        root="share/data", train=True, download=download, transform=transform
    )
    full_test_dataset = datasets.MNIST(
        root="share/data", train=False, download=download, transform=transform
    )
    # Split the training dataset into training and validation datasets
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * validation_split)
    train_size = total_train_size - val_size

    # This should remain the same everytime since we've set the seed
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    total_test_size = len(full_test_dataset)
    if sub_set_size > total_test_size:
        raise ValueError(
            "final_set_size cannot be larger than the total test dataset size."
        )
    remaining_test_size = total_test_size - sub_set_size

    # This should remain the same everytime since we've set the seed
    test_dataset, final_dataset = random_split(
        full_test_dataset, [remaining_test_size, sub_set_size]
    )

    # save the subsitute initial dataset to csv and folder for easy manipulation later
    aug_dataset = None
    if save_substitute_path is not None:

        rows = []
        image_dir = os.path.join(save_substitute_path, "it0")

        # if it exists then delete it
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)  # recreate it

        for idx in range(len(final_dataset)):
            image, label = final_dataset.__getitem__(idx)
            image_id = f"image_{idx}.pt"
            image_path = os.path.join(image_dir, image_id)
            torch.save(image, image_path)
            rows.append(
                {"image_id": idx, "og_id": idx, "true_label": label, "augmented_id": 0}
            )

        annotations_df = pd.DataFrame(rows)

        annotations_path = os.path.join(save_substitute_path, "annotations.csv")
        annotations_df.to_csv(annotations_path, index=False)

        # create substitute dataset
        aug_dataset = AugmentedDataset(
            annotations_path=annotations_path,
            image_dir=save_substitute_path,
            init_size=sub_set_size,
        )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # here we use aug_dataset instead of final_dataset to save iterations of the dataset
    final_loader = None
    if aug_dataset is not None:
        final_loader = DataLoader(
            dataset=aug_dataset, batch_size=batch_size, shuffle=True
        )

    return train_loader, val_loader, test_loader, test_dataset, final_loader, aug_dataset


def load_cifar10_dataset(
    batch_size=64, validation_split=0.1, sub_set_size=500, save_substitute_path=None
):
    # check if seed was set
    check_seed_setted()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    )
    full_train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    full_test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    # Split the training dataset into training and validation datasets
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * validation_split)
    train_size = total_train_size - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    total_test_size = len(full_test_dataset)
    if sub_set_size > total_test_size:
        raise ValueError(
            "final_set_size cannot be larger than the total test dataset size."
        )
    remaining_test_size = total_test_size - sub_set_size
    test_dataset, final_dataset = random_split(
        full_test_dataset, [remaining_test_size, sub_set_size]
    )

    # save the subsitute initial dataset to csv and folder for easy manipulation later
    aug_dataset = None
    if save_substitute_path is not None:

        rows = []
        image_dir = os.path.join(save_substitute_path, "it0")

        # if it exists then delete it
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)  # recreate it

        for idx in range(len(final_dataset)):
            image, label = final_dataset.__getitem__(idx)
            image_id = f"image_{idx}.pt"
            image_path = os.path.join(image_dir, image_id)
            torch.save(image, image_path)
            rows.append(
                {"image_id": idx, "og_id": idx, "true_label": label, "augmented_id": 0}
            )

        annotations_df = pd.DataFrame(rows)

        annotations_path = os.path.join(save_substitute_path, "annotations.csv")
        annotations_df.to_csv(annotations_path, index=False)

        # create substitute dataset
        aug_dataset = AugmentedDataset(
            annotations_path=annotations_path,
            image_dir=save_substitute_path,
            init_size=sub_set_size,
        )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    final_loader = None
    if aug_dataset is not None:
        final_loader = DataLoader(
            dataset=aug_dataset, batch_size=batch_size, shuffle=True
        )

    return train_loader, val_loader, test_loader, test_dataset, final_loader, aug_dataset

def load_minist_dataset(
    batch_size=64,
    validation_split=0.1,
    sub_set_size=100,
    download=False,
    save_substitute_path=None,
    npz_path="share/data/MINIST/raw/minist.npz"
):
    # check if seed was set
    check_seed_setted()

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    full_train_dataset = datasets.MNIST(
        root="share/data", train=True, download=download, transform=transform
    )
    full_test_dataset = datasets.MNIST(
        root="share/data", train=False, download=download, transform=transform
    )
    # Split the training dataset into training and validation datasets
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * validation_split)
    train_size = total_train_size - val_size

    # This should remain the same everytime since we've set the seed
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    total_test_size = len(full_test_dataset)
    if sub_set_size > total_test_size:
        raise ValueError(
            "final_set_size cannot be larger than the total test dataset size."
        )
    remaining_test_size = total_test_size - sub_set_size

    # This should remain the same everytime since we've set the seed
    test_dataset = full_test_dataset
    final_dataset = load_dataset_from_npz(npz_path, batch_size)

    # save the subsitute initial dataset to csv and folder for easy manipulation later
    aug_dataset = None
    if save_substitute_path is not None:

        rows = []
        image_dir = os.path.join(save_substitute_path, "it0")

        # if it exists then delete it
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)  # recreate it

        for idx in range(len(final_dataset)):
            image, label = final_dataset.__getitem__(idx)
            image_id = f"image_{idx}.pt"
            image_path = os.path.join(image_dir, image_id)
            torch.save(image, image_path)
            rows.append(
                {"image_id": idx, "og_id": idx, "true_label": label, "augmented_id": 0}
            )

        annotations_df = pd.DataFrame(rows)

        annotations_path = os.path.join(save_substitute_path, "annotations.csv")
        annotations_df.to_csv(annotations_path, index=False)

        # create substitute dataset
        aug_dataset = AugmentedDataset(
            annotations_path=annotations_path,
            image_dir=save_substitute_path,
            init_size=sub_set_size,
        )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # here we use aug_dataset instead of final_dataset to save iterations of the dataset
    final_loader = None
    if aug_dataset is not None:
        final_loader = DataLoader(
            dataset=aug_dataset, batch_size=batch_size, shuffle=True
        )

    return train_loader, val_loader, test_loader, test_dataset, final_loader, aug_dataset

class NpzDataset(Dataset):
    def __init__(self, npz_file):

        # Load the .npz file
        data = np.load(npz_file)
        
        # Assume the .npz file has 'data' and 'labels' arrays
        self.data = torch.tensor(data['data'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)

    def __len__(self):
        """Returns the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_dataset_from_npz(npz_file, batch_size, shuffle=True):

    dataset = NpzDataset(npz_file)
    return dataset