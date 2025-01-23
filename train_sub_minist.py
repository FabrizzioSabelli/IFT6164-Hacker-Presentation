import torch
from src.utils import set_seed
from src.dataloaders import load_minist_dataset
from src.models import Oracle, Substitute, DummyCNN
from src.train import train_substitute
import numpy as np


def main():
    # set seed to make sure its correct

    sub_set_size = 100
    val_split = 0.1
    batch_size = 32
    lambda_ = 0.1
    train_epochs = 10
    run_name = "arch_dnn_custom_dataset_norm2_T"
    set_seed()
    """
    custom_data = np.load("share/data/MINIST/raw/minist.npz")
    c_data = custom_data["data"]
    c_labels = custom_data["labels"]
    print(c_data.shape)
    save_minist_folder = "share/data/MINIST/tensors/"
    for i in range(c_data.shape[0]):
        t = torch.Tensor(c_data)
        torch.save(c_data, f"share/data/MNIST/substitute/{run_name}/it0/image_{i}.pt")
    """

    (
        train_loader,
        val_loader,
        test_loader,
        test_dataset,
        final_loader,
        augmented_dataset,
    ) = load_minist_dataset(
        batch_size=batch_size,
        validation_split=val_split,
        sub_set_size=sub_set_size,
        download=False,
        save_substitute_path=f"share/data/MINIST/substitute/{run_name}",
    )

    # MINIST substitute
    mnist_sub = DummyCNN(
        [16, 32, 64],
        [64],
        kernel_size=3,
        input_shape=(1, 28, 28),
        num_classes=10,
    )
    mnist_oracle = Oracle("share/classifier_weights/mnist_norm_small/model_epoch_9.pth")

    path_to_save = f"share/model/MINIST/substitute/{run_name}"
    train_substitute(
        mnist_sub,
        mnist_oracle,
        augmented_dataset,
        path_to_save,
        test_loader=test_loader,
        lambda_=lambda_,
        train_epochs = train_epochs,
        lr = 0.01
    )


if __name__ == "__main__":
    main()
