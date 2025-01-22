from src.utils import set_seed
from src.dataloaders import load_cifar10_dataset
from src.models import Oracle, Substitute, DummyCNN
from src.train import train_substitute


def main():
    # set seed to make sure its correct

    sub_set_size = 150
    val_split = 0.1
    epsilon = 0.3
    batch_size = 32
    lambda_ = 0.1
    set_seed()

    # CIFAR-10 substitute

    c10_sub = DummyCNN(
        [16, 32, 64, 128, 256],
        [256, 64],
        kernel_size=3,
        input_shape=(3, 32, 32),
        num_classes=10,
    )
    c10_oracle = Oracle("classifier_weights/c10_resnext.pth")
    train_loader, val_loader, test_loader, final_loader, augmented_dataset = (
        load_cifar10_dataset(
            batch_size=batch_size,
            validation_split=val_split,
            sub_set_size=sub_set_size,
            save_substitute_path="data/CIFAR-10/substitute/arch_dnn_0",
        )
    )
    path_to_save = "model/CIFAR-10/substitute/arch_dnn_0"
    train_substitute(
        c10_sub,
        c10_oracle,
        augmented_dataset,
        path_to_save,
        test_loader=test_loader,
        lambda_=lambda_,
    )


if __name__ == "__main__":
    main()
