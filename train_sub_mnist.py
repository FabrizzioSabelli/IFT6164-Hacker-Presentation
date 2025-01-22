from src.utils import set_seed
from src.dataloaders import load_mnist_dataset
from src.models import Oracle, Substitute, DummyCNN
from src.train import train_substitute


def main():
    # set seed to make sure its correct

    sub_set_size = 150
    val_split = 0.1
    batch_size = 32
    lambda_ = 0.1
    train_epochs = 10
    run_name = "arch_dnn_0_T"
    set_seed()

    (
        train_loader,
        val_loader,
        test_loader,
        test_dataset,
        final_loader,
        augmented_dataset,
    ) = load_mnist_dataset(
        batch_size=batch_size,
        validation_split=val_split,
        sub_set_size=sub_set_size,
        download=False,
        save_substitute_path=f"share/data/MNIST/substitute/{run_name}",
    )

    # MNIST substitute

    mnist_sub = DummyCNN(
        [16, 32, 64, 96],
        [192, 64],
        kernel_size=3,
        input_shape=(1, 28, 28),
        num_classes=10,
    )
    mnist_oracle = Oracle("share/classifier_weights/mnist_cnn.pth")
    train_loader, val_loader, test_loader, tets_dataset, final_loader, augmented_dataset = (
        load_mnist_dataset(
            batch_size=batch_size,
            validation_split=val_split,
            sub_set_size=sub_set_size,
            download=False,
            save_substitute_path=f"share/data/MNIST/substitute/{run_name}",
        )
    )
    path_to_save = f"share/model/MNIST/substitute/{run_name}"
    train_substitute(
        mnist_sub,
        mnist_oracle,
        augmented_dataset,
        path_to_save,
        test_loader=test_loader,
        lambda_=lambda_,
        train_epochs = train_epochs
    )


if __name__ == "__main__":
    main()
