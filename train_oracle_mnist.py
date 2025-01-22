from src.models import DummyCNN
from src.train import train_classifier
from src.dataloaders import load_mnist_dataset
from src.utils import set_seed


def main():

    sub_set_size = 150
    val_split = 0.1
    batch_size = 32
    set_seed()
    train_loader, val_loader, test_loader, final_loader, augmented_dataset = (
        load_mnist_dataset(
            batch_size=batch_size,
            validation_split=val_split,
            sub_set_size=sub_set_size,
            download=False,
            save_substitute_path="data/MNIST/substitute/arch_dnn_0",
        )
    )

    test_model = DummyCNN(
        [16, 32, 64, 96],
        [192, 64],
        kernel_size=3,
        input_shape=(1, 28, 28),
        num_classes=10,
    )
    train_classifier(
        test_model,
        train_loader,
        val_loader,
        test_loader,
        save_interval=1,
        save_name="test_mnist",
    )

    train_loader, val_loader, test_loader, final_loader, augmented_dataset = (
        load_mnist_dataset(32, validation_split=0.1, sub_set_size=150, download=False)
    )
    smol_mnist_cnn = DummyCNN(
        [16, 32], [32], kernel_size=3, input_shape=(1, 28, 28), num_classes=10
    )
    print(sum(p.numel() for p in smol_mnist_cnn.parameters()))
    train_classifier(
        smol_mnist_cnn,
        train_loader,
        val_loader,
        "classifier_weights/small_mnist_cnn",
        epochs=10,
        lr=0.001,
    )


if __name__ == "__main__":
    main()
