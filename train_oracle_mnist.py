from src.models import DummyCNN
from src.train import train_classifier
from src.dataloaders import load_mnist_dataset
from src.utils import set_seed


def main():

    sub_set_size = 150
    val_split = 0.1
    batch_size = 32
    set_seed()

    train_loader, val_loader, test_loader, test_dataset, final_loader, augmented_dataset = (
        load_mnist_dataset(32, validation_split=0.1, sub_set_size=150, download=False)
    )

    for  data, target in val_loader:
        print(data.min(), data.max(), data.shape)
        break

    smol_mnist_cnn = DummyCNN(
        [16, 32], [32], kernel_size=3, input_shape=(1, 28, 28), num_classes=10
    )

    print(sum(p.numel() for p in smol_mnist_cnn.parameters()))

    train_classifier(
        smol_mnist_cnn,
        train_loader,
        val_loader,
        "share/classifier_weights/small_mnist_cnn",
        epochs=10,
        lr=0.0007,
    )


if __name__ == "__main__":
    main()
