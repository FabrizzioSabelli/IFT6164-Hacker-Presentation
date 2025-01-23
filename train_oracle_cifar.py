from src.models import resnext18_cifar10
from src.train import train_classifier
from src.dataloaders import load_cifar10_dataset
from src.utils import set_seed


def main():

    sub_set_size = 500
    val_split = 0.1
    batch_size = 128
    set_seed()

    train_loader, val_loader, test_loader, test_dataset, final_loader, augmented_dataset = load_cifar10_dataset(batch_size, val_split, sub_set_size)

    for  data, target in val_loader:
        print(data.min(), data.max(), data.shape)
        break

    model = resnext18_cifar10()

    print(sum(p.numel() for p in model.parameters()))

    train_classifier(
        model,
        train_loader,
        val_loader,
        "share/classifier_weights/cifar10_resnext_norm",
        epochs=200,
        lr=0.005,
    )


if __name__ == "__main__":
    main()