from src.attacks import AdversarialDataset
from src.utils import set_seed
from src.dataloaders import load_mnist_dataset
from src.models import Oracle, Substitute


def main():

    sub_set_size = 150
    val_split = 0.1
    epsilon = 0.3
    batch_size = 32
    set_seed()

    (
        train_loader,
        val_loader,
        test_loader,
        test_dataset,
        final_loader,
        augmented_dataset,
    ) = load_mnist_dataset(
        batch_size,
        validation_split=val_split,
        sub_set_size=sub_set_size,
        download=False,
        save_substitute_path="share/data/MNIST/substitute/arch_dnn_0",
    )
    test_oracle = Oracle("share/classifier_weights/mnist_cnn.pth", "cpu")
    test_sub = Substitute(
        "share/model/MNIST/substitute/arch_dnn_0/it6/model_epoch_10.pth", "cpu"
    )
    test_adv_d = AdversarialDataset(
        test_set=test_dataset,
        image_dir="share/data/MNIST/substitute/arch_dnn_0/adversial/fgsm",
    )
    test_adv_d.attack_FGSM(substitute=test_sub, oracle=test_oracle, epsilon=epsilon)


if __name__ == "__main__":
    main()
