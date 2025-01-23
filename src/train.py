# Functions used to train the classifier and substitute model
import torch
from .utils import check_seed_setted, set_device
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd
import shutil
from .dataloaders import load_mnist_dataset, load_cifar10_dataset
from torch.utils.data import DataLoader
from .augmented_dataset import AugmentedDataset


def train_classifier(
    model,
    train_loader,
    val_loader,
    save_name,
    epochs=200,
    lr=0.001,
    batch_size=32,
    optimizer=None,
    scheduler=None,
    overwrite=False,
    keep_progess=True,
):
    # check if seed was setted and set the device
    check_seed_setted()

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    # clear the model weights save folder
    if os.path.exists(save_name):
        shutil.rmtree(save_name)

    # create the folder since it doens't exist or didn't
    os.makedirs(save_name, exist_ok=True)

    max_accuracy_validation = -1
    # keep progress bar or not
    epoch_progress = tqdm(
        range(1, epochs + 1),
        desc="Training Progress",
        unit="epoch",
        leave=keep_progess,
    )

    model_metrics = []
    for epoch in epoch_progress:
        train_batches(model, train_loader, optimizer, criterion, epoch)

        if val_loader is not None:
            correct, total, test_loss, accuracy = test_batches(
                model, val_loader, criterion, epoch=epoch
            )

            model_metrics_row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "accuracy": accuracy,
                "test_loss": test_loss,
            }

            epoch_progress.set_postfix(
                test_loss=f"{test_loss:.4f}",
                accuracy=f"{correct}/{total} ({accuracy:.2f}%)",
                lr=f"{optimizer.param_groups[0]['lr']}",
            )

            if (
                save_name and max_accuracy_validation < accuracy
            ):  # save model with highest validation accuracy

                model_metrics_row["saved"] = (
                    "True"  # since want to read easily in table
                )
                max_accuracy_validation = accuracy
                save_model_to_drive(
                    model, save_name, int(epoch), overwrite_epochs=overwrite
                )
            else:
                model_metrics_row["saved"] = "False"

            model_metrics.append(model_metrics_row)

        else:
            # only used in subsitute training

            save_model_to_drive(
                model, save_name, int(epoch), overwrite_epochs=overwrite
            )

        scheduler.step()
    # end of training so we then save the metrics
    if len(model_metrics) > 0:
        model_metrics_df = pd.DataFrame(model_metrics)
        model_metrics_df.to_csv(f"{save_name}/model_metrics.csv", index=False)


def train_batches(model, loader, optimizer, criterion, epoch):

    # set model to device
    device, device_name = set_device(model)

    # set model to train mode
    model.train()

    # create tqdm batch progress bar
    batch_progress = tqdm(
        loader,
        desc=f"Epoch {epoch} Training ({device_name})",
        leave=False,
        unit="batch"
    )

    # train the model
    for data, target in batch_progress:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # raw logits from model

        # TODO check this
        loss = criterion(output, torch.flatten(target))  # using CrossEntropyLoss here
        loss.backward()
        optimizer.step()

        batch_progress.set_postfix(loss=loss.item())


def test_batches(model, loader, criterion, epoch):

    # set model to device
    device, device_name = set_device(model)

    # set model to eval mode
    model.eval()

    # test metrics
    test_loss = 0
    correct = 0

    # create tqdm batch progress bar
    batch_progress = tqdm(
        loader,
        desc=f"Epoch {epoch} Testing {device_name}",
        leave=False,
        unit="batch",
    )
    with torch.no_grad():
        for data, target in batch_progress:
            data, target = data.to(device), target.to(device)
            output = model(data)  # raw logits from model
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)  # is this correct
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return correct, len(loader.dataset), test_loss, accuracy


def save_model_to_drive(model, name, epoch, overwrite_epochs):

    if overwrite_epochs:
        full_path = f"{name}/model.pth"
    else:
        full_path = f"{name}/model_epoch_{str(epoch)}.pth"
    torch.save(model, full_path)


# Function to train a substitute model given a substitute and an oracle
# "model/MNIST/substitute/arch_dnn_0"
def train_substitute(
    substitute,
    oracle,  # oracle model
    subdataset: AugmentedDataset,  # substitute dataset used during Jacobian Augmentation
    save_name,
    jac_aug_epochs=6,  # number of jacobian augmentation epochs
    train_epochs=10,  # number of epochs to train substitute model on each augmentation
    lr=0.01,
    lambda_=0.1,  # lambda parameter for the Jacobian Augmentation
    batch_size=32,
    test_loader=None,  # large testset with no validation set
    optimizer=None,
    scheduler=None,
    log_interval=5,
    save_interval=5,
    overwrite=False,
    first_run=True,
):
    # check if seed was setted and set the device
    check_seed_setted()

    if first_run:
        # set the oracle values for the first set as we need them to iterate over dataset later
        oracle_values = []

        # set oracle to device and eval model
        device, device_name = set_device(oracle)

        oracle.eval()
        # query the oracle
        with torch.no_grad():
            for idx in range(len(subdataset)):
                image_id = f"image_{idx}.pt"
                image_path = os.path.join(
                    subdataset.image_dir, "it0", image_id
                )  # no augmentation yet so 0
                image = torch.load(image_path, map_location=torch.device(device))

                oracle_values.append(oracle.predict(image))

            # TODO check this
            subdataset.init_oracle_values(
                oracle_values=[ov.to(torch.device("cpu")) for ov in oracle_values]
            )

    # initialize loader
    train_loader = DataLoader(subdataset, batch_size=batch_size, shuffle=True)

    save_path = save_name + "/it0"  # initial model

    # clear the model weights save folder
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # create the folder since it doens't exist or didn't
    os.makedirs(save_path, exist_ok=True)

    model_metrics = []

    # train the first time and get weights
    train_classifier(
        model=substitute,
        train_loader=train_loader,
        val_loader=None,
        epochs=train_epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=None,  # AdamW with momentum 0.9
        scheduler=None,  # CosineAnnealing
        save_name=save_path,
        overwrite=False,
        keep_progess=False,
    )

    # jacobian augmentation
    epoch_progress = tqdm(
        range(1, jac_aug_epochs + 1), desc="Training Progress", unit="epoch"
    )

    # test initial model
    if test_loader is not None:
        test_subsitute(
            substitute=substitute,
            test_loader=test_loader,
            epoch=0,
            model_metrics=model_metrics,
            epoch_progress=epoch_progress,
        )

    for i in epoch_progress:

        # first: jacobian augmentation and oracle call
        subdataset.jacobian_augmentation(
            oracle=oracle.to(device), substitute=substitute.to(device), lambda_=lambda_
        )

        # recreate the dataloader
        train_loader = DataLoader(subdataset, batch_size=batch_size, shuffle=True)

        save_path = save_name + f"/it{i}"  # initial model

        # clear the model weights save folder
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # create the folder since it doens't exist or didn't
        os.makedirs(save_path, exist_ok=True)

        # retrain
        train_classifier(
            model=substitute,
            train_loader=train_loader,
            val_loader=None,
            epochs=train_epochs,
            lr=lr,
            batch_size=batch_size,
            optimizer=None,  # AdamW with momentum 0.9
            scheduler=None,  # CosineAnnealing
            save_name=save_path,
            overwrite=False,
            keep_progess=False,
        )

        # test the accuracy of the model on the outside dataset
        # sanity check as this should go up
        if test_loader is not None:
            test_subsitute(
                substitute=substitute,
                test_loader=test_loader,
                epoch=i,
                model_metrics=model_metrics,
                epoch_progress=epoch_progress
            )

    if len(model_metrics) > 0:
        model_metrics_df = pd.DataFrame(model_metrics)
        model_metrics_df.to_csv(f"{save_name}/model_metrics.csv", index=False)


def test_subsitute(substitute, test_loader, epoch, model_metrics: list, epoch_progress):
    # test the accuracy of the model on the outside dataset
    # sanity check as this should go up

    correct, total, test_loss, accuracy = test_batches(
        substitute, test_loader, criterion=nn.CrossEntropyLoss(), epoch=epoch
    )

    model_metrics_row = {
        "epoch": f"it{epoch}",
        "accuracy": accuracy,
        "test_loss": test_loss,
    }

    model_metrics.append(model_metrics_row)

    epoch_progress.set_postfix(
        test_loss=f"{test_loss:.4f}", accuracy=f"{correct}/{total} ({accuracy:.2f}%)"
    )
