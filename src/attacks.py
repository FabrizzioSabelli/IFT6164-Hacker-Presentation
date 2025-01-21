import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import shutil
from tqdm import tqdm
from torch.nn.functional import cross_entropy
import os


class AdversialDataset(Dataset):
    def __init__(self, testloader, image_dir):
        """
        Args:
            testloader (DataLoader): DataLoader with og images that will be used to create adversial images
            image_dir (str): Directory where to put adversial images with attack type in name
        """
        self.annotations_df = pd.read_csv(
            image_dir + +"/annotations.csv"
        )  # columns ["adversial_id", "image_id","adversial_label", "sub_label", "oracle_label"]
        self.testloader = testloader

    def __len__(self):
        return len(self.annotations_df.index)

    def __getitem__(self, idx):
        """This will simply get the adversial image"""

        image_name = f"image_{idx}.pt"

        img_path = self.image_dir + "/" + image_name
        # label = self.annotations_df.iloc[idx]["og_label"]

        # Load the image
        image = torch.load(img_path)

        return image

    # Fast-Gradient Sign Method for computing adversial attacks
    def attack_FGSM(self, substitute, oracle, epsilon, batch_size=32):
        """Create the Black box attack and returns the success rate on the subsititute and the transferability to the oracle"""

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        rows = []

        # create folder for images
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)

        # create the adversial images
        for idx, (x, y) in enumerate(
            tqdm(self.testloader, desc="Creating attacks", leave=False, unit="batch")
        ):
            x, y = x.to(device), y.to(device)
            substitute.to(device)

            x_adv = self.FGSM(model=substitute, x=x, y=y, epsilon=epsilon)

            # get the true labels to compare later
            y_substitute = substitute.predict(x)
            y_oracle = oracle.predict(x)

            for idx in range(batch_size):

                x_adv_idx = x_adv[
                    idx
                ]  # test if this gets the correct image depending on all sizes

                new_idx = idx + idx * batch_size
                new_image_id = f"image_{new_idx}"
                new_image_path = self.image_dir + "/" + new_image_id

                x_adv_idx.to(device)

                # get the true labels to compare later
                y_substitute_adv = substitute.predict(x_adv_idx)
                y_oracle_adv = oracle.predict(x_adv_idx)

                rows.append(
                    {
                        "adversial_id": new_idx,
                        "image_id": idx,
                        "sub_label": y_substitute,
                        "oracle_label": y_oracle,
                        "adv_sub_label": y_substitute_adv,
                        "adv_oracle_label": y_oracle_adv,
                    }
                )

                # save new image
                torch.save(
                    x_adv_idx,
                    new_image_path,
                )

        self.annotations_df = pd.DataFrame(rows)
        self.annotations_df.to_csv(self.image_dir + "/annotations.csv", index=False)

        # return transferability and attack success rate on oracle and substitute
        substitute_success_frac = self.annotations_df[
            [self.annotations_df["sub_label"] == self.annotations_df["adv_sub_label"]]
        ].sum()
        oracle_success_frac = self.annotations_df[
            [
                self.annotations_df["oracle_label"]
                == self.annotations_df["adv_oracle_label"]
            ]
        ].sum()
        substitute_success = substitute_success_frac / len(self.annotations_df.index)
        oracle_success_succss = oracle_success_frac / len(self.annotations_df.index)

        print(
            f"Substitute attack success rate: {substitute_success}, fraction: {substitute_success_frac}/{len(self.annotations_df.index)}"
        )
        print(
            f"Oracle attack success rate: {oracle_success_succss}, fraction: {oracle_success_frac}"
        )

        return substitute_success, oracle_success_succss

    def test_transferability(self, oracle, batch_size=32):

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        adversial_loader = DataLoader(self, batch_size=batch_size, shuffle=False)
        oracle_true = []
        # create the adversial images
        for idx, (x, y) in enumerate(
            tqdm(
                self.testloader, desc="True Oracle requests", leave=False, unit="batch"
            )
        ):
            x, y = x.to(device), y

            # get the true labels to compare later
            oracle_true.append(oracle.predict(x))

        oracle_adversial = []
        for idx, (x, y) in enumerate(
            tqdm(
                adversial_loader,
                desc="Adversial Oracle requests",
                leave=False,
                unit="batch",
            )
        ):
            x = x.to(device)

            # get the true labels to compare later
            oracle_adversial.append(oracle.predict(x))

        # TODO test this

        oracle_true_np = np.array(oracle_true)
        oracle_adversial_np = np.array(oracle_adversial)

        # Find the indices where the arrays differ
        different_positions = np.where(oracle_true_np != oracle_adversial_np)[0]

        # Count the number of different positions
        attack_success_frac = len(different_positions)
        attack_sucess = attack_success_frac / len(self.annotations_df.index)
        print(
            f"Oracle attack success rate: {attack_sucess}, fraction: {attack_success_frac}/{len(self.annotations_df.index)}"
        )

        return attack_sucess
