from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import shutil


class AugmentedDataset(Dataset):
    def __init__(self, annotations_path, image_dir, init_size, reservoir_sampling=None):
        """
        Args:
            annotations (str): CSV file containing image file name and corresponding label.
            img_dir (str): Directory with all the images represented as tensors.
            init_size (int): Number of images in the initial dataset.
        """
        self.annotations_df = pd.read_csv(
            annotations_path
        )  # columns ["image_id", "oracle_label", "og_id", "augmented_id"]
        self.image_dir = image_dir
        self.annotations_path = annotations_path
        self.aug_iters = 0  # initially no augmentations
        self.init_size = init_size
        self.reservoir_sampling = reservoir_sampling

    def reservoir_sampling(self, dataset, idx):
        pass

    def set_oracle_values(self, oracle_values):
        self.annotations_df["oracle_label"] = oracle_values
        self.annotations_df.to_csv(self.annotations_path, index=False)

    def jacobian_augmentation(
        self,
        oracle,
        substitute,
        lambda_=0.1,
    ):

        if self.reservoir_sampling is not None:
            # dataset = self.reservoir_sampling(dataset, idx)
            pass

        else:
            sample_size = self.init_size

        tmp_aug_iters = self.aug_iters + 1

        # create new directory to save. If replace true then we overwrite the previous directory
        folder = self.image_dir + "/" + f"it{tmp_aug_iters}"
        # if it exists then delete it
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)  # create it

        new_annotations = []
        dataset_size = len(self.annotations_df.index)
        # TODO modify later for CIFAR-10
        for idx in range(sample_size):

            x, y = self.__getitem__(idx)

            jacobian = torch.autograd.functional.jacobian(
                substitute, x.unsqueeze(dim=0)
            ).squeeze()  # remove all the 1s
            # jacobian input (batch_size, num_channels, height, width) output (batch_size, num_output, batch_size, num_channels, height, width)
            x_new = x + lambda_ * torch.sign(jacobian[y, :, :])

            new_idx = dataset_size + idx
            new_image_id = f"image_{new_idx}.pt"
            new_image_path = folder + "/" + new_image_id

            # save new image
            torch.save(
                x_new,
                new_image_path,
            )

            # predict or forward
            y_new = oracle.predict(x_new)

            new_annotations.append(
                {
                    "image_id": new_image_id,
                    "oracle_label": y_new,
                    "og_id": new_idx % self.init_size,
                    "augmented_id": tmp_aug_iters,
                    "true_label": -1,  # since we don't care
                }
            )

        # we finish by updating at the end not to mess up the dataloading with getitem
        self.aug_iters += 1  # new augmentation iteration
        new_annotations_df = pd.DataFrame(new_annotations)

        # save the new image annotations
        self.annotations_df = pd.concat([self.annotations_df, new_annotations_df])
        self.annotations_df.to_csv(self.annotations_path, index=False)

    def __len__(self):
        return len(self.annotations_df.index)

    # TODO: implement reservoir sampling indexing example and in this
    def __getitem__(self, idx):
        """This will go into the image directory and load the correct image as a tensor given the augmented iteration and the original image index.

        For example, there are 150 images initially so in f"{image_dir}/0/image_{idx}.pt", {idx} ranges from 0 to 149.
        Then the augmented images of the first iteration will be f"{image_dir}/1/image_{idx}.pt", {idx} ranges from 150 to 299.
        And the augmented images of the second iteration will be f"{image_dir}/2/image_{idx}.pt", {idx} ranges from 300 to 449.
        And so on. In this example, we assume no reservoir sampling, if not the indexes will be different.
        """

        # Name of image file is f"{image_dir}/{aug_id}/image_{idx}.pt"
        aug_id = idx // self.init_size

        image_name = f"image_{idx}.pt"
        folder = f"it{aug_id}"

        img_path = self.image_dir + "/" + folder + "/" + image_name
        label = self.annotations_df.iloc[idx]["oracle_label"]

        # Load the image
        image = torch.load(img_path)

        return image, label

    def get_augmentation_path(self, og_idx) -> dict[str, torch.Tensor]:
        """Function that returns the path of an augmented image given the original image index. Returns a dictionnary with keys augmented_id and values the images as tensors."""
        image_id = self.annotations_df[
            self.annotations_df["og_id"] == og_idx
        ].sort_values(by=["augmented_id"], ascending=True)[
            ["image_id"], ["augmented_id"]
        ]
        aug_images = {}
        for idx, row in image_id.iterrows():

            img_path = os.path.join(self.image_dir, row["image_id"])
            aug_images[row["augmented_id"]] = torch.load(img_path)
        return aug_images

    def get_augmented_image(self, og_idx, aug_idx):
        "Function that returns the image_id and image of a specific augmented image from the agumented image dataset"

        image_id = self.annotations_df[
            self.annotation_df["og_id"]
            == og_idx & self.annotations_df["augmented_id"]
            == aug_idx
        ]["image_id"]
        img_path = os.path.join(self.image_dir, image_id)
        image = torch.load(img_path)
        return image_id, image
