import os
import torch
import matplotlib.pyplot as plt

def inspect_images(root_folder: str, num_images: int = 5, cmap: str = 'viridis'):
    """
    Inspect a set number of torch tensors saved as .pt files as images using matplotlib.

    Args:
        root_folder (str): The root folder containing subfolders with .pt files.
        num_images (int): The number of images to inspect (default is 5).
        cmap (str): The colormap to use for displaying images (default is 'viridis').

    Returns:
        None: Displays the images using matplotlib.
    """
    # List to store all paths to .pt files
    pt_files = []

    # Walk through all subfolders and collect .pt files
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))

    # Limit the number of files to inspect
    if len(pt_files) == 0:
        print("No .pt files found in the specified folder.")
        return

    pt_files = pt_files[:num_images]

    # Plot each tensor
    for i, pt_file in enumerate(pt_files):
        # Load the tensor
        tensor = torch.load(pt_file)

        # Check if tensor is 2D (grayscale) or 3D (RGB) and adjust for plotting
        if len(tensor.shape) == 3 and tensor.shape[0] in (1, 3):  # Assume CHW format
            tensor = tensor.permute(1, 2, 0)  # Convert to HWC for visualization
            if tensor.shape[-1] == 1:  # Grayscale stored in CHW format
                tensor = tensor.squeeze(-1)
        
        # Convert tensor to numpy for matplotlib
        img = tensor.numpy()

        # Plot the image
        plt.figure(figsize=(6, 6))
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap=cmap)
        else:  # RGB
            plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i + 1}: {os.path.basename(pt_file)}")
        plt.show()

if __name__ == "__main__":
    inspect_images("share/data/MNIST/substitute/arch_dnn_0_norm_T", 3)