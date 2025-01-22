# utils functions
import torch
import numpy as np
import random

def set_seed():
    """Function to manually set the seed in random, numpy and torch for reproducibility."""
    # Set the seed value for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch seed for CPU
    torch.manual_seed(seed)

    # PyTorch seed for CUDA (single and multi-GPU setups)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    global is_seed_setted

    is_seed_setted = True


def check_seed_setted():

    global is_seed_setted
    if (
        "is_seed_setted" not in globals() and not is_seed_setted
    ):  # is_seed_setted must be true
        raise AssertionError(
            "Seed must be set at the start of the file before running this function. Simply import set_seed() from utils.py and call the function at the start of your main function."
        )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_device(model):
    """Function that sets the device and mode for the model. Device works for CUDA, METAL and CPU."""
    # Get the device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "GPU cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_name = "GPU mps"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    model.to(device)

    return device, device_name


def mount_drive():
    """Function to mount the Google Drive locally. Also creates a global variable with a path to your drive for easy access."""
    # Mount the drive

    # from google.colab import drive, auth
    # drive.mount('/content/drive')

    pass