import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
Used to convert our custom mnists to easy to load in torch numpy format.
"""

def create_image_dataset(folder_path, image_size=(64, 64)):
    """
    Create a dataset from image files in a folder, using the first character
    of the filenames (assumed to be an integer) as labels.
    """
    images = []
    labels = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue
        
        # Get the label from the first character of the filename
        try:
            label = int(filename[0])
        except ValueError:
            print(f"Skipping file '{filename}' - first character is not an integer.")
            continue
        
        # Open the image and process it
        try:
            with Image.open(file_path) as img:
                # Convert to RGB (if not already) and resize
                img = img.convert("RGB").resize(image_size)
                # Convert image to a NumPy array
                img_array = np.array(img)/255.0
                img_array = np.expand_dims(np.mean(img_array, axis=-1), axis=0)
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")
            continue

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

if __name__ == "__main__":
    images, labels = create_image_dataset('minist', image_size=(28, 28))
    np.savez_compressed("minist.npz", data=images, labels=labels)
    test = np.load("minist.npz")
    ti = test["data"]
    tl = test["labels"]
    # Just sanity checks
    print(ti, ti.shape, tl, tl.shape)
    idxs = np.random.randint(0, 99, 10)
    for i in idxs:
        iarray = ti[i, 0]
        plt.imshow(iarray, cmap="gray")
        plt.colorbar()  # Optional: show color scale
        plt.show()
