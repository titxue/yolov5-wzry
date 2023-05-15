import os
import cv2
import numpy as np


def load_dataset(image_path, label_path, image_exts, label_ext):
    """
    Load a dataset consisting of images and corresponding labels.

    Args:
        image_path (str): The path to the directory containing the images.
        label_path (str): The path to the directory containing the labels.
        image_exts (list of str): The allowed file extensions for image files.
        label_ext (str): The file extension for label files.

    Returns:
        A list of tuples, where each tuple contains the following elements:
            - image: The loaded image as a NumPy array.
            - label: The loaded label as a NumPy array.
            - image_path: The path to the image file.
            - label_path: The path to the label file.
    """
    dataset = []

    # Iterate over all images in the dataset
    for filename in os.listdir(image_path):
        # Check if the file is an image
        if any(filename.endswith(ext) for ext in image_exts):
            # Load the image and its corresponding label
            image_path = os.path.join(image_path, filename)
            label_path = os.path.join(label_path, filename.replace(*image_exts, label_ext))
            image = cv2.imread(image_path)
            label = np.loadtxt(label_path)

            # Append the data to the dataset list
            dataset.append((image, label, image_path, label_path))

    return dataset


train_dataset = load_dataset("labelImgXml/JPEGImages/train", "labelImgXml/Annotations/train", [".jpg", ".jpeg", ".png"], ".txt")
val_dataset = load_dataset("labelImgXml/JPEGImages/valid", "labelImgXml/Annotations/valid", [".jpg", ".jpeg", ".png"], ".txt")

# Print the first element of the training set
print(train_dataset[0])

# Print the first element of the validation set
print(val_dataset[0])
