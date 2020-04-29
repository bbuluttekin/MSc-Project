import datetime

import matplotlib.pyplot as plt
import numpy as np


def display_images_from_dataset(dataset, n=9):
    plt.figure(figsize=(13, 13))
    subplot = 331
    for i, (image, label) in enumerate(dataset):
        plt.subplot(subplot)
        plt.axis('off')
        plt.imshow(image.numpy().astype(np.uint8))
        plt.title(label.numpy().decode("utf-8"), fontsize=16)
        subplot += 1
        if i == n - 1:
            break
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def visualize_augmented_images(original, augmented):
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()
    plt.savefig(f"augmented-image-{int(datetime.datetime.now().timestamp())}")
