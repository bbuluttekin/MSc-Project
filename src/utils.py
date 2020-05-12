import datetime

import boto3
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
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original image', fontsize=15)
    plt.axis('off')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image', fontsize=15)
    plt.axis('off')
    plt.imshow(augmented)
    plt.savefig(f"augmented-image-{int(datetime.datetime.now().timestamp())}",
                dpi=100,
                bbox_inches='tight',
                pad_inches=0.1)


def plot_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


def download_from_s3(bucket_name, filename, targetname=None):
    s3_client = boto3.client('s3')
    if targetname:
        with open(targetname, "wb") as f:
            s3_client.download_fileobj(
                bucket_name,
                filename,
                f
            )
    else:
        with open(filename, "wb") as f:
            s3_client.download_fileobj(
                bucket_name,
                filename,
                f
            )


def upload_to_s3(bucket_name, filename, targetname=None):
    s3_client = boto3.client('s3')
    if targetname:
        with open(filename, "rb") as f:
            s3_client.upload_fileobj(
                f,
                bucket_name,
                targetname
            )
    else:
        with open(filename, "rb") as f:
            s3_client.upload_fileobj(
                f,
                bucket_name,
                filename
            )
