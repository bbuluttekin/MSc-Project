import numpy as np
from tensorflow import keras


def img_to_array(path_for_class_one, path_for_class_zero, image_size=225):
    """Helper fuction to create numpy array from the images.

    Arguments:
        path_for_class_one {Path object} -- Path to binary class 1
        path_for_class_zero {Path object} -- Path to binary class 0

    Keyword Arguments:
        image_size {int} -- Size for the square image reshape (default: {225})

    Returns:
        Tuple -- Tuple of X, y
    """
    X = []
    y = []

    for image in list(path_for_class_one.glob("*.jpeg")):
        img = keras.preprocessing.image.load_img(
            image, target_size=(image_size, image_size))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.array(img).ravel()
        X.append(img)
        y.append(1)

    for image in list(path_for_class_zero.glob("*.jpeg")):
        img = keras.preprocessing.image.load_img(
            image, target_size=(image_size, image_size))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.array(img).ravel()
        X.append(img)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    return X, y
