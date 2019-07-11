import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import glob
from pathlib import Path
import gc
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from collections import Counter

data_dir = Path("/content/chest_xray")

train_dir = data_dir / "train"

test_dir = data_dir / "test"

val_dir = data_dir / "val"

train_dir_normal = train_dir / "NORMAL"
train_dir_pne = train_dir / "PNEUMONIA"

test_dir_normal = test_dir / "NORMAL"
test_dir_pne = test_dir / "PNEUMONIA"

normal_val = val_dir / "NORMAL"
val_dir_pne = val_dir / "PNEUMONIA"

IMAGE_SIZE = 225


def img_to_array(path_for_class_one, path_for_class_zero):
    """
    Documentation needed.
    """
    X = []
    y = []

    for image in list(path_for_class_one.glob("*.jpeg")):
        img = keras.preprocessing.image.load_img(
            image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.array(img).ravel()
        X.append(img)
        y.append(1)

    for image in list(path_for_class_zero.glob("*.jpeg")):
        img = keras.preprocessing.image.load_img(
            image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.array(img).ravel()
        X.append(img)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    return X, y


X, y = img_to_array(train_dir_pne, train_dir_normal)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=3, verbose=2)
clf.fit(X_train, y_train)

X_test, y_test = img_to_array(test_dir_pne, test_dir_normal)

print(accuracy_score(y_test, clf.predict(X_test)))
