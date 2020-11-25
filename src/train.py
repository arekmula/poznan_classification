import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src import utils


def data_processing(x: np.ndarray) -> np.ndarray:
    images_resized = []
    for image in x:
        image_resized = cv2.resize(image, (800, 600))  # TODO: Check smaller sizes
        images_resized.append(image_resized)

    return np.asarray(images_resized)


def create_vocab_model(train_descriptors, nb_words=20):
    kmeans = cluster.KMeans(n_clusters=nb_words, random_state=42)
    kmeans.fit(train_descriptors)
    pickle.dump(kmeans, open("vocab_model.p", "wb"))


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)  # KMeans returns indexes of words
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(data: np.ndarray,
                            feature_detector_descriptor,
                            vocab_model
                            ) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def train():
    data_path = Path('../../train/')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    images, labels = utils.load_dataset(data_path)

    images = data_processing(images)

    # utils.show_images_and_labels(x, y)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.6,
                                                                            random_state=42, stratify=labels)
    test_images, valid_images, test_labels, valid_labels = train_test_split(test_images, test_labels, train_size=0.5,
                                                                            random_state=42, stratify=test_labels)

    feature_detector_descriptor = cv2.AKAZE_create()  # TODO: Check different parameters

    # Uncomment if you want to create new vocab model
    # train_descriptors = []
    # for image in train_images:
    #     # TODO: Moze sprawdz maske binarna na srodku obrazu
    #     # Drugim argumentem może być maska binarna, która służy do zawężenia obszaru z którego
    #     # uzyskujemy punkty kluczowe/deskryptor – jako, że nam akurat zależy na całym obrazie,
    #     # podaliśmy tam wartość None.
    #     for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
    #         train_descriptors.append(descriptor)

    # Uncomment if you want to create new vocab model
    # create_vocab_model(train_descriptors)

    with Path('vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)

    X_train = apply_feature_transform(train_images, feature_detector_descriptor, vocab_model)
    y_train = train_labels

    X_test = apply_feature_transform(test_images, feature_detector_descriptor, vocab_model)
    y_test = test_labels

    X_valid = apply_feature_transform(valid_images, feature_detector_descriptor, vocab_model)
    y_valid = valid_labels

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    # print(classifier.score(X_train, y_train))
    print(classifier.score(X_test, y_test))
    print(classifier.score(X_valid, y_valid))


if __name__ == "__main__":
    train()
