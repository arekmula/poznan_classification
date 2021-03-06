import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn import svm

import utils


def data_processing(x: np.ndarray) -> np.ndarray:
    print("Processing data")
    images_resized = []
    for image in x:
        image_resized = cv2.resize(image, (1200, 1000))
        images_resized.append(image_resized)

    return np.asarray(images_resized)


def create_vocab_model(train_descriptors, nb_words=64):
    print("Creating vocab model")
    kmeans = cluster.KMeans(n_clusters=nb_words, random_state=42)
    kmeans.fit(train_descriptors)
    pickle.dump(kmeans, open("vocab_model.p", "wb"))
    return kmeans


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
    print("Applying feature transform")
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def train():
    data_path = Path('../../train_poznan_classification/')  # You can change the path here
    images, labels = utils.load_dataset(data_path)

    images = data_processing(images)

    # uncomment this function if you want to see your dataset
    # utils.show_images_and_labels(x, y)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,
                                                                            random_state=42, stratify=labels)

    feature_detector_descriptor = cv2.AKAZE_create()

    train_descriptors = []
    for image in train_images:
        for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
            train_descriptors.append(descriptor)
    vocab_model = create_vocab_model(train_descriptors)

    X_train = apply_feature_transform(train_images, feature_detector_descriptor, vocab_model)
    y_train = train_labels

    X_test = apply_feature_transform(test_images, feature_detector_descriptor, vocab_model)
    y_test = test_labels

    classifier = svm.SVC(kernel="poly")
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

    pickle.dump(classifier, open("clf.p", "wb"))


if __name__ == "__main__":
    train()
