"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: PrepareTrainTest.py
    Program: File that creates the train test file in order to train the model with
"""
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils


class PrepareTrainTest:
    """
        Method that creates the train test hdf5 file
    """

    def createTrainTestH5PY(path, name_file, malignant_images, benign_images):
        print("-" * 40)
        print("Splitting data into train, test and validation...")
        print("-" * 40)
        # Concatenate all data and create labels
        all_data = np.vstack((malignant_images, benign_images))
        labels = np.concatenate(
            (np.zeros(len(malignant_images)), np.ones(len(benign_images)))
        )

        # Do train test split in order to get the training, test and validation data
        X_train, X_test, y_train, y_test = train_test_split(
            all_data, labels, test_size=0.4, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, stratify=y_test
        )

        # Make labels categorical
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        y_val = np_utils.to_categorical(y_val)

        print("-" * 40)
        print("Creating h5py train test file...")
        print("-" * 40)
        dataset_output_path = path + name_file
        with h5py.File(dataset_output_path, "w") as hdf:
            hdf.create_dataset("X_train", data=X_train, compression="gzip")
            hdf.create_dataset("y_train", data=y_train, compression="gzip")
            hdf.create_dataset("X_test", data=X_test, compression="gzip")
            hdf.create_dataset("y_test", data=y_test, compression="gzip")
            hdf.create_dataset("X_val", data=X_val, compression="gzip")
            hdf.create_dataset("y_val", data=y_val, compression="gzip")

    """
        Method that reads the train test hdf5 file
    """

    def readDataH5PY(path, name_file):
        print("-" * 40)
        print("Reading h5py train test file...")
        print("-" * 40)
        dataset = h5py.File(path + name_file, "r")
        return (
            dataset["X_train"][()],
            dataset["X_test"][()],
            dataset["X_val"][()],
            dataset["y_train"][()],
            dataset["y_test"][()],
            dataset["y_val"][()],
        )

