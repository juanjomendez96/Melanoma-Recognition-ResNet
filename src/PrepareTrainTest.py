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
"""
    Name: PrepareTrainTest
    
    Description: This class has been created in order split the input data into train, test and validation for the training.
"""


class PrepareTrainTest:
    def __init__(self, main_path, file_name):
        self.main_path = main_path
        self.file_name = file_name

    """
        Name: createTrainTestH5PY

        Inputs: - path: Path of the hdf5 file.
                - name_file: String for the name of the hdf5 file that keeps the train, test and validation data.
                - malignant_images: Array with the malignant images.
                - benign_images: Array with the benign images.

        Returns: None.

        Description: This function splits the recieved data into train, test and validation. At the end, it saves the data into a hdf5 file.
    """
    def createTrainTestH5PY(self, malignant_images, benign_images):
        print("-" * 40)
        print("Splitting data into train, test and validation...")
        print("-" * 40)
        # Concatenate all data and create labels
        all_data = np.vstack((malignant_images, benign_images))
        labels = np.concatenate(
            (np.zeros(len(malignant_images)), np.ones(len(benign_images))))

        # Do train test split in order to get the training, test and validation data
        X_train, X_test, y_train, y_test = train_test_split(all_data,
                                                            labels,
                                                            test_size=0.4,
                                                            stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_test,
                                                        y_test,
                                                        test_size=0.5,
                                                        stratify=y_test)

        # Make labels categorical
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        y_val = np_utils.to_categorical(y_val)

        print("-" * 40)
        print("Creating h5py train test file...")
        print("-" * 40)
        dataset_output_path = self.main_path + self.file_name
        with h5py.File(dataset_output_path, "w") as hdf:
            hdf.create_dataset("X_train", data=X_train, compression="gzip")
            hdf.create_dataset("y_train", data=y_train, compression="gzip")
            hdf.create_dataset("X_test", data=X_test, compression="gzip")
            hdf.create_dataset("y_test", data=y_test, compression="gzip")
            hdf.create_dataset("X_val", data=X_val, compression="gzip")
            hdf.create_dataset("y_val", data=y_val, compression="gzip")

    """
        Name: readDataH5PY

        Inputs: - path: Path of the hdf5 file.
                - name_file: String for the name of the hdf5 file that keeps the train, test and validation data.

        Returns: - dataset: A dictionary that keeps the value of the train, test and validation data.

        Description: This function reads the data for the training from a hdf5 file and returns a dictionary that keeps them.
    """
    def readDataH5PY(self):
        print("-" * 40)
        print("Reading h5py train test file...")
        print("-" * 40)
        dataset = h5py.File(self.main_path + self.file_name, "r")
        return (
            dataset["X_train"][()],
            dataset["X_test"][()],
            dataset["X_val"][()],
            dataset["y_train"][()],
            dataset["y_test"][()],
            dataset["y_val"][()],
        )
