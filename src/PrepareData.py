"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: PrepareData.py
    Program: File to prepare the images and save them onto a h5py file
"""
import numpy as np
import h5py
import os
import json

from tqdm import tqdm  # This library is used to know the percentage of images already classified and saved
"""
    Name: PrepareData
    
    Description: This class has been created in order to classify the images before the training.
"""


class PrepareData:
    def __init__(self, path_images, main_path):
        self.path_images = path_images
        self.main_path = main_path

    """
        Name: createH5PY

        Inputs: - path_images: Path of the images.
                - path_datasets: Path where it is going to save the images.

        Returns: None.

        Description: This function classifies the images into malignant and benign. After that, creates a hdf5 file in order to save them.
    """
    def createH5PY(self):
        files = []
        with h5py.File(self.path_datasets + "dataset.hdf5", "w") as hdf:
            benign_group = hdf.create_group("benign_images")
            malignant_group = hdf.create_group("malignant_images")
            print("Creating h5py file...")
            # Read all images from directory
            for r, d, f in os.walk(self.path_images):
                for file in tqdm(f):
                    if ".json" in file:
                        paths = os.path.join(r, file)
                        with open(paths) as json_file:
                            data_json = json.load(json_file)
                            if (data_json["meta"]["clinical"]
                                ["benign_malignant"] == "malignant"):
                                data = image.load_img(paths[:-4] + "jpg",
                                                      target_size=[512, 512])
                                malignant_group.create_dataset(
                                    file[:-4] + "jpg",
                                    data=data,
                                    compression="gzip")
                            elif (data_json["meta"]["clinical"]
                                  ["benign_malignant"] == "benign"):
                                data = image.load_img(paths[:-4] + "jpg",
                                                      target_size=[512, 512])
                                benign_group.create_dataset(file[:-4] + "jpg",
                                                            data=data,
                                                            compression="gzip")

    """
        Name: readDataH5PY

        Inputs: - path: Path of the hdf5 file.

        Returns: - malignant_images: An array with all the malignant images.
                 - benign_images: An array with the benign images.
        
        Description: This function reads the hdf5 file and returns two array with the different types of images.
    """
    def readDataH5PY(self):
        malignant_images = []
        benign_images = []
        with h5py.File(self.main_path + "dataset.hdf5", "r") as hdf:
            malignant_items = list(hdf.get("malignant_images").items())

            print("-" * 40)
            print("Getting malignant images...")
            print("-" * 40)

            for items in tqdm(malignant_items):
                malignant_images.append(
                    np.array(hdf.get("malignant_images").get(items[0])))

            print("-" * 40)
            print("Getting benign images...")
            print("-" * 40)

            benign_items = list(hdf.get("benign_images").items())
            for items in tqdm(benign_items):
                benign_images.append(
                    np.array(hdf.get("benign_images").get(items[0])))

        return malignant_images, benign_images
