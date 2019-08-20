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

# import cv2
from tqdm import tqdm
from keras.preprocessing import image


class PrepareData:
    """
        Method that creates an hdf5 file in order to keep classified the images
    """

    def createH5PY(path_images, path_datasets):
        files = []
        with h5py.File(path_datasets + "dataset.hdf5", "w") as hdf:
            benign_group = hdf.create_group("benign_images")
            malignant_group = hdf.create_group("malignant_images")
            print("Creating h5py file...")
            # Read all images from directory
            for r, d, f in os.walk(path_images):
                for file in tqdm(f):
                    if ".json" in file:
                        paths = os.path.join(r, file)
                        with open(paths) as json_file:
                            data_json = json.load(json_file)
                            if (
                                data_json["meta"]["clinical"]["benign_malignant"]
                                == "malignant"
                            ):
                                data = image.load_img(
                                    paths[:-4] + "jpg", target_size=[512, 512]
                                )
                                malignant_group.create_dataset(
                                    file[:-4] + "jpg", data=data, compression="gzip"
                                )
                            elif (
                                data_json["meta"]["clinical"]["benign_malignant"]
                                == "benign"
                            ):
                                data = image.load_img(
                                    paths[:-4] + "jpg", target_size=[512, 512]
                                )
                                benign_group.create_dataset(
                                    file[:-4] + "jpg", data=data, compression="gzip"
                                )

    """
        Method that reads all the images and returns them in order to start the train test split process
    """

    def readDataH5PY(path):
        malignant_images = []
        benign_images = []
        with h5py.File(path + "dataset.hdf5", "r") as hdf:
            malignant_items = list(hdf.get("malignant_images").items())

            print("-" * 40)
            print("Getting malignant images...")
            print("-" * 40)

            for items in tqdm(malignant_items):
                malignant_images.append(
                    np.array(hdf.get("malignant_images").get(items[0]))
                )

            print("-" * 40)
            print("Getting benign images...")
            print("-" * 40)

            benign_items = list(hdf.get("benign_images").items())
            for items in tqdm(benign_items):
                benign_images.append(np.array(hdf.get("benign_images").get(items[0])))

        return malignant_images, benign_images

    """
        This method equalizes the images in order to get a better results in training process

    def equalizeImages(images):
        print("-"*40)
        print("Equalizing images...")
        print("-"*40)

        result = []
        for image in tqdm(images):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(cv2.dilate(image, np.ones((5,5), np.uint8), iterations = 2))
            image = cv2.erode(image,  np.ones((5,5), np.uint8), iterations= 2)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            result.append(image)
        return result
    """
