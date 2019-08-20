"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: main.py
    Program: Main file to run the project
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

# Additional libraries
from ResNet import ResNet
from PrepareTrainTest import PrepareTrainTest
from PrepareData import PrepareData
from AuxFunctions import AuxFunctions

# Python libraries
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.callbacks import TensorBoard, EarlyStopping
import sys

"""
    Section for input params
"""
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", type=int, help="Number of epochs.")
parser.add_argument("--learningrate", "-lr", type=float, help="Value of learning rate.")
parser.add_argument("--batch", "-b", type=int, help="Numbers of batch.", default=10)
parser.add_argument("--datasets", "-d", type=str, help="Path for h5py files.")
parser.add_argument("--images", "-i", type=str, help="Path lesion images.")
parser.add_argument(
    "--optimizer",
    "-o",
    type=int,
    help="Optimizer used: 0.Adam 1.RMSprop 2.SGD",
    default=0,
)
parser.add_argument(
    "--patience",
    "-p",
    type=int,
    help="Number of epochs without improving validation loss.",
)
parser.add_argument(
    "--file", "-f", type=int, help="Images to train with. 0. No equalized 1. Equalized"
)
args = parser.parse_args()

epochs = args.epochs
lr = args.learningrate
batch_size = args.batch
path_datasets = args.datasets
path_images = args.images
opt = args.optimizer
patience = args.patience
is_equalized = args.file

""" 
    Section to get all images before training

# Create dataset.hf5py
PrepareData.createH5PY(path_images, path_datasets)
# Get images from dataset

malignant_images, benign_images = PrepareData.readDataH5PY(path_datasets)
malignant_equalized = PrepareData.equalizeImages(malignant_images)
benign_equalized = PrepareData.equalizeImages(benign_images)
"""

"""
    Prepare train-test data section
"""

if is_equalized == 0:
    name_file = "train-test-val.hdf5"
elif is_equalized == 1:
    name_file = "train-test-val-equalized.hdf5"
else:
    print("Error! H5PY file does not exists...")
    sys.exit(-1)
# PrepareTrainTest.createTrainTestH5PY(path_datasets, name_file, malignant_equalized, benign_equalized)
X_train, X_test, X_val, y_train, y_test, y_val = PrepareTrainTest.readDataH5PY(
    path_datasets, name_file
)

"""
    ResNet model section
"""
model = ResNet.buildModel(lr, opt)

if opt == 0:
    type_opt = "Adam"
elif opt == 1:
    type_opt = "RMSprop"
elif opt == 2:
    type_opt = "SGD"

callbacks = [
    TensorBoard(
        log_dir="./logs/" + type_opt + "_b" + str(batch_size) + "/",
        write_images=True,
        write_graph=True,
        update_freq="epoch",
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        verbose=1,
        restore_best_weights=True,
    ),
]

history = ResNet.train_model(
    model, X_train, y_train, X_test, y_test, X_val, y_val, epochs, batch_size, callbacks
)

print("-" * 40)
print("Summary:")
print("\t optimizer -> " + type_opt)
print("\t lr -> ", lr)
print("\t epochs -> ", epochs)
print("\t batch size -> ", batch_size)
print("\t patience -> ", patience)
print("-" * 40)

ResNet.evaluateModel(model, X_test, y_test)

AuxFunctions.create_confusion_matrix(model, X_test, y_test)
AuxFunctions.create_plots_train_test(history)
AuxFunctions.saveWeights(model, "logs/")
