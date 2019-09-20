"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: main.py
    Program: Main file to run the project
"""
import os

# This is to select the graphic card to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model

# Additional libraries
from ResNet import ResNet
from PrepareTrainTest import PrepareTrainTest
from AuxFunctions import AuxFunctions

# Needed inputs in order to run the network
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", type=int, help="Number of epochs.")
parser.add_argument("--learningrate",
                    "-lr",
                    type=float,
                    help="Value of learning rate.")
parser.add_argument("--batch",
                    "-b",
                    type=int,
                    help="Numbers of batch.",
                    default=10)
parser.add_argument("--datasets", "-d", type=str, help="Path for h5py files.")
parser.add_argument("--traintest",
                    "-tt",
                    type=str,
                    help="Name of the train test validation hdf5 file.")
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
args = parser.parse_args()

epochs = args.epochs
lr = args.learningrate
batch_size = args.batch
path_datasets = args.datasets
file_name = args.traintest
opt = args.optimizer
patience = args.patience

if epochs <= 0:
    print("Error! The number of epochs must be greater than 0")
    print("Please, run main.py -h to see the options")
    sys.exit(-1)
elif batch_size <= 0:
    print("Error! Batch size must be greater than 0")
    print("Please, run main.py -h to see the options")
    sys.exit(-1)
elif opt != 0 and opt != 1 and opt != 2:
    print("Error! The optimizer must be 0, 1 or 2")
    print("Please, run main.py -h to see the options")
    sys.exit(-1)
elif patience <= 0:
    print("Error! Patience must be greater than 0")
    print("Please, run main.py -h to see the options")
    sys.exit(-1)

#  Prepare train-test data section

ptt = PrepareTrainTest(path_datasets, file_name)

if os.path.isdir(path_datasets):
    # PrepareTrainTest.createTrainTestH5PY(malignant_equalized, benign_equalized)
    X_train, X_test, X_val, y_train, y_test, y_val = ptt.readDataH5PY()
else:
    print("Error! H5PY file does not exists...")
    sys.exit(-1)

rn = ResNet(lr, opt, batch_size, epochs)
# Start the section where the model is built
model = rn.buildModel()

if opt == 0:
    type_opt = "Adam"
elif opt == 1:
    type_opt = "RMSprop"
elif opt == 2:
    type_opt = "SGD"

# Creation of the model's callbacks Tensorboard to create the graphs with the results and EarlyStopping to stop the training when it is not improving
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

history = rn.trainModel(model, X_train, y_train, X_val, y_val, callbacks)

print("-" * 40)
print("Summary:")
print("\t optimizer -> " + type_opt)
print("\t lr -> ", lr)
print("\t epochs -> ", epochs)
print("\t batch size -> ", batch_size)
print("\t patience -> ", patience)
print("-" * 40)

rn.evaluateModel(model, X_test, y_test)

if os.path.exists("./results/") == False:
    print("Creating results directory...")
    os.mkdir("./results/")

ax = AuxFunctions(model, history, "results/", type_opt, batch_size)
ax.create_confusion_matrix(X_test, y_test)
ax.create_plots_train_test()
ax.saveWeights()
