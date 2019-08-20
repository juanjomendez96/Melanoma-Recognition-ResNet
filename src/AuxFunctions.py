"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: AuxFunctions.py
    Program: File that contains the additional functions to make the network works
"""
import tensorflow as ts
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Activation, Flatten, Conv2D, BatchNormalization
from keras import layers


class AuxFunctions:
    """
        Method that is used in order to build the ResNet model
    """

    def identity_block(X, f, filters, stage, block):
        # Retrieve filters
        F1, F2, F3 = filters

        # Create the shortcut
        shortcutX = X

        # First component of the main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        # Second component of the main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        # Third component of the main path
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        X = layers.Add()([X, shortcutX])
        x = Activation("relu")(X)

        return X

    """
        Method that is used in order to build the ResNet model
    """

    def convolutional_block(X, f, filters, stage, block, s=2):
        F1, F2, F3 = filters

        # Create the shortcut
        shortcutX = X

        # Main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        # Second component of the main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        # Third component of the main path
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1))(X)
        X = BatchNormalization(axis=3)(X)

        # ShortCut path
        shortcutX = Conv2D(F3, (1, 1), strides=(s, s))(shortcutX)
        shortcutX = BatchNormalization(axis=3)(shortcutX)

        # Add shortcut value to the main path and pass it through a RELU activation
        X = layers.Add()([X, shortcutX])
        X = Activation("relu")(X)

        return X

    """
        Method that creates the confusion matrix and save it into a PNG image
    """

    def create_confusion_matrix(model, X_test, y_test):
        print("-" * 40)
        print("Creating confusion matrix...")
        print("-" * 40)
        y_test_confusion_matrix = np.argmax(y_test, axis=1)

        prediction = model.predict(X_test)
        y_pred = np.argmax(prediction, axis=1)

        matrix = confusion_matrix(y_test_confusion_matrix, y_pred)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        sns.heatmap(matrix, annot=True, ax=ax)

        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Matriz de confusión")
        ax.xaxis.set_ticklabels(["malignant", "benign"])
        ax.yaxis.set_ticklabels(["malignant", "benign"])

        plt.savefig("confusion_matrix.png")

    """
        Method that creates a summary of the training process loss
    """

    def create_plots_train_test(history):
        print("-" * 40)
        print("Creating train test plot...")
        print("-" * 40)
        plt.clf()
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"], label="test")
        plt.xlabel("Train epochs")
        plt.ylabel("Error")
        plt.legend(["train", "test"], loc="lower left")
        plt.savefig("train-test.png")

    """
        Method that saves the weights of the model
    """

    def saveWeights(model, path):
        print("-" * 40)
        print("Saving weights...")
        print("-" * 40)
        model.save_weights(path + "best_weights.hdf5")

