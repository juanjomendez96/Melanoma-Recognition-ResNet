"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: AuxFunctions.py
    Program: File that contains the additional functions to make the network works
"""

import tensorflow as ts
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn.metrics import confusion_matrix

"""
    Name: AuxFunction
    
    Description: This class has been created in order to keep the additional functions needed to create, train and evaluate the ResNet model.
"""

class AuxFunctions:

    """
        Name: create_confusion_matrix

        Inputs: - model: ResNet model already trained and compiled.
                - X_test: Array that keeps the test data of the model.
                - y_test: Array that keeps the test labels of the model.

        Returns: None.

        Description: This function has been created in order to show the confusion matrix of the model. Here we have used an additional library in order to get a confusion matrix image.
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
        Name: create_plots_train_test

        Inputs: - history: Information about the trained model.

        Returns: None.

        Description: This function creates a graphics with the train and validation losses. It creates an image with that values.
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
        plt.legend(["train", "validation"], loc="lower left")
        plt.savefig("train-validation.png")

    """
        Name: saveWeights

        Inputs: - model: ResNet model.
                - path: Path where to keep the weights of the trained model.

        Returns: None.

        Description: The aim of this function is to save the weights of the model after the training has been completed. 
    """

    def saveWeights(model, path):
        print("-" * 40)
        print("Saving weights...")
        print("-" * 40)
        model.save_weights(path + "best_weights.hdf5")

