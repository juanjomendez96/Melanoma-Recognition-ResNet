"""
    Project: Melanoma recognition
    Author: Juan José Méndez Torrero
    File: ResNet.py
    Program: File that creates the model and trains it with the images already processed
"""
import tensorflow as tf
import numpy as np

import keras
from keras.layers import (
    Dense,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    ZeroPadding2D,
    Input,
    AveragePooling2D,
)
from keras import Model
from keras import optimizers
from keras import losses
from keras import layers

from AuxFunctions import AuxFunctions

"""
    Name: ResNet
    
    Description: This class has been created in order to create, train and evaluate the model of the ResNet.
"""

class ResNet:
    def __init__(self, lr, opt, batch_size, epochs):
        self.lr = lr
        self.opt = opt
        self.batch_size = batch_size
        self.epochs = epochs
        
    """
        Name: trainModel

        Inputs: - model: Keras model that creates the residual network.
                - X_train: Array that keeps the input data for the training.
                - y_train: Array that keeps the label of the input data for the training.
                - X_val: Array that keeps the data of the validation for the training.
                - y_val: Array that keeps the labels of the validation for the training.
                - epochs: Integer that says how many epoch has to be trained the model.
                - batch_size: Integer that keeps the size of the batch for the training.
                - callbacks: Array that keeps the information for the early stopping and tensorboard's graphics.

        Returns: - history: Array that keeps the model information after being trained.

        Description: This function train the already compiled model and return the results of the training.
    """

    def trainModel(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        callbacks,
    ):
        history = model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
        return history

    """
        Name: evaluateModel

        Inputs: - model: Model of the residual network.
                - X_test: Array with the images of the test.
                - y_test: Array with label of the test.

        Returns: None.

        Description: This function has been created in order to evaluate the already trained model.
    """

    def evaluateModel(self, model, X_test, y_test):
        print("-" * 40)
        print("Evaluating model...")
        print("-" * 40)

        score = model.evaluate(X_test, y_test, verbose=0)

        print("-" * 40)
        print("Test loss-> ", score[0])
        print("Test accuracy-> ", score[1])
        print("-" * 40)

    """
        Name: identityBlock

        Inputs: - X: Layers of the model.
                - f: Size of the convolutional layer's kernel.
                - filters: Array with the filter's size of the convolotional layers.
                - block: String that represents the name of the block.

        Returns: - X: Layer of the model.

        Description: This function has been created in order to create an identity block that creates a shortcut that skips one or more layers.
    """

    def identityBlock(self, X, f, filters, block):
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
        Name: convolutionalBlock

        Inputs: - X: Layers of the model.
                - f: Size of the convolutional layer's kernel.
                - filters: Array with the filter's size of the convolotional layers.
                - block: String that represents the name of the block.
                - stride: Stride of the layer that combines the main path with the shortcuts.

        Returns: - X: Layer of the model.

        Description: This function has has been created in order to combine the main path of the model with the shortcuts.
    """

    def convolutionalBlock(self, X, f, filters, block, s=2):
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
        Name: buildModel

        Inputs: - lr: Learning rate value in order to build the model.
                - opt: Type of optimazer in order to build the model.

        Returns: - model: Model already built.

        Description: This function has been created in order to build the residual network model. It returns the model to use it in other functions.
    """

    def buildModel(self, lr, opt):
        print("-" * 40)
        print("Creating model...")
        print("-" * 40)
        input_shape = (512, 512, 3)
        classes = 2

        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(32, (7, 7), strides=(2, 2))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        X = self.convolutionalBlock(
            X, f=3, filters=[64, 64, 256], block="a", s=1
        )
        X = self.identityBlock(X, 3, [64, 64, 256], block="b")
        X = self.identityBlock(X, 3, [64, 64, 256], block="c")

        X = self.convolutionalBlock(
            X, f=3, filters=[128, 128, 512], block="a", s=1
        )
        X = self.identityBlock(X, 3, [128, 128, 512], block="b")
        X = self.identityBlock(X, 3, [128, 128, 512], block="c")
        X = self.identityBlock(X, 3, [128, 128, 512], block="d")

        X = AveragePooling2D((2, 2))(X)

        X = Conv2D(16, (3, 3), strides=(2, 2))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        X = Flatten()(X)

        X = Dense(classes, activation="softmax")(X)

        model = Model(inputs=X_input, outputs=X)
        if self.opt == 0:
            model.compile(
                optimizer=optimizers.Adam(
                    lr=self.lr,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=None,
                    decay=0.0,
                    amsgrad=False,
                ),
                loss=losses.binary_crossentropy,
                metrics=["accuracy"],
            )
        elif self.opt == 1:
            model.compile(
                optimizer=optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=None, decay=0.0),
                loss=losses.binary_crossentropy,
                metrics=["accuracy"],
            )
        elif self.opt == 2:
            model.compile(
                optimizer=optimizers.SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=True),
                loss=losses.binary_crossentropy,
                metrics=["accuracy"],
            )

        model.summary()
        return model

