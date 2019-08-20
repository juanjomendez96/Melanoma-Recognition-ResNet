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

from AuxFunctions import AuxFunctions


class ResNet:
    """
        Method that trains the model and returns its history in order to create a figure with the results of the training
    """

    def train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        epochs,
        batch_size,
        callbacks,
    ):
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
        return history

    """
        Method that evaluates the model with the test data
    """

    def evaluateModel(model, X_test, y_test):
        print("-" * 40)
        print("Evaluating model...")
        print("-" * 40)

        score = model.evaluate(X_test, y_test, verbose=0)

        print("-" * 40)
        print("Test loss-> ", score[0])
        print("Test accuracy-> ", score[1])
        print("-" * 40)

    """
        Function that build and compile the full model with the convolutional and identity blocks
    """

    def buildModel(lr, opt):
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

        X = AuxFunctions.convolutional_block(
            X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1
        )
        X = AuxFunctions.identity_block(X, 3, [64, 64, 256], stage=2, block="b")
        X = AuxFunctions.identity_block(X, 3, [64, 64, 256], stage=2, block="c")

        X = AuxFunctions.convolutional_block(
            X, f=3, filters=[128, 128, 512], stage=2, block="a", s=1
        )
        X = AuxFunctions.identity_block(X, 3, [128, 128, 512], stage=3, block="b")
        X = AuxFunctions.identity_block(X, 3, [128, 128, 512], stage=3, block="c")
        X = AuxFunctions.identity_block(X, 3, [128, 128, 512], stage=3, block="d")

        X = AveragePooling2D((2, 2))(X)

        X = Conv2D(16, (3, 3), strides=(2, 2))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

        X = Flatten()(X)

        X = Dense(classes, activation="softmax")(X)

        model = Model(inputs=X_input, outputs=X)
        if opt == 0:
            model.compile(
                optimizer=optimizers.Adam(
                    lr=lr,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=None,
                    decay=0.0,
                    amsgrad=False,
                ),
                loss=losses.binary_crossentropy,
                metrics=["accuracy"],
            )
        elif opt == 1:
            model.compile(
                optimizer=optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0),
                loss=losses.binary_crossentropy,
                metrics=["accuracy"],
            )
        elif opt == 2:
            model.compile(
                optimizer=optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=True),
                loss=losses.binary_crossentropy,
                metrics=["accuracy"],
            )

        model.summary()
        return model

