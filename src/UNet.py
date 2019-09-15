import tensorflow as ts
import numpy as np

import keras

from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, UpSampling2D, Concatenate

from keras import Model
from keras import optimizers
from keras import losses

from AuxFunctions import AuxFunctions


class UNet:
    def train_model(model, X_train, y_train, X_val, y_val):
        model.fit(X_train, batch_size=10, epochs=10, validation_data=(X_val))

    def evaluate_model(model, X_test, y_test):
        score = model.evaluate(X_test, y_test, verbose=0)
        return score

    def predict_model(model, X_test):
        predicted = model.predict(X_test)
        return predicted

    def buildModel():
        input_shape = (512, 512, 3)

        inputs = Input(input_shape)

        # Downsampling block
        conv1 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(inputs)
        conv1 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool1)
        conv2 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool2)
        conv3 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool3)
        conv4 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottleneck block
        conv5 = Conv2D(
            1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool4)
        conv5 = Conv2D(
            1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Upsampling block
        up6 = UpSampling2D(size=(2, 2))(drop5)
        up6 = Conv2D(
            512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(up6)
        merge6 = Concatenate(axis=3)([drop4, up6])
        conv6 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge6)
        conv6 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(
            256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(up7)
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge7)
        conv7 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = Conv2D(
            128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(up8)
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge8)
        conv8 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = Conv2D(
            64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(up9)
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge9)
        conv9 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)

        conv9 = Conv2D(
            2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv10 = Conv2D(1, 3, activation="softmax")(conv9)

        model = Model(input=inputs, output=conv10)
        model.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss=losses.binary_crossentropy,
            metrics=["accuracy"],
        )

        return model
