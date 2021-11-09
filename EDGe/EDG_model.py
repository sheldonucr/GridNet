#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate
from tensorflow.keras import Model, regularizers
from tensorflow.keras.regularizers import l2

class encoder(Model):

    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = (
                Conv2D(16, (3,3), activation='relu',padding='SAME'),
                Conv2D(16, (3,3), activation='relu',padding='SAME'),
                Conv2D(16, (3,3), activation='relu',padding='SAME'),
            )
        self.max1 = MaxPooling2D(2, padding='same')

        self.conv2 = (
                Conv2D(32, (3,3), activation='relu',padding='SAME'),
                Conv2D(32, (3,3), activation='relu',padding='SAME'),
                Conv2D(32, (3,3), activation='relu',padding='SAME'),
            )
        self.max2 = MaxPooling2D(2, padding='same')

        self.conv3 = (
                Conv2D(64, (5,5), activation='relu',padding='SAME'),
                Conv2D(64, (5,5), activation='relu',padding='SAME'),
                Conv2D(64, (5,5), activation='relu',padding='SAME'),
            )
        self.max3 = MaxPooling2D(2, padding='same')

    def call(self, input):

        x = input

        x = self.conv1[0](x)
        x = self.conv1[1](x)
        x = self.conv1[2](x)
        x = self.max1(x)
        layer1 = x

        x = self.conv2[0](x)
        x = self.conv2[1](x)
        x = self.conv2[2](x)
        x = self.max2(x)
        layer2 = x

        x = self.conv3[0](x)
        x = self.conv3[1](x)
        x = self.conv3[2](x)
        x = self.max3(x)
        layer3 = x

        return (input,layer1,layer2,layer3)

class decoder(Model):

    def __init__(self):
        super(decoder, self).__init__()
        self.conv0 = (
                Conv2DTranspose(64, (7,7), activation='relu',padding='SAME'),
                Conv2DTranspose(64, (7,7), activation='relu',padding='SAME'),
                Conv2DTranspose(64, (7,7), activation='relu',padding='SAME'),
            )
        self.max1 = UpSampling2D(2)
        self.conv1 = (
                Conv2DTranspose(32, (7,7), activation='relu',padding='SAME'),
                Conv2DTranspose(32, (7,7), activation='relu',padding='SAME'),
                Conv2DTranspose(32, (7,7), activation='relu',padding='SAME'),
            )
        self.max2 = UpSampling2D(2)
        self.conv2 = (
                Conv2DTranspose(16, (3,3), activation='relu',padding='SAME'),
                Conv2DTranspose(16, (3,3), activation='relu',padding='SAME'),
                Conv2DTranspose(16, (3,3), activation='relu',padding='SAME'),
            )
        self.max3 = UpSampling2D(2)
        self.conv3 = Conv2DTranspose(1, (3,3), activation='relu',padding='SAME')

    def call(self, input_and_layers):

        input, layer1, layer2, layer3 = input_and_layers
        x = layer3

        x = self.conv0[0](x)
        x = self.conv0[1](x)
        x = self.conv0[2](x)

        x = self.max1(x)
        x = Concatenate()([x, layer2])
        x = self.conv1[0](x)
        x = self.conv1[1](x)
        x = self.conv1[2](x)

        x = self.max2(x)
        x = Concatenate()([x, layer1])
        x = self.conv2[0](x)
        x = self.conv2[1](x)
        x = self.conv2[2](x)

        x = self.max3(x)
        x = Concatenate()([x, input])
        x = self.conv3(x)

        return x

class autoencoder(Model):

    def __init__(self):
        super(autoencoder, self).__init__()
        self._encoder = encoder()
        self._decoder = decoder()

    def call(self, x):
        input_and_layers = self._encoder(x)
        x = self._decoder(input_and_layers)
        return x