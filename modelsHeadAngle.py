# coding=utf-8

''' Definición del modelo de 'Neuronal Network' usado '''

# Importación de librerías necesarias
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten

def modelo(width, height, depth, regress = False):

    inputShape = (width, height, depth)
    filtros = [16, 32, 64]                                                          # Pongo 3 filtros de momento porque el volumen de datos
                                                                                    # es proporcional a los filtros (+ Datos = + Filtros).


    model = Sequential()

    # Filtro 1
    model.add(Conv2D(filtros[0], (3, 3), padding = 'same', input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Filtro 2
    model.add(Conv2D(filtros[1], (3, 3), padding = 'same', input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Filtro 3
    model.add(Conv2D(filtros[2], (3, 3), padding = 'same', input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (2, 2)))


    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.5))

    model.add(Dense(4))
    model.add(Activation("relu"))

    if regress:
        model.add(Dense(1, activation = "linear"))

    return model