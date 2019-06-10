# coding=utf-8

''' Script que permite cargar las imágenes y los atributos de estas del conjunto de datos '''

# Importación de librerías necesarias
import pandas as pd                                                                 # Librería que permite la lectura de CSV.
import numpy as np                                                                  # Líbreria para el manejo de arrays.
import imageio                                                                      # Librería para el manejo de imágenes.
import glob
import os                                                                           # Librería para utilitzar funciones de SO.


# Función que devuelve el conjunto de datos, de inputPath, en formato 'data frame'.
def load_images_attributes(inputPath):

    cols = ["Persona", "Fotograma", "AnguloYaw", "AnguloPitch", "AnguloRoll"]       # Inicializa la lista de columnas en el fichero CSV.

    df = pd.read_csv(inputPath, sep = " ", header = None, names = cols)             # Carga el fichero CSV usando 'pandas'.

    personas = df["Persona"].value_counts().keys().tolist()                         # Determina el número de sujeto de la BBDD.
    counts = df["Persona"].value_counts().tolist()                                  # Cuenta el número de datos que contiene cada sujeto                                                                                # postal.

    for (personas, count) in zip(personas, counts):                                 # Bucle de todos los sujetos, con sus
                                                                                    # correspondientes cuentas.

        if count < 25:                                                              # El sujeto que no tenga más de 25 cabezas con
                                                                                    # su código, será eliminado del conjunto de datos, ya
                                                                                    # que si existen pocas cabezas en ese sujeto,
                                                                                    # puede provocar inestabilidad en el sistema.
            idxs = df[df["Persona"] == personas].index
            df.drop(idxs, inplace = True)

    return df


# Función que carga las imágenes de los sujetos y las devuelve en el formato 'array'.
def load_images(df, inputPath):

    images = []                                                                     # Inicialización del vector de imágenes.

    for i in df.index.values:                                                       # Bucle de todos los índices de los sujetos.
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])              # Busca los 'path' de las imágenes de cada cabeza.
        headsPaths = sorted(list(glob.glob(basePath)))                              # Ordena los 'path' encontrados, para asegurar de
                                                                                    # que estos siempre se encuentran en el mismo orden.

        for headPath in headsPaths:                                                 # Bucle que pasa por todas las imágenes de la BBDD
                                                                                    # y las coloca en una lista de imágenes.
            img = imageio.imread(headPath)
            images.append(img)

    vector = np.array(images)                                                       # Se pasa de lista a 'array', pero se redimensiona porque
                                                                                    # la CNN (tensorflow) requiere dimensión 4.
    vectorImagenes = vector.reshape(vector.shape[0], vector.shape[1], vector.shape[2], 1)

    return vectorImagenes
