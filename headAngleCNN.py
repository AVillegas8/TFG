# coding=utf-8

# Importación de las librerias necesarias
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import modelsHeadAngle
import datasetsHeadAngle
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import lossHeadAngle

# El argumento 'argparse' --dataset permite especificar el path del conjunto de datos desde terminal, sin modificar el script
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, required = True,
                help = "path to input dataset of images")
args = vars(ap.parse_args())

# Construcción del path del archivo de entrada .txt que contiene la información de cada cabeza del conjunto de datos
print("[INFO] loading head attributes...")
inputPath = os.path.sep.join([args["dataset"], "datFinal.txt"])

# Carga de los datos
df = datasetsHeadAngle.load_images_attributes(inputPath)
vectorAngulosYaw = np.array(df["AnguloYaw"])                                        # Vector de ángulos creado con el campo "AnguloYaw"
                                                                                    # del 'df'.

# Carga de las imágenes
print("[INFO] loading head images...")
images = datasetsHeadAngle.load_images(df, args["dataset"])

sumaImagenes = 0                                                                    # Normalización de las imágenes
for i in range (images.shape[0]):
    for j in range (images.shape[1]):
        for k in range (images.shape[2]):
            for q in range (images.shape[3]):
                sumaImagenes += images[i][j][k][q]
mediaImagenes = sumaImagenes / images.size
vectorImagenes = images - mediaImagenes                                             # Vector de imágenes normalizadas


# Partición de los datos en 'training' --> 75% y 'testing' --> 25% (ÁngulosYaw)
trainAttrX1, testAttrX, trainImagesX1, testImagesX = train_test_split(vectorAngulosYaw, vectorImagenes, test_size = 0.25, random_state = 42)

# Partición de los datos de entrenamiento en 'training--> 75% y 'validation' --> 25% (ÁngulosYaw)
trainAttrX, valAttrX, trainImagesX, valImagesX = train_test_split(trainAttrX1, trainImagesX1, test_size = 0.25, random_state = 42)

# Creación de las variables del conjunto de entrenamiento y test (ÁngulosYaw)
train_imagesYaw = trainImagesX
train_labelsYaw = trainAttrX
val_imagesYaw = valImagesX
val_labelsYaw = valAttrX
test_imagesYaw = testImagesX
test_labelsYaw = testAttrX

# Creación de la CNN y compilación del modelo usando 'mean absolute percentage error' como pérdida (loss), lo que implica que
# se busca minimizar el porcentaje absoluto de diferencia entre nuestro ángulo (predicción) y el ángulo real.
model = modelsHeadAngle.modelo(240, 240, 1, regress = True)
opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss = lossHeadAngle.lossFunction, optimizer = opt)

# Resumen del modelo
print("")
print("SUMMARY")
print("")
summary = model.summary()
print("")

# Entrenamiento del modelo.
print("[INFO] training model...")
historyYaw = model.fit(train_imagesYaw, train_labelsYaw, epochs = 1, batch_size = 8, verbose = 1, validation_data = (val_imagesYaw, val_labelsYaw))

# Predicciones en los datos de 'test'
print("[INFO] predicting YAW angles...")
predictions = model.predict(test_imagesYaw)
print("")
print("PREDICCIONES...")
print(predictions)
print("")
print("")
print("VALORES REALES...")
print(test_labelsYaw)


# Características del modelo - Graphic Epoch-Loss
plt.plot(historyYaw.history['loss'])
plt.plot(historyYaw.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

if __name__ == '__main__':
    import sys
