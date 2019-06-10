# coding=utf-8

''' Definición de la función de pérdida que usará el modelo '''

# Importación de librerías necesarias
import tensorflow as tf
import numpy as np

def lossFunction (realAngle, predictionAngle):

    realAngle1 = (realAngle * np.pi) / 180
    predictionAngle1 = (predictionAngle * np.pi) / 180

    difference = tf.atan2(tf.sin(realAngle1 - predictionAngle1), tf.cos(realAngle1 - predictionAngle1))
    lossRadians = (difference * 180) / np.pi
    loss = tf.reduce_mean(tf.abs(lossRadians))

    return loss




