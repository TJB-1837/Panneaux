import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

# défition des chemins 
strLabels = "labels.csv"
strPath = "myData"


labels =  pd.read_csv("labels.csv")

# Sélection de classes à visualiser
numClasses = 43

def printDataSet(nClass):
    fig, axs = plt.subplots(nrows=nClass, ncols=5, figsize=(10,10))
    for i in range(nClass):
        classePath = os.path.join(strPath, str(i))
        images = os.listdir(classePath)
        for j in range(5):
            img = cv2.imread(os.path.join(classePath, images[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].axis("off")
            if j == 2:
                axs[i, j].set_title(f"Classe{i}")
    plt.show()

def printDistribution(nClass):
    plt.figure()
    for i in range(nClass):
        classePath = os.path.join(strPath, str(i))
        images = os.listdir(classePath)
        plt.bar(classePath, len(images), align='center', alpha=0.5, color="blue")
    plt.show()

# Méthodes de pré-traitement
def C2G(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normaliser(img):
    return img/255.0

def egaliser(img):
    return cv2.equalizeHist(img)

def preTraitement(img):
    img = C2G(img)
    img = egaliser(img)
    img = normaliser(img)
    return img

ImageDataGenerator()

img = cv2.imread("myData/0/0_9960_1577671998.6182477.png")
img_pp =  preTraitement(img)

dataGen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8,1.2])

model = Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500,activation='relu'))
model.add(keras.layers.Dense(43,activation='softmax'))

model.summary()

