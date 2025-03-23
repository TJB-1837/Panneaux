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
img = cv2.imread("myData/0/0_9960_1577671998.6182477.png")
image_size = img.shape

# Sélection de classes à visualiser
batch_size = 32
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

dataGen = ImageDataGenerator(  #génère des variations des images instantannément
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8,1.2])

# récupération des données
X_train = dataGen.flow_from_directory(
    directory=strPath,
    target_size=image_size,
    batch_size=batch_size )

Y_train = 


# Création du modèle 

model = Sequential()    #création du model dit Sequential où l'on décrit chaque couche 
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1))) #couche de convolution 2d * 32, de taille 3,3 qui prend en entrée des images 32,32,1 (en niveaux de gris)
model.add(keras.layers.MaxPooling2D(2,2)) #couche de "reduction" de la taille
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten()) #couche de "reduction" de la dimension (2d -> 1d)
model.add(keras.layers.Dense(500,activation='relu')) #couches denses qui relient tous les neurones ensembles 
model.add(keras.layers.Dense(43,activation='softmax')) #couche de sortie
model.summary() #résume le modèle 

#compile le modèle 
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) 

# Entrainement du modèle 
model.fit(
    x=
    y=
)


