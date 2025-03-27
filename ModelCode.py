import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os 
import tensorflow
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import random
import pickle

# paramètres
strLabels = "labels.csv"
strPath = "myData"
labels =  pd.read_csv("labels.csv")
classe_list = os.listdir(strPath)
numClasses = len(classe_list)
batch_size_val = 50
epochs_val = 15
steps_per_epoch_val= 1000      #(len(X_train) // batch_size)
img_dim = (32,32,3)
test_size_val=0.2
validation_size_val=0.2


# Méthodes de pré-traitement
#passe l'image en noir et blanc
def C2G(img):   
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#met toutes les valeurs des pixels de 0 à 1 pour une meilleur compréhension par la machine
def normaliser(img):  
    return img/255.0

#Recentre la plage de niveaux de gris 
def egaliser(img):    
    return cv2.equalizeHist(img)

def preTraitement(img): 
    img = C2G(img)
    img = egaliser(img)
    img = normaliser(img)
    return img

#chargement des ensembles de données pour l'entrainement

x = []
y = []
print("Classes détectées : ", numClasses)
print("Importation des classes : ")
for i in range(numClasses):
    print(f"{i}", end=" ")
    classePath = os.path.join(strPath, str(i))
    images = os.listdir(classePath)
    for j in range(len(images)):
        if cv2.imread(os.path.join(classePath, images[j])) is None:
            print(f"Erreur pour l'image {j} de la classe {i} !")
        else:
            img = cv2.imread(os.path.join(classePath, images[j]))
            x.append( img )
            y.append( i )
    

X = np.array(x)
#X = X.reshape(X.shape[0],32,32,1)
Y = np.array(y)
Y = to_categorical(Y,num_classes=numClasses)
print(f"\nX shape: {X.shape}, Y shape: {Y.shape}")

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size_val)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train,test_size=validation_size_val)

print("Valeurs min/max avant prétraitement :")
print("Min :", X_train.min(), "Max :", X_train.max())

#Pretraitement de toutes les images 
X_train=np.array(list(map(preTraitement,X_train)))  
X_validation=np.array(list(map(preTraitement,X_validation)))
X_test=np.array(list(map(preTraitement,X_test)))

print("Valeurs min/max après prétraitement :")
print("Min :", X_train.min(), "Max :", X_train.max())

#affichage de 10 images de X
plt.figure("Images de X_train après prétraitement")
for i in range(1,10):
    plt.subplot(1,10,i)
    plt.imshow(X_train[i])
plt.show()
print(f"X_tr shape : {X_train.shape}")
print(f"X_t shape : {X_test.shape}")
print(f"X_v shape : {X_validation.shape}")

# Dimension des sets d'entrainement incorrect -> reshape (taille du set, dimension x de l'image, dimension y, 1 -> car niveau de gris)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
# affichage des shapes de chaque ensemble et du one hot pour Y 
print(f"X_tr shape : {X_train.shape}, Y_tr shape: {Y_train.shape}")
print(f"X_t shape : {X_test.shape}, Y_t shape : {Y_test.shape}")
print(f"X_v shape : {X_validation.shape}, Y_v shape : {Y_validation.shape}")



# Sélection de classes à visualiser
def printDataSet(nClass):
    fig, axs = plt.subplots(nrows=nClass, ncols=5, figsize=(10,10))
    for i in range(nClass):
        classePath = os.path.join(strPath, str(i))
        images = os.listdir(classePath)
        for j in range(5):
            img = cv2.imread(os.path.join(classePath, images[j]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].axis("off")
            if j == 2:
                axs[i, j].set_title(f"Classe{i}")
    plt.show()


def printDistribution(nClass,path):
    plt.figure()
    for i in range(nClass):
        classePath = os.path.join(path, str(i))
        images = os.listdir(classePath)
        plt.bar(i, len(images), align='center', alpha=0.5, color="blue")
    plt.title("Répartition des images")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.show()

printDistribution(numClasses,strPath)

#img = cv2.imread("myData/0/0_9960_1577671998.6182477.png")
#img_pp =  preTraitement(img)

# Ajout de poids pour chaque classe car le dataset est mal réparti
classes_y = np.unique(Y_train.argmax(axis=1)) # récupération des classes existantes
class_weights = compute_class_weight(class_weight="balanced",classes=classes_y, y=Y_train.argmax(axis=1))
class_weight_dict = {i: class_weights[i] for i in range( len(classes_y))}
print(class_weight_dict)

#génère en temps réel des variations des images pour une meilleur reconnaissance
dataGen = ImageDataGenerator(
    rotation_range=10, #rotation de +/- 15 degrés
    width_shift_range=0.1, #décalage longeur de 10%
    height_shift_range=0.1, #décalage largeur de 10%
    zoom_range=0.2, #Zoom de 20%
    shear_range=0.1#, #inclination de 10 degrés
    #brightness_range=[0.9,1.1] #variation de la luminosité de 90 à 110 % -------------------> Bug
    ) 

dataGen.fit(X_train)
batches= dataGen.flow(X_train,Y_train,batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch,y_batch = next(batches)
 
# TO SHOW AGMENTED IMAGE SAMPLES
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()

plt.title("Images générées par le dataGen")
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(img_dim[0],img_dim[1]),cmap="gray")
    axs[i].axis('off')
plt.show()

# Création du modèle couches après couches 
def myModel():

    # Hyper paramètres
    nb_filtres = 64           # Nb de filtres de convolutions 
    filtre_size1 = (5,5)      # Taille du filtre de kernel pour les premières convolutions -détect
    filtre_size2 = (3,3)      # Taille du filtre de kernel pour les secondes convolutions 
    nb_neurones = 500         # Nb de neurones dans la couche discrète
    pc_neurones_disable = 0.5 # Pourcentage de neurones désactivés à chaque "étapes" de l'apprentissage -> le modèle ne se repose pas trop sur qq neurones uniquement
    pool_size = (2,2)         # Taille de l'image à réduire 

    #création du model dit Sequential où l'on décrit chaque couche 
    model = Sequential()    
    model.add(keras.layers.Conv2D(nb_filtres, filtre_size1, input_shape=(img_dim[0],img_dim[1],1), activation='relu')) #couche de convolution 2d * 32, de taille 3,3 qui prend en entrée des images 32,32,1 (en niveaux de gris)
    model.add(keras.layers.Conv2D(nb_filtres, filtre_size1, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size)) #couche de "reduction" de la taille

    model.add(keras.layers.Conv2D(nb_filtres//2, filtre_size2, activation='relu'))
    model.add(keras.layers.Conv2D(nb_filtres//2, filtre_size2, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size))
    model.add(keras.layers.Dropout( pc_neurones_disable ))

    model.add(keras.layers.Flatten()) #couche de "reduction" de la dimension (2d -> 1d)
    model.add(keras.layers.Dense(nb_neurones,activation='relu')) #couches denses qui relient tous les neurones ensembles 
    model.add(keras.layers.Dropout( pc_neurones_disable ))  #Désactive certain neurones pour que le modèle ne se repose pas sur quelques N uniquement
    model.add(keras.layers.Dense(43,activation='softmax')) #couche de sortie
    print(model.summary()) #résume le modèle 

    #compile le modèle 
    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"]) 
    return model

model = myModel()

# Entrainement du modèle 
def model_fit(model):    # Retourne un historique de statistiques du model entrainé, le model en paramètre devient automatiquement le modèle entrainé
    trained_model = model.fit( 
        dataGen.flow(X_train,Y_train,batch_size=batch_size_val),
        steps_per_epoch = steps_per_epoch_val,          #nb d'image à traiter
        epochs = epochs_val,                            # nb de parcours de l'ensemble 
        validation_data = (X_validation,Y_validation),  #données de validation 
        class_weight = class_weight_dict,               #poid de chaque classe
        shuffle=1
    )    
    return trained_model

model_history = model_fit(model)

# Affiche le résultat de l'entrainement
plt.figure()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

#Evaluation du model
eval = model.evaluate(x=X_test, y=Y_test,batch_size=batch_size_val)
print(f"Loss = {eval[0]}, accuracy = {eval[1]}")

# Prédictions
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Convertir les probabilités en classes
y_true = Y_test.argmax(axis=1)  # Les vraies classes

# Rapport de classification
print(classification_report(y_true, y_pred_classes))

# exportation du model 
model.save("model_panneaux.keras")
