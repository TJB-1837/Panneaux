import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os 
import keras 
import tensorflow as tf

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



def activeCam():
    while True:
        #acquisition du flux vidéo
        ret, frame = camera.read()
        preprocessed_frame = preTraitement(frame) 
        resized_preprocessed_frame = cv2.resize(preprocessed_frame, (32,32))
        #print(f"rpf shape : {resized_preprocessed_frame.shape}")
        tensor = tf.convert_to_tensor(resized_preprocessed_frame)
        tensor = np.expand_dims(resized_preprocessed_frame,axis=0)
        tensor = tensor.astype(np.float32)
        #print(f"tensor shape : {tensor.shape}")
        prediction = model.predict(tensor)
        
        classe_num = prediction.argmax()
        classe = data.get(str(classe_num)) 
        proba = prediction[0, classe_num]*100
        if(proba>80):
            cv2.putText(frame, f"Classe : {classe_num}, {classe};", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame, f"% : {proba:.2f};", (10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame, f"Classe : None, None; % : None", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
     
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    data = {}
    data_path = "labels.csv"
    with open(data_path,'r') as f:
        for ligne in f:
            d = ligne.split(',')
            data[d[0]] = str(d[1]).replace("\n","")

    model_path = "model_panneaux.keras"
    model = keras.models.load_model(model_path)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Erreur avec l'ouverture de la caméra")

    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_heigth = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    activeCam()