import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os 
import keras 
import tensorflow as tf
import time
import win32gui
import win32ui
import win32con
import win32api
from PIL import ImageGrab
import ctypes

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

def preTraitement_uint8(img): 
    img = C2G(img)
    img = egaliser(img)
    img = img.astype(np.uint8) 
    return img



seuil_bas = 100
seuil_haut = 200

# activation de la camera
def activeCam(data):
    interpreter = tf.lite.Interpreter(model_path="model_panneaux.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    

    while True:
        #acquisition du flux vidéo
        cap = activeCap("Euro Truck Simulator 2")
        if not cap[0]:
            print("impossible de trouver la fenêtre")
            break
        frame = cap[1]
        #cv2.imshow("window",frame)

        
        #Détection des zones d'intérêt
        preprocessed_frame_uint8 = preTraitement_uint8(frame)
        #print(f"pf shape : {preprocessed_frame.shape}")
        edges = cv2.Canny(preprocessed_frame_uint8, seuil_bas, seuil_haut) 
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (0,0,255), 2)  


        for cnt in contours:  
            x,y,w,h = cv2.boundingRect(cnt)   # récupération de toutes les positions des contours
            prediction_done = False
            if 50 < w < 300 and 50 < h < 300: # élimination des contours trop petits ou trop grands
                ri = frame[y:y+h,x:x+w]         #ri = région d'intérêt
                resize_ri = cv2.resize(ri, (32,32))
                pt_ri = preTraitement(resize_ri)
                pt_ri = np.expand_dims(pt_ri, axis=-1)
                tensor_ri = tf.convert_to_tensor(pt_ri)
                tensor_ri = np.expand_dims(pt_ri,axis=0)
                tensor_ri = tensor_ri.astype(np.float32)


                interpreter.set_tensor(input_details[0]['index'], tensor_ri)
                start_time = time.time()
                interpreter.invoke()
                end_time = time.time()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                
                inference_time_ms = (end_time - start_time) * 1000  
                prediction_done = True

            #affichage des prédictions
            if prediction_done:
                classe_num = np.argmax(prediction)
                classe = data.get(str(classe_num)) 
                proba = prediction[0, classe_num]*100
                if(proba>80):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{classe} ({proba:.1f}%)", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.putText(frame, f"{classe} ({proba:.1f}%)", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.putText(frame, f"{inference_time_ms:.0f}ms", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
  
        cv2.imshow("Window_ML", frame)

        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

def activeCap(window_name):
    window = win32gui.FindWindow(None, window_name)
    window_found = True
    if window == 0:
        print(f"Fenêtre {window_name} introuvable")
        window_found = False
        return [window_found, None]
    
    

    gauche, haut, droite, bas = win32gui.GetWindowRect(window) # Postition des points avec le signe +
    w =  droite - gauche                                       # haut  +-----------/
    h = bas - haut                                             #       |           |
                                                               #       |           |
                                                               #       /-----------+ bas
                                                               # gauche            droite
    
    #accès au contexte graphique de la fenêtre
    windowDC = win32gui.GetWindowDC(window)
    mfcDC = win32ui.CreateDCFromHandle(windowDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC,w,h)
    saveDC.SelectObject(saveBitMap)

    result = ctypes.windll.user32.PrintWindow(window, saveDC.GetSafeHdc(), 2)  # 2 = PW_RENDERFULLCONTENT
    
    if result:
        # Convertir en array numpy
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        frame = np.frombuffer(bmpstr, dtype='uint8')
        frame = frame.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
        frame = frame[:,:,:3]  # Enlever le canal alpha
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        print("PrintWindow a échoué")
        frame = None

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(window, windowDC)

    return [window_found, frame]




if __name__=="__main__":
    data = {}
    data_path = "labels.csv"
    with open(data_path,'r') as f:
        for ligne in f:
            d = ligne.split(',')
            data[d[0]] = str(d[1]).replace("\n","")

    model_path = "model_panneaux.keras"
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    
    with open("model_panneaux.tflite", "wb") as f:
        f.write(tflite_model)

    activeCam(data)