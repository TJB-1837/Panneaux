import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os 
import keras 
import tensorflow as tf
from PySide6 import QtCore, QtWidgets, QtGui
import random
import sys

# activation de la camera
"""def getCam():
    ret, frame = camera.read()
    preprocessed_frame = preTraitement(frame)
    #print(f"pf shape : {preprocessed_frame.shape}")
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

    return [frame,classe_num,classe,proba]"""
    


class Worker(QtCore.QObject):
    result = QtCore.Signal(list)
    finished = QtCore.Signal()

    data = {}
    data_path = "labels.csv"
    with open(data_path,'r') as f:
        for ligne in f:
            d = ligne.split(',')
            data[d[0]] = str(d[1]).replace("\n","")

    model_path = "model_panneaux.keras"
    model = keras.models.load_model(model_path)

    def __init__(self):
        super().__init__()
        self.end = False
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        if not self.camera.isOpened():
            print("Erreur avec l'ouverture de la caméra")
        
    def run(self): 
        while not self.end:  
            self.getCam()
        self.finished.emit()

    def setEnd(self,bool):
        self.end = bool

    def getCam(self):
        ret, frame = self.camera.read()
        preprocessed_frame = self.preTraitement(self,frame)
        #print(f"pf shape : {preprocessed_frame.shape}")
        resized_preprocessed_frame = cv2.resize(preprocessed_frame, (32,32))
        #print(f"rpf shape : {resized_preprocessed_frame.shape}")
        tensor = tf.convert_to_tensor(resized_preprocessed_frame)
        tensor = np.expand_dims(resized_preprocessed_frame,axis=0)
        tensor = tensor.astype(np.float32)
        #print(f"tensor shape : {tensor.shape}")
        prediction = self.model.predict(tensor)
            
        classe_num = prediction.argmax()
        classe = self.data.get(str(classe_num)) 
        proba = prediction[0, classe_num]*100  

        res = [frame,classe_num,classe,proba]
        self.result.emit(res)
    
    # Méthodes de pré-traitement
    #passe l'image en noir et blanc
    @staticmethod
    def C2G(img):   
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #met toutes les valeurs des pixels de 0 à 1 pour une meilleur compréhension par la machine
    @staticmethod
    def normaliser(img):  
        return img/255.0

    #Recentre la plage de niveaux de gris
    @staticmethod 
    def egaliser(img):    
        return cv2.equalizeHist(img)

    @staticmethod 
    def preTraitement(self,img): 
        img = self.C2G(img)
        img = self.egaliser(img)
        img = self.normaliser(img)
        return img
    
    def getCam(self):
        return self.camera

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.buttonStart = QtWidgets.QPushButton("Start capture")
        self.buttonEnd = QtWidgets.QPushButton("End capture")
        self.frame = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label = QtWidgets.QLabel("Not recording",alignment=QtCore.Qt.AlignCenter)    
        self.text = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        #self.frame.setPixmap(QtGui.QPixmap("ImageAReconnaitre.jpg"))
        self.frame.setScaledContents(True)

        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.frame)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttonStart)
        self.layout.addWidget(self.buttonEnd)

        self.buttonStart.clicked.connect(self.startClicked)
        self.buttonEnd.clicked.connect(self.endClicked)

    def startWorker(self):
        self.thread = QtCore.QThread()
        self.worker = Worker() 
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(self.update_labels)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.worker.setEnd(False)
        self.worker.run()


    def startClicked(self):
        self.label.setText("Recording")
        self.startWorker()

    def endClicked(self):
        self.worker.setEnd(True)
        self.label.setText("Recording ended")

    
    def update_labels(self):
        frame, classe_num, classe_name, proba = self.worker.result
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qFrame = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.frame.setPixmap(QtGui.QPixmap(qFrame))
        if(proba>=80):
            self.text.setText(f"Classe : [{classe_num}] {classe_name}\nProbability : {proba}")
        else:
            self.text.setText(f"Classe : [None] None\nProbability : None")
   

        

if __name__=="__main__":

    



    app = QtWidgets.QApplication(sys.argv)

    widget = MyWidget()
    widget.resize(800,800)
    widget.show()

    sys.exit(app.exec())
