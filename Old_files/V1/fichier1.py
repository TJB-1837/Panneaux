import cv2
import numpy as np
import matplotlib.pyplot as plt


Reco = cv2.imread("ImageAReconnaitre.jpg")
Reco_rgb = cv2.cvtColor(Reco,cv2.COLOR_BGR2RGB)
Reco_hsv = cv2.cvtColor(Reco_rgb,cv2.COLOR_RGB2HSV)



plt.figure()
plt.axis("off")
plt.subplot(2,2,1)
plt.imshow(Reco)
plt.title("Image BGR")
plt.subplot(2,2,2)
plt.imshow(Reco_rgb)
plt.title("Image RGB")
plt.subplot(2,2,3)
plt.imshow(Reco_hsv)
plt.title("Image HSV")

basRouge1 = np.array([0,100,100])
hautRouge1 = np.array([10,255,255])
basRouge2 = np.array([160,100,100])
hautRouge2 = np.array([180,255,255])

masqueR1 = cv2.inRange(Reco_hsv, basRouge1,hautRouge1)
masqueR2 = cv2.inRange(Reco_hsv, basRouge2,hautRouge2)
masqueRouge = masqueR1 + masqueR2

plt.subplot(2,2,4)
plt.imshow(masqueRouge,cmap='gray')
plt.title("Masque")
plt.show()

# application du masque à l'image rgb
image_masque = cv2.bitwise_and(Reco_rgb,Reco_rgb,mask=masqueRouge)
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.imshow(masqueRouge,cmap='gray')
plt.title("Masque")
plt.subplot(2,1,2)
plt.imshow(image_masque)
plt.title("Image masquée")
plt.show()

# Détection des contours 

def callback(input): # méthode de la video pour les Trackbars
    pass

windowname = "CannyIMG"
cv2.namedWindow(windowname)
cv2.createTrackbar('minT',windowname,0,255,callback)
cv2.createTrackbar('maxT',windowname,0,255,callback)

while True:
    if(cv2.waitKey(0)== ord('q')):
       break

    minT = cv2.getTrackbarPos('minT',windowname)
    maxT = cv2.getTrackbarPos('maxT',windowname)
    imGauss = cv2.GaussianBlur(image_masque,(5,5),0)
    img_Canny = cv2.Canny(imGauss,200 ,210)
    #img_Canny = cv2.Canny(imGauss,120 ,165)
    cv2.imshow(windowname,img_Canny)
    
cv2.destroyAllWindows()


# Détection des Formes / contours fermés

contours, hierachie = cv2.findContours(img_Canny,)
imContours = cv2.drawContours(Reco_rgb,contours)
cv2.imshow("Contours",imContours)