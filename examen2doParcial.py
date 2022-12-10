import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

cord1x =0
cord1y =0
cord2x =0
cord2y =0

def assignValueC1(x,y):
    global cord1y
    global cord1x
    cord1x = x
    cord1y = y
    #print(x,y)

def assignValueC2(x,y):
		global cord2y
		global cord2x
		cord2x = x
		cord2y = y
		#print(x,y)


def distancia(cord1x,cord1y,cord2x,cord2y):
			global distanceValue 
			distanceValue= math.sqrt( ((cord1x-cord2x)**2)+((cord1y-cord2y)**2) )
			print("La distancia de dos puntos ", "Punto 1 = ", cord1x,",", cord1y,"Punto 2 = ",cord2x,",", cord2y,"es de", distanceValue)
            

def segmentacion ():
				img = cv2.imread(r'jit.jpg')
				b,g,r = cv2.split(img)
				rgb_img = cv2.merge([r,g,b])
				gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

				kernel = np.ones((2,2),np.uint8)

				closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

				sure_bg = cv2.dilate(closing,kernel,iterations=3)

				dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

				ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

				sure_fg = np.uint8(sure_fg)
				unknown = cv2.subtract(sure_bg,sure_fg)

				ret, markers = cv2.connectedComponents(sure_fg)
				markers = markers+1

				markers[unknown==255] = 0
				markers = cv2.watershed(img,markers)
				img[markers == -1] = [255,0,0]
				plt.subplot(211),plt.imshow(rgb_img)
				plt.title('Imagen original'), plt.xticks([]), plt.yticks([])
				plt.subplot(212),plt.imshow(thresh, 'gray')
				plt.imsave(r'thresh.png',thresh)
				plt.title("Segmentacion"), plt.xticks([]), plt.yticks([])
				plt.tight_layout()
				plt.show()




#Old
imagen = cv2.imread("jit.jpg")
size=np.asarray(imagen)
alto=size.shape[0]
ancho=size.shape[1]

#New
windowName = 'ImagenPuntos'
img = np.zeros((ancho,alto,3),np.uint8)
cv2.namedWindow(windowName)

def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        assignValueC1(x,y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        assignValueC2(x,y)

cv2.setMouseCallback(windowName, CallBackFunc)	


while (True):
    cv2.imshow(windowName, imagen)
    if cv2.waitKey(20) == 27:
          break	
cv2.destroyAllWindows()
distancia(cord1x,cord1y,cord2x,cord2y)
segmentacion()
