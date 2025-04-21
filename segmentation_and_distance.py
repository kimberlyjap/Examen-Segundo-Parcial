import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Variables globales para las coordenadas
cord1x, cord1y = 0, 0
cord2x, cord2y = 0, 0

# Función para asignar el valor al primer punto
def assignValueC1(x, y):
    global cord1x, cord1y
    cord1x, cord1y = x, y
    print(f"Punto 1 seleccionado en: ({cord1x}, {cord1y})")

# Función para asignar el valor al segundo punto
def assignValueC2(x, y):
    global cord2x, cord2y
    cord2x, cord2y = x, y
    print(f"Punto 2 seleccionado en: ({cord2x}, {cord2y})")

# Función para calcular la distancia entre dos puntos
def distancia(cord1x, cord1y, cord2x, cord2y):
    if cord1x == 0 and cord1y == 0 or cord2x == 0 and cord2y == 0:
        print("¡Error! Ambos puntos deben ser seleccionados antes de calcular la distancia.")
        return
    distanceValue = math.sqrt(((cord1x - cord2x) ** 2) + ((cord1y - cord2y) ** 2))
    print(f"La distancia entre Punto 1({cord1x}, {cord1y}) y Punto 2({cord2x}, {cord2y}) es de: {distanceValue:.2f} unidades.")
    return distanceValue

# Función de segmentación con visualización
def segmentacion():
    img = cv2.imread('jit.jpg')  # Ruta de la imagen
    if img is None:
        print("¡Error! No se pudo cargar la imagen.")
        return
    
    # Convertir a RGB y gris
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Umbralización
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Operaciones morfológicas
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Convertir en imágenes de tipo uint8
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Componentes conectados
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1

    # Eliminar las áreas desconocidas
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # Visualización con matplotlib
    plt.subplot(211), plt.imshow(rgb_img)
    plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(thresh, 'gray')
    plt.imsave('thresh.png', thresh)  # Guardar la imagen binaria
    plt.title('Segmentación'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Configuración inicial de la ventana
imagen = cv2.imread("jit.jpg")
if imagen is None:
    print("¡Error! No se pudo cargar la imagen.")
    exit()

alto, ancho, _ = imagen.shape
windowName = 'ImagenPuntos'
img = np.zeros((alto, ancho, 3), np.uint8)
cv2.namedWindow(windowName)

# Función de callback para el mouse
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        assignValueC1(x, y)
        cv2.circle(imagen, (x, y), 5, (0, 0, 255), -1)  # Dibujar un círculo rojo
    elif event == cv2.EVENT_RBUTTONDOWN:
        assignValueC2(x, y)
        cv2.circle(imagen, (x, y), 5, (255, 0, 0), -1)  # Dibujar un círculo azul

# Establecer el callback para la ventana
cv2.setMouseCallback(windowName, CallBackFunc)

# Mostrar la imagen y esperar interacción
while True:
    cv2.imshow(windowName, imagen)
    key = cv2.waitKey(20)
    if key == 27:  # Salir si presiona "Esc"
        break

# Calcular distancia y hacer segmentación
distancia(cord1x, cord1y, cord2x, cord2y)
segmentacion()

cv2.destroyAllWindows()

