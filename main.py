import os
import cv2
import numpy as np
import random

# ? Sección de constantes
# * Ruta del folder donde se encuentran las imagenes del dataset
DATA_PATH = 'dataset/trainingSet'
IMG_QUANTITY = 5 # ! Valores --> 5 para testing | 50 para entrenamiento

# ? Configuración de los puntos sobre la imagen
POINT_QUANTITY = 5 # ! Valores --> 5 para testing | 30 o más para entrenamiento
IMG_SIZE = 27 # ? Se maneja solo un valor ya que son imagenes cuadradas de la forma LxL px
REGION_VALUE = 6 # ? Valor que maneja la distancia hacia dentro de la imagen considerada como región de interes
POINT_COLOR = [7, 49, 255] # ! --> Rojo ; Proposito de testing, se eliminara posteriormente

def generateConfiguration():
  """
  * Funcion que genera la configuracion de los puntos sobre la imagen
  * @return: Lista con las configuraciones de los puntos
  """
  config = []
  for i in range(POINT_QUANTITY):
    config.append([random.randint(0 + REGION_VALUE, IMG_SIZE - REGION_VALUE), random.randint(0 + REGION_VALUE, IMG_SIZE - REGION_VALUE)])
  return config

def imageBinarization(img, pointsConfig):
  """
  * Funcion que binariza la imagen
  * @param img: Imagen a binarizar
  * @param pointsConfig: Configuracion de los puntos sobre la imagen
  * @return: Imagen binarizada
  """

  # print(img)
  # ? Se inicializa la lista que representará la imagen en formato binario
  binaryIMG = np.zeros((POINT_QUANTITY, ), dtype=int)
  openedImage = cv2.imread(img)

  # ? Referencia => https://stackoverflow.com/questions/9780632/how-do-i-determine-if-a-color-is-closer-to-white-or-black#:~:text=The%20former%20is%20easy%3A%20convert,closer%20to%20white%20(255).
  for index, points in enumerate(pointsConfig):
    (B, G, R) = openedImage[points[0], points[1]]
    luminance = .2126*R + 0.7152*G + 0.0722*B
    
    if luminance >= 128:
      binaryIMG[index] = 1

  return binaryIMG

def readImages(folder, pointsConfig):
  """
  * Funcion que lee las imagenes del dataset y dependiendo de la cantidad de imagenes que se desea, retorna una lista con la longitud establecida
  * @param folder: Ruta del folder donde se encuentran las imagenes del dataset
  * @param pointsConfig: Configuracion de los puntos sobre la imagen
  * @return: Lista con las imagenes del dataset
  """
  images = []
  imgCount = 0
  for filename in os.listdir(f'{DATA_PATH}/{folder}'):
      if imgCount < IMG_QUANTITY:
          imgCount += 1
          images.append(imageBinarization(f'{DATA_PATH}/{folder}/{filename}', pointsConfig))
      else:
        break
  return images

def learningPhase():
  """
  * Funcion que trabaja sobre la memoria asociativa a partir de las imagenes del dataset
  * @param 
  * @return
  """

  pass

# ! Sección de ejecución ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

  # * Consiguiendo la lista de folders para cada tipo de imagen, i.e cada digito
  imgFolders = os.listdir(DATA_PATH)
  
  # * Generando la configuracion de los puntos
  pointsConfig = generateConfiguration()

  # ? Referencia al método map --> https://realpython.com/python-map-function/

  imgMatrix = list(map(lambda folder: readImages(folder, pointsConfig), imgFolders))


  # * Inicializando la memoria asociativa
  AM = np.zeros((len(imgMatrix) - 1, POINT_QUANTITY), dtype=int)

# ? -------------------------------------------------------------------------------------------------------------------------------------
# ? Test area -> Esta se puede modificar para probar la funcionalidad conforme se vaya avanzando, se eliminará al finalizar el proyecto
# ? Todo lo que este encima de esta contará como código funcional, es decir, que se procurará no modificarlo

  # print(pointsConfig)

  for list in imgMatrix:
    for bImg in list:
      print(bImg)
    print("--------------")

  print(AM)

  testImg = cv2.imread(f'{DATA_PATH}/0/img_1.jpg')

  binarImage = cv2.imread(f'{DATA_PATH}/0/img_1.jpg')

  # * Test de como se debería apreciar la configuracion de los puntos sobre la imagen, la img se verá muy pequeña
  # * Si se quiere apreciar más detalladamente el como se "aplican" los puntos, se puede tomar una screenshot y hacer zoom

  for points in pointsConfig:
    testImg[points[0], points[1]] = POINT_COLOR

  # ? Referencia => https://stackoverflow.com/questions/9780632/how-do-i-determine-if-a-color-is-closer-to-white-or-black#:~:text=The%20former%20is%20easy%3A%20convert,closer%20to%20white%20(255).

  for points in pointsConfig:
    (B, G, R) = binarImage[points[0], points[1]]
    luminance = .2126*R + 0.7152*G + 0.0722*B

    print("BLACK" if luminance < 128 else "WHITE")

  cv2.imshow('Testing', testImg)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
