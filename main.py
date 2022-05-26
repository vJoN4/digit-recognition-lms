import os
import cv2
import numpy as np
import random

# ? Sección de constantes
# * Ruta del folder donde se encuentran las imagenes del dataset
DATA_PATH = 'dataset/trainingSet'
SAMPLE_PATH = 'dataset/testSample'
IMG_QUANTITY = 50 # ! Valores --> 5 para testing | 50 para entrenamiento

# ? Configuración de los puntos sobre la imagen
POINT_QUANTITY = 30 # ! Valores --> 5 para testing | 30 o más para entrenamiento
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
  * @param img: Ruta de la imagen a binarizar
  * @param pointsConfig: Configuracion de los puntos sobre la imagen
  * @return: Imagen binarizada
  """

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

def learningPhase(imgMatrix, AM):
  """
  * Funcion que trabaja sobre la memoria asociativa a partir de las imagenes del dataset
  * @param imgMatrix: Lista con las imagenes del dataset
  * @param AM: Memoria asociativa
  * @return AM: Memoria asociativa actualizada
  """

  for index, imgList in enumerate(imgMatrix):
    for imgVector in imgList:
      AM[index] = AM[index] + (2 * np.array(imgVector) - 1)

  return AM

# ! Sección de ejecución ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

  # * Consiguiendo la lista de folders para cada tipo de imagen, i.e cada digito
  imgFolders = os.listdir(DATA_PATH)
  
  # * Generando la configuracion de los puntos
  pointsConfig = generateConfiguration()

  # ? Referencia al método map --> https://realpython.com/python-map-function/
  # * Generando la matriz de imagenes
  imgMatrix = list(map(lambda folder: readImages(folder, pointsConfig), imgFolders))


  # * Inicializando la memoria asociativa
  AM = np.zeros((len(imgMatrix), POINT_QUANTITY), dtype=int)

# ? -------------------------------------------------------------------------------------------------------------------------------------
# ? Test area -> Esta se puede modificar para probar la funcionalidad conforme se vaya avanzando, se eliminará al finalizar el proyecto
# ? Todo lo que este encima de esta contará como código funcional, es decir, que se procurará no modificarlo

  print(len(imgMatrix))

  # for list in imgMatrix:
  #   for bImg in list:
  #     print(bImg)
  #   print("--------------")

  print(AM)

  learningPhase(imgMatrix, AM)

  print(AM)

  # ! Test con memoria asociativa generada e imagen "aleatoria"
  imgTest = np.array(imageBinarization(f'{SAMPLE_PATH}/img_285.jpg', pointsConfig))

  x = np.transpose([imgTest])

  # ? Referencia: https://www.statology.org/operands-could-not-be-broadcast-together-with-shapes/
  y = np.transpose(AM.dot(x))[0]

  print("\n\n----->", np.shape(y))

  # print(y)
  print(np.where(y == np.amax(y))[0][0])
