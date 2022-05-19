import os
import cv2
import random

# ? Sección de constantes
# * Ruta del folder donde se encuentran las imagenes del dataset
DATA_PATH = 'dataset/trainingSet'
IMG_QUANTITY = 50

# ? Configuración de los puntos sobre la imagen
POINT_QUANTITY = 30
IMG_SIZE = 27 # ? Se maneja solo un valor ya que son imagenes cuadradas de la forma LxL
POINT_COLOR = [7, 49, 255] # ! --> Rojo

def generateConfiguration():
  """
  * Funcion que genera la configuracion de los puntos sobre la imagen
  * @return: Lista con las configuraciones de los puntos
  """
  config = []
  for i in range(POINT_QUANTITY):
    config.append([random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)])
  return config

def readImages(folder):
  """
  * Funcion que lee las imagenes del dataset y dependiendo de la cantidad de imagenes que se desea, retorna una lista con la longitud establecida
  * @param folder: Ruta del folder donde se encuentran las imagenes del dataset
  * @return: Lista con las imagenes del dataset
  """
  images = []
  imgCount = 0
  for filename in os.listdir(f'{DATA_PATH}/{folder}'):
      if imgCount < IMG_QUANTITY:
          imgCount += 1
          images.append(f'{DATA_PATH}/{folder}/{filename}')
      else:
        break
  return images

if __name__ == '__main__':
  # * Consiguiendo la lista de folders para cada tipo de imagen, i.e cada digito
  imgFolders = os.listdir(DATA_PATH)
  
  # ? Referencia al método map --> https://realpython.com/python-map-function/

  imgMatrix = list(map(lambda folder: readImages(folder), imgFolders))

  # * Generando la configuracion de los puntos
  pointsConfig = generateConfiguration()

# -------------------------------------------------------------------------------------------------------------------------------------
# ? Test area -> Esta se puede modificar para probar la funcionalidad conforme se vaya avanzando, se eliminará al finalizar el proyecto
# ? Todo lo que este encima de esta contará como código funcional, es decir, que se procurará no modificarlo

  print(pointsConfig)

  print(len(imgMatrix))

  testImg = cv2.imread(imgMatrix[0][0])

  # * Test de como se debería apreciar la configuracion de los puntos sobre la imagen, la img se verá muy pequeña
  # * Si se quiere apreciar más detalladamente el como se "aplican" los puntos, se puede tomar una screenshot y hacer zoom

  for points in pointsConfig:
    testImg[points[0], points[1]] = POINT_COLOR

  cv2.imshow('Testing', testImg)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
