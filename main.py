import os
import cv2
import numpy as np
import random
import uuid
import json
import jsbeautifier
import sys

# ? SECCIÓN DE CONSTANTES

# ! RUTAS
DATA_PATH = 'dataset/trainingSet' # * Ruta del folder donde se encuentran las imagenes del dataset para entrenamiento
SAMPLE_PATH = 'dataset/testSample'# * Ruta del folder para las imagenes de testing / ejecución
RESULTS_PATH = 'results' # * Ruta del folder donde se guardaran los archivos con los resultados de cada entrenamiento

# ? Configuración de los puntos sobre la imagen
am_gen_quantity = 1
img_quantity = 100 # ! Valores --> 5 para testing | 50 para entrenamiento
point_quantity = 50 # ! Valores --> 5 para testing | 30 o más para entrenamiento
IMG_SIZE = 27 # ? Se maneja solo un valor ya que son imagenes cuadradas de la forma LxL px
REGION_VALUE = 5 # ? Valor que maneja la distancia hacia dentro de la imagen considerada como región de interes
POINT_COLOR = [7, 49, 255] # ! --> Rojo ; Proposito de testing

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

def generateConfiguration():
  """
  * Funcion que genera la configuracion de los puntos sobre la imagen
  * @return: Lista con las configuraciones de los puntos
  """
  config = []
  for i in range(point_quantity):
    config.append([random.randint(0 + REGION_VALUE, IMG_SIZE - REGION_VALUE), random.randint(0 + REGION_VALUE, IMG_SIZE - REGION_VALUE)])
  return config

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

def imageBinarization(img, pointsConfig):
  """
  * Funcion que binariza la imagen
  * @param img: Ruta de la imagen a binarizar
  * @param pointsConfig: Configuracion de los puntos sobre la imagen
  * @return: Imagen binarizada
  """

  # ? Se inicializa la lista que representará la imagen en formato binario
  binaryIMG = np.zeros((point_quantity, ), dtype=int)
  openedImage = cv2.imread(img)

  # ? Referencia => https://stackoverflow.com/questions/9780632/how-do-i-determine-if-a-color-is-closer-to-white-or-black#:~:text=The%20former%20is%20easy%3A%20convert,closer%20to%20white%20(255).
  for index, points in enumerate(pointsConfig):
    (B, G, R) = openedImage[points[0], points[1]]
    luminance = .2126*R + 0.7152*G + 0.0722*B
    
    if luminance >= 128:
      binaryIMG[index] = 1

  return binaryIMG

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

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
      if imgCount < img_quantity:
          imgCount += 1
          images.append(imageBinarization(f'{DATA_PATH}/{folder}/{filename}', pointsConfig))
      else:
        break
  return images

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

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

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

def binaryzeAM(classList):
  """
  * Funcion que binariza cada 'lista' de la matriz asociativa entrenada
  * @param classList: Lista con los valores de la matriz asociativa entrenada
  * @return: tmpList: Lista con los valores binarizados de la matriz asociativa entrenada
  """

  tmpList = []
  for value in classList:
    tmpList.append(0 if value == 0 else 1)

  return np.array(tmpList)

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

def testEfficiencyRate(AM, pointsConfig):
  """
  * Funcion que guarda el dataset en un archivo JSON
  * @param AM: Memoria asociativa
  * @param pointsConfig: Configuracion de los puntos usada en el dataset
  * @return: efficiencyRate: Arreglo el porcentaje de calidad de la memoria asociativa
  """
  files = os.listdir(f'{SAMPLE_PATH}')
  correctImgsNum = 0 # * Contador para conocer cuantas imagenes se clasificaron correctamente
  TEST_IMG_QUANTITY = len(files) # * Cantidad de imagenes a clasificar
  JSON_CORRECT_IMG = open("correctNumbers.json") # * Archivo JSON que contiene los números de las imagenes correctamente clasificadas

  correctImgs = json.load(JSON_CORRECT_IMG)["sample-numbers"] 

  for index, _ in enumerate(files):
    imgTest = np.array(imageBinarization(f'{SAMPLE_PATH}/img_{index + 1}.jpg', pointsConfig))

    x = np.transpose([imgTest])

    # ? Referencia: https://www.statology.org/operands-could-not-be-broadcast-together-with-shapes/
    y = np.transpose(AM.dot(x))[0]

    estimatedNum = np.where(y == np.amax(y))[0][0]
    correctNum = correctImgs[index]

    if (estimatedNum == correctNum): 
      correctImgsNum += 1

  efficiencyRate = correctImgsNum / TEST_IMG_QUANTITY

  return efficiencyRate

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

def saveResults(id, AM, pointsConfig, quality, efficiency):
  """
  * Funcion que guarda el dataset en un archivo JSON
  * @param id: Id generado para identificar el archivo generado
  * @param AM: Memoria asociativa
  * @param pointsConfig: Configuracion de los puntos usada en el dataset
  * @param quality: Arreglo con los porcentajes de calidad por clase
  * @param efficiency: Porcentaje de eficiencia de la memoria asociativa entrenada
  * @return: None
  """

  # ? Referencia => https://www.pythontutorial.net/python-basics/python-unpack-list/
  _id, *_ = id.split('-')

  JSON = {
    'id': id,
    'pointsQuantity': point_quantity,
    'imageQuantity': img_quantity,
    'pointsConfig': pointsConfig,
    'AM': AM.tolist(),
    'quality': quality.tolist(),
    'efficiency': round((efficiency * 100), 4)
  }

  options = jsbeautifier.default_options()
  options.indent_size = 2

  if not os.path.exists(f'./{RESULTS_PATH}'):
    os.makedirs(f'./{RESULTS_PATH}')

  with open(f'{RESULTS_PATH}/AM-{_id}.json', 'w') as outfile:
    # ? Referencia => https://stackoverflow.com/questions/62434326/how-to-pretty-print-json-with-long-array-in-the-same-line
    outfile.write(jsbeautifier.beautify(json.dumps(JSON), options))

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

def main():

  # * Generando ID para el archivo de la memoria asociativa
  ID = str(uuid.uuid4())

  # * Consiguiendo la lista de folders para cada tipo de imagen, i.e cada digito
  imgFolders = os.listdir(DATA_PATH)
  
  # * Generando la configuracion de los puntos
  pointsConfig = generateConfiguration()

  # ? Referencia al método map --> https://realpython.com/python-map-function/
  # * Generando la matriz de imagenes
  imgMatrix = list(map(lambda folder: readImages(folder, pointsConfig), imgFolders))

  # * Inicializando la memoria asociativa
  AM = np.zeros((len(imgMatrix), point_quantity), dtype=int)

  # * Entrenando la memoria asociativa
  AM = learningPhase(imgMatrix, AM)

  # # * Caculando calidad de la memoria asociativa
  binaryAM = list(map(binaryzeAM, np.copy(AM)))
  quality = sum(np.transpose(binaryAM)) / point_quantity

  # * Probando porcentaje de eficiencia de la memoria asociativa
  efficiencyRate = testEfficiencyRate(AM, pointsConfig)

  # # * Guardando la memoria asociativa, la configuracion de los puntos y un ID (con proposito de distinguir entre cada entrenamiento)
  saveResults(ID, AM, pointsConfig, quality, efficiencyRate)

# ? --------------------------------------------------------------------------------------------------------------------------------------------------

# ! SECCIÓN PARA MANEJO DEL MENÚ

HELP_MENU = str(
  "\n--> LEARNING MATRIX STEINBUCH <--\n\n" +
  "\t Args:\n\n" +
  "\t\t 1 Arg --> Cantidad de memorias asociativas que se desean genenrar\n\n" +
  "\t\t 2 Arg --> Cantidad imagenes a tomar por clase (número comprendido entre 1 y 100)\n\n" +
  "\t\t 3 Arg --> Cantidad puntos a generar\n\n"
)

# ! Funciones para el manejo del menú usado en base a la cantidad de argumentos recibidos por consola

def getFirstArgCondition(args): return args[0] >= 1
def getSecondArgCondition(args): return (args[1] >= 1 and args[1] <= 100)

def cancelProcess():
  global am_gen_quantity 
  am_gen_quantity = 0

def getHelp(_):
  print(HELP_MENU)
  cancelProcess()

def genAM(args):
  if (getFirstArgCondition(args)):
    global am_gen_quantity 
    am_gen_quantity = args[0]
  else:
    print("ERROR: Argumento invalido, verifique la ayuda con --help")
    cancelProcess()

def genAM_IMG(args):
  if (getFirstArgCondition(args) and getSecondArgCondition(args)):
    global am_gen_quantity,img_quantity
    am_gen_quantity = args[0]
    img_quantity = args[1]
  else:
    print("ERROR: Argumentos invalidos, verifique la ayuda con --help")
    cancelProcess()

def genAM_IMG_POINTS(args):
  if (getFirstArgCondition(args) and getSecondArgCondition(args) and args[2] >= 1):
    global am_gen_quantity,img_quantity,point_quantity
    am_gen_quantity = args[0]
    img_quantity = args[1]
    point_quantity = args[2]
  else:
    print("ERROR: Argumentos invalidos, verifique la ayuda con --help")
    cancelProcess()

ARGS_MENU = {
  "0" : getHelp,
  "1" : genAM,
  "2" : genAM_IMG,
  "3" : genAM_IMG_POINTS
}

# ! Sección de ejecución -----------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

  args = sys.argv[1:]
  nArgs = len(args)

  if (nArgs and nArgs <=3) :
    help_condition = args[0] == '--help'
    ARGS_MENU["0" if help_condition else str(nArgs)](... if help_condition else list(map(lambda arg: int(arg), args)))
  elif nArgs:
    print("Cantidad de argumentos invalida, verifique la ayuda con --help")

  if (am_gen_quantity > 1): 
    print("ENTRENANDO MEMORIAS ASOCIATIVAS SOLICITADAS")
    for i in range(am_gen_quantity):
      main()
      print(".......")
    print("PROCESO TERMINADO")
  elif (am_gen_quantity == 1):
    print("ENTRENANDO MEMORIA ASOCIATIVA")
    main()
    print("PROCESO TERMINADO")
