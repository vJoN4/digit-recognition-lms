import numpy as np
import os
import sys
import json
from main import SAMPLE_PATH, RESULTS_PATH, imageBinarization

def testMultipleAM(AMList, pointsList, nAm):
  """
  * Funcion que prueba varias memorias asociativas a la vez
  * @param AMList: Lista de memorias asociativas
  * @param pointsList: Lista de la configuracion de los puntos
  * @param nAM: Cantidad de memorias a probar entre si
  * @return: efficiencyRate: Porcentaje de efectividad del entrenamiento
  """
  files = os.listdir(f'{SAMPLE_PATH}')
  correctImgsNum = 0 # * Contador para conocer cuantas imagenes se clasificaron correctamente
  TEST_IMG_QUANTITY = len(files) # * Cantidad de imagenes a clasificar
  JSON_CORRECT_IMGS = json.load(open("correctNumbers.json"))["sample-numbers"]

  aResults = []

  # * Se recorren cada imagen y se prueba con cada memoria asociativa
  # * Una vez se probo cada imagen y cada memoria dió su resultado
  # * Se extrae solo el valor en el que más memorias asociativas coincidieron
  for index, _ in enumerate(files):
    innerResults = []

    # ? Referencia: https://www.geeksforgeeks.org/python-list-slicing/
    for AM_index, AM in enumerate(AMList[:nAm]):
      imgTest = np.array(imageBinarization(f'{SAMPLE_PATH}/img_{index + 1}.jpg', pointsList[AM_index]))
      
      x = np.transpose([imgTest])

      y = np.transpose(AM.dot(x))[0]

      innerResults.append(np.where(y == np.amax(y))[0][0])

    # ? Referencia: https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/#:~:text=Make%20use%20of%20Python%20Counter,by%20using%20most_common()%20method.
    aResults.append(max(set(innerResults), key = innerResults.count))

  # * Comparando resultados finales con los correspondientes a la imagen
  for index, estimatedNum in enumerate(aResults):
    if (estimatedNum == JSON_CORRECT_IMGS[index]):
      correctImgsNum += 1

  efficiencyRate = correctImgsNum / TEST_IMG_QUANTITY

  return efficiencyRate

# ? ----------------------------------------------------------------------------------------------------------------------

def main(nAM = 0):# * Se verifica que el directorio de resultados exista
  if os.path.exists(f'./{RESULTS_PATH}'):

    # * Se verifica que el directorio de resultados no este vacio
    aFiles = os.listdir(f'./{RESULTS_PATH}')
    aPointsConfig = []
    aMemories = []
    if len(aFiles):
      for AM_FILE in aFiles[:nAM] if nAM != 0 else aFiles:

        # * Se abre el archivo JSON
        AM_JSON = json.load(open(f"{RESULTS_PATH}/{AM_FILE}"))

        # * Se extraen los valores de interes del archivo JSON y se guardan en listas
        aPointsConfig.append(AM_JSON['pointsConfig'])
        aMemories.append(np.array(AM_JSON['AM']))

      effectiveness = testMultipleAM(aMemories, aPointsConfig, nAM if nAM != 0 else len(aFiles))
      print(f"LA EFECTIVIDAD DE {len(aFiles)} MEMORIAS ASOCIATIVAS ES: {round((effectiveness * 100), 4)}")
    else:
      print("No hay memorias asociativas entrenadas para consultar")
  else:
    print(f"No existe el directorio {RESULTS_PATH}, por lo tanto no hay memorias asociativas para consultar")

# ! Sección de ejecución -----------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

  # * Se espera que se ingrese como parametro la cantidad de memorias asociativas a probar entre si
  args = sys.argv[1:]
  nArgs = len(args)

  if (nArgs):
    nAM = int(args[0])
    if (nAM >= 1):
      main(nAM)
    else:
      print("El parametro debe ser un número positivo mayor o igual que 1")
  else:
    main()