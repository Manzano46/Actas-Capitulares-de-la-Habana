import json
import os
from pathlib import Path
import subprocess
import sys
from tkinter import Image
import xml.etree.ElementTree as ET

from Recognition.SimpleHTR.SimpleHTR.src.main import run_inference


def binarize_image(input_image, output_bw_image):
    """Binariza la imagen y guarda el resultado."""
    subprocess.run(['kraken', '-i', input_image, output_bw_image, 'binarize'])

def segment_image(input_bw_image, output_lines_dir):
    """Segmenta la imagen binarizada en líneas y guarda las imágenes segmentadas."""
    # Crea el directorio donde se guardarán las imágenes segmentadas
    os.makedirs(output_lines_dir, exist_ok=True)

    # Ejecuta la segmentación
    subprocess.run(['kraken', '-i', input_bw_image, 'lines.json', 'segment'])

    # Extrae las posiciones de las líneas (esto es solo un ejemplo, deberías parsear XML adecuadamente)
    segment_image_from_json(input_bw_image,"lines.json", output_lines_dir)



def segment_image_from_json(image_path, json_file, output_lines_dir):
    """Segmenta la imagen en líneas usando un archivo JSON y guarda las imágenes segmentadas."""
    # Abrir el archivo JSON y cargar los datos
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Abrir la imagen original
    original_image = Image.open(image_path)
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_lines_dir, exist_ok=True)
    
    # Iterar sobre las líneas en el archivo JSON
    for idx, line in enumerate(data['lines']):
        # Obtener las coordenadas de la caja delimitadora (bbox)
        bbox = line['bbox']  # [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox
        
        # Recortar la imagen usando las coordenadas
        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
        
        # Guardar la imagen recortada
        cropped_image.save(f"{output_lines_dir}/line_{idx + 1}.png")
    

def run_ocr_on_lines(output_lines_dir, ocr_models):
    """Realiza OCR sobre las imágenes segmentadas con varios modelos OCR y guarda los resultados."""
    all_ocr_results = []  # Lista para almacenar los resultados de todos los modelos OCR

    # Primer comando: cambiar el modo de Kraken (activando --no-segmentation)
    subprocess.run(['kraken', 'ocr', '--no-segmentation'])

    # Iterar sobre cada imagen segmentada
    for line_image in os.listdir(output_lines_dir):

        line_ocr_results = []  # Lista para almacenar los resultados de OCR para cada imagen

        # Ejecutar OCR con cada modelo
        for model in ocr_models:
            print(f"Ejecutando OCR con el modelo {model} sobre la imagen {line_image}")
            output_txt = './Output/' + line_image[:-4] + '-' + model + '.txt'  # Nombre del archivo de salida
            subprocess.run(
                ['kraken', '-i', './segmented_lines/' + line_image, output_txt, 'ocr', '--no-segmentation', '-m', model + '.mlmodel'])

            # Leer el archivo generado por Kraken para esta imagen
            with open(output_txt, 'r', encoding='utf-8') as f:
                ocr_text = f.read()

            # Añadir los resultados de OCR del modelo
            line_ocr_results.append({
                'model': model,
                'text': ocr_text
            })

        # Añadir los resultados de esta imagen (con todos los modelos) a la lista global
        all_ocr_results.append({
            'image': line_image,  # Nombre de la imagen segmentada
            'ocr_results': line_ocr_results  # Resultados de OCR para cada modelo
        })

    #Guardar los resultados en un archivo
    output_file = 'ocr_results.txt'
    with open(output_file, 'w') as f:
        for image_result in all_ocr_results:
            f.write(f"Resultados para la imagen: {image_result['image']}\n")
            for idx, model_result in enumerate(image_result['ocr_results']):
                f.write(f"\tModelo {ocr_models[idx]}:\n")
                f.write(f"\t{model_result}\n")
                f.write("\n")
    

import os
import subprocess

def run_ocr_on_image(image, output_dir, ocr_models):
    """
    Realiza OCR sobre una imagen y guarda los resultados en archivos de texto.
    
    Parameters:
    - image (str): Ruta de la imagen original.
    - segmented_lines (str): Directorio que contiene las imágenes segmentadas (líneas).
    - output_dir (str): Directorio donde se guardarán los resultados de OCR.
    - ocr_models (list): Lista de modelos de OCR a utilizar.
    """
    # Asegurarse de que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    # Ejecutar OCR con cada modelo
    for model in ocr_models:
        output_txt = output_dir + image[:-4] + '-' + model + '.txt'  # Nombre del archivo de salida
        print(f"Ejecutando OCR con el modelo {model} sobre la imagen {image}")
        
        # Ejecutar el comando de OCR
        subprocess.run(['kraken', '-i', image, output_txt, 'segment', 'ocr', '-m', model + '.mlmodel'])

    print("OCR completado para todas las imágenes segmentadas.")

def get_transcription(input_image):
    """Función principal que automatiza el proceso."""

    #output_bw_image = input_image[:-4] + '.png'  # Imagen binarizada
    output_bw_image = input_image
    output_lines_dir = 'segmented_lines'  # Directorio para las líneas segmentadas
    output_dir = './Output/'  # Archivo donde se guardará el texto OCR
    ocr_models = ['bdd-wormser-scriptorium-abbreviated-0.2', 'McCATMuS_nfd_nofix_V1']

    if os.path.exists(output_lines_dir):
        # Eliminar todos los archivos dentro de la carpeta de lineas
        for archivo in os.listdir(output_lines_dir):
            ruta_archivo = os.path.join(output_lines_dir, archivo)
            try:
                # Si es un archivo, lo eliminamos
                if os.path.isfile(ruta_archivo):
                    os.remove(ruta_archivo)
            except Exception as e:
                print(f"Error al eliminar {ruta_archivo}: {e}")

    # Paso 1: Binarización
    #binarize_image(input_image, output_bw_image)

    segment_image(output_bw_image, output_lines_dir)

    simpleHTR_lines = []

    for line in os.listdir(output_lines_dir):
        simpleHTR_lines.append(run_inference('Recognition/Kraken/segmented_lines/' + line)[0])

    # Paso 2: Segmentacion y OCR 
    run_ocr_on_image(output_bw_image, output_dir, ocr_models)

    ocr_results = []

    ocr_results.append(simpleHTR_lines)

    for result in os.listdir(output_dir):
        # Leer el archivo generado por Kraken para esta imagen
        with open(output_dir + result, 'r', encoding='utf-8') as f:
            ocr_text = f.read()

        # Añadir los resultados de OCR del modelo
        ocr_results.append({
            'image-model': result,
            'text': ocr_text
        })

    return ocr_results


def binarize_and_segment(input_img):

    output_bw_image = input_img[:-4] + '.png'  # Imagen binarizada
    output_lines_dir = 'segmented_lines'  # Directorio para las líneas segmentadas

    # Paso 1: Binarización
    binarize_image(input_img, output_bw_image)

     # Paso 2: Segmentacion
    segment_image(output_bw_image, output_lines_dir)


get_transcription('../../data/output_preprocessing/binarized.png')