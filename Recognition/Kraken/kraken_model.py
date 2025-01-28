import json
import os
import subprocess
import shutil
from PIL import Image
import xml.etree.ElementTree as ET

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

    # Iterar sobre cada imagen segmentada
    for line_image in os.listdir(output_lines_dir):
        line_image_path = os.path.join(output_lines_dir, line_image)
        line_ocr_results = []  # Lista para almacenar los resultados de OCR para cada imagen

        # Ejecutar OCR con cada modelo
        for model in ocr_models:
            print(f"Ejecutando OCR con el modelo {model} sobre la imagen {line_image_path}")
            result = subprocess.run(
                ['kraken', '-i', line_image_path, line_image + '.txt', 'ocr', '-m', model],
                capture_output=True
            )
            # Añadir los resultados de OCR del modelo
            line_ocr_results.append(result.stdout.decode('utf-8'))

        # Añadir los resultados de esta imagen (con todos los modelos) a la lista global
        all_ocr_results.append({
            'image': line_image,  # Nombre de la imagen segmentada
            'ocr_results': line_ocr_results  # Resultados de OCR para cada modelo
        })

    # Guardar los resultados en un archivo
    output_file = 'ocr_results.txt'
    with open(output_file, 'w') as f:
        for image_result in all_ocr_results:
            f.write(f"Resultados para la imagen: {image_result['image']}\n")
            for idx, model_result in enumerate(image_result['ocr_results']):
                f.write(f"\tModelo {ocr_models[idx]}:\n")
                f.write(f"\t{model_result}\n")
                f.write("\n")
    

def main(input_image, output_bw_image, output_lines_dir, output_file):
    """Función principal que automatiza el proceso."""
    # Paso 1: Binarización
    #binarize_image(input_image, output_bw_image)

    # Paso 2: Segmentación (espera a que termine la binarización)
    #input_image_for_segment = Image.open(output_bw_image)
    segment_image(output_bw_image, output_lines_dir)

    # Paso 3: OCR (espera a que termine la segmentación)
    run_ocr_on_lines(output_lines_dir, ocr_models)

if __name__ == "__main__":
    input_image = 'line.jpg'  # Imagen de entrada
    output_bw_image = 'binarized.png'  # Imagen binarizada
    output_lines_dir = 'segmented_lines'  # Directorio para las líneas segmentadas
    output_file = 'output.txt'  # Archivo donde se guardará el texto OCR
    ocr_models = ['bdd-wormser-scriptorium-abbreviated-0.2', 'McCATMuS_nfd_nofix_V1']

    main(input_image, output_bw_image, output_lines_dir, output_file)
