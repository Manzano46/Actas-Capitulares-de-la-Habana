import os
import cv2
import argparse
import numpy as np
from PIL import Image
import kraken.binarization

def custom_grayscale(image_path, red_weight=0.1, green_weight=0.1, blue_weight=0.8):
    """
    Convierte una imagen a escala de grises usando pesos personalizados para los canales RGB.

    Args:
        image_path (str): Ruta de la imagen original.
        red_weight (float): Peso para el canal rojo.
        green_weight (float): Peso para el canal verde.
        blue_weight (float): Peso para el canal azul.

    Returns:
        numpy.ndarray: Imagen en escala de grises personalizada.
    """
    # Cargar la imagen en formato RGB
    image = cv2.imread(image_path)
    if image is None:
        print("Error al cargar la imagen.")
        return None

    # Convertir BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Separar los canales de color
    blue, green, red = cv2.split(image)

    # Aplicar la fórmula personalizada
    grayscale_custom = (blue_weight * blue +
                        green_weight * green +
                        red_weight * red).astype(np.uint8)

    return grayscale_custom

def cleanup_image(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder+'_morph'):
        os.makedirs(output_folder+'_morph')

    # Iterar sobre todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Comprobar si el archivo es una imagen (por extensión)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            # Leer la imagen en escala de grises
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error al leer la imagen: {input_path}")
                continue

            gray_image = custom_grayscale(input_path)

            equalized_image = cv2.equalizeHist(gray_image)

            denoised_image = cv2.GaussianBlur(equalized_image, (11, 11), 0)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # kernel = np.ones((3,3))
            eroded = cv2.erode(denoised_image, kernel, iterations=3)

            dilated = cv2.dilate(eroded, kernel, iterations=2)

            denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            edges = cv2.Canny(denoised_image, 75, 150)

            output_filename = f"{os.path.splitext(filename)[0]}.png"
            output_path_morph = os.path.join(output_folder+'_morph', output_filename)

            cv2.imwrite(output_path_morph, dilated)

            # Carga la imagen
            image_morph = Image.open(output_path_morph)

            # Aplica binarización
            binary_image = kraken.binarization.nlbin(image_morph)

            # Guarda la imagen binarizada
            output_path = os.path.join(output_folder, output_filename)
            binary_image.save(output_path)

            print(f"Procesada: {filename} -> {output_path}")
        else:
            print(f"Archivo no compatible: {filename}")

    print("Procesamiento completo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplica el filtro Canny a las imágenes de una carpeta.")
    parser.add_argument("input_folder", type=str, help="Ruta de la carpeta que contiene las imágenes originales.")
    parser.add_argument("output_folder", type=str, help="Ruta de la carpeta donde se guardarán las imágenes procesadas.")

    args = parser.parse_args()
    cleanup_image(args.input_folder, args.output_folder)