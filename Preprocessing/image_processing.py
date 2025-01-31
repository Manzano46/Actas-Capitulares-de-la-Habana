import os
import cv2
import argparse
import numpy as np

def cleanup_image(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            edges = cv2.Canny(denoised_image, 75, 150)

            kernel = np.ones((3, 3), np.uint8) 
            dilated = cv2.dilate(edges, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)

            output_filename = f"{os.path.splitext(filename)[0]}.png"
            output_path = os.path.join(output_folder, output_filename)

            cv2.imwrite(output_path, eroded)

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