import cv2
import numpy as np

# 1. Cargar imagen
image_path = './documento.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# 2. Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Binarizar imagen (utilizando Otsu)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.dilate(thresh, kernel, iterations=1)

# 4. Invertir si el texto está en negro sobre blanco
#    (queremos que el "texto" sean los valores que sumen más alto)
#inv = cv2.bitwise_not(thresh)
inv = thresh

# 5. Calcular proyección horizontal
projection = np.sum(inv, axis=1)  # Suma por filas

# 6. Buscar transiciones para identificar líneas
projection = np.sum(inv, axis=1)
max_val = np.max(projection)
mean_val = np.mean(projection)

# Por ejemplo, podrías subir el umbral a algo mayor al default
line_threshold = max(0.2 * max_val, 1.0 * mean_val)
# Ajusta según los resultados que observes

inside_line = False
start = 0
lines = []

for i, row_sum in enumerate(projection):
    if row_sum > line_threshold and not inside_line:
        # Empieza una línea
        inside_line = True
        start = i
    elif row_sum <= line_threshold and inside_line:
        # Terminó la línea
        inside_line = False
        end = i
        # Guardamos esa subimagen
        line_img = inv[start:end, :]
        lines.append(line_img)

print(lines)
# 7. Mostrar o guardar el resultado de cada línea
for idx, l_img in enumerate(lines):
    cv2.imwrite(f'line_{idx}.png', l_img)
    #O puedes mostrarlo en pantalla:
#     cv2.imshow(f'line_{idx}', l_img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


