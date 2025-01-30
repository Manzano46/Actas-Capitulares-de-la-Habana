import numpy as np
from PIL import Image

def caminar_por_imagen(imagen, umbral=4):
    """
    Algoritmo para caminar por los píxeles negros de la imagen y agruparlos según densidad.
    
    :param imagen: Imagen binarizada (array de numpy con valores 0 y 255).
    :param umbral: Número mínimo de píxeles negros en el vecindario para considerar un área.
    :return: Lista de áreas (clusters) de píxeles negros.
    """
    imagen = np.array(imagen)

    # Dimensiones de la imagen
    filas, columnas = imagen.shape
    
    # Matriz para marcar los píxeles visitados
    visitado = np.zeros_like(imagen, dtype=bool)
    
    # Direcciones de los vecinos (Moore neighborhood)
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Función para verificar si un píxel está dentro de los límites
    def dentro_de_limites(x, y):
        return 0 <= x < filas and 0 <= y < columnas
    
    # Función para contar los píxeles negros en el vecindario
    def contar_vecindario(x, y):
        conteo = 0
        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if dentro_de_limites(nx, ny) and imagen[nx, ny] == 0:  # Píxel negro
                conteo += 1
        return conteo
    
    # Lista para almacenar los grupos/clústeres de píxeles
    clusters = []
    
    # Recorrer todos los píxeles de la imagen
    for i in range(filas):
        for j in range(columnas):
            if imagen[i, j] == 0 and not visitado[i, j]:
                # Verificar el vecindario de este píxel
                if contar_vecindario(i, j) >= umbral:
                    # Crear un nuevo clúster
                    cluster = []
                    # Búsqueda DFS para encontrar todos los píxeles conectados
                    pila = [(i, j)]
                    while pila:
                        x, y = pila.pop()
                        if not visitado[x, y]:
                            visitado[x, y] = True
                            cluster.append((x, y))
                            for dx, dy in direcciones:
                                nx, ny = x + dx, y + dy
                                if dentro_de_limites(nx, ny) and imagen[nx, ny] == 0 and not visitado[nx, ny]:
                                    pila.append((nx, ny))
                    clusters.append(cluster)
    
    return clusters


def obtener_bbox(cluster):
    """
    Obtiene el bounding box (caja delimitadora) de un clúster de píxeles.
    
    :param cluster: Una lista de tuplas (x, y) que representa los píxeles de un clúster.
    :return: Las coordenadas del bounding box (min_x, max_x, min_y, max_y).
    """
    # Encuentra las coordenadas mínimas y máximas de las coordenadas x e y en un solo paso
    min_x = min(cluster, key=lambda p: p[0])[0]
    max_x = max(cluster, key=lambda p: p[0])[0]
    min_y = min(cluster, key=lambda p: p[1])[1]
    max_y = max(cluster, key=lambda p: p[1])[1]
    
    return min_x, max_x, min_y, max_y


def cortar_cluster_con_bbox(imagen, cluster):
    """
    Recorta la imagen usando un bounding box para un clúster específico.
    
    :param imagen: Imagen original binarizada (array de numpy con valores 0 y 255).
    :param cluster: Un clúster representado como una lista de tuplas (x, y) con los píxeles del clúster.
    :return: Imagen recortada donde los píxeles fuera del clúster son blancos.
    """
    # Obtener el bounding box del clúster
    min_x, max_x, min_y, max_y = obtener_bbox(cluster)
   
    # Recortar la imagen original según el bounding box 
    imagen_recortada = imagen.crop((min_x, min_y, max_x, max_y))

    return imagen_recortada


def cortar_clusters(imagen):
    """
    Corta los píxeles de la imagen original que pertenecen a los clústeres y deja en blanco los demás.
    
    :param imagen: Imagen original binarizada (array de numpy con valores 0 y 255).
    :param clusters: Lista de clusters, cada clúster es una lista de tuplas (x, y) con los píxeles del clúster.
    :return: Imagen modificada con los píxeles de los clústeres, en blanco los que no pertenecen a ningún clúster.
    """
    image = Image.open(imagen)
    clusters = caminar_por_imagen(image, 2)
    
    # Para cada clúster, copiamos los píxeles correspondientes a la imagen original
    for i,cluster in enumerate(clusters):
        cluster_img = cortar_cluster_con_bbox(image, cluster)
        cluster_img.save(f'output_lines/line_{i}.png')


cortar_clusters('line.png')
    

