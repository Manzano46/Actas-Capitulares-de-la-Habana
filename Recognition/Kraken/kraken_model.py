import os
from PIL import Image
from kraken import binarization, pageseg, rpred
from kraken.lib import models

def ocr_handwritten_image(image_path, seg_model_path, rec_model_path):
    """
    Realiza:
      1. Binarización de la imagen (nlbin).
      2. Segmentación en líneas con un modelo distinto (pageseg.segment + model=).
      3. Reconocimiento línea a línea con rpred.rpred y otro modelo distinto.
    Retorna el texto reconocido.
    """

    # 1) Verificar rutas
    if not os.path.isfile(seg_model_path):
        raise FileNotFoundError(f"No se encontró el modelo de segmentación: {seg_model_path}")
    if not os.path.isfile(rec_model_path):
        raise FileNotFoundError(f"No se encontró el modelo de reconocimiento: {rec_model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    # 2) Cargar los modelos
    print("[INFO] Cargando modelo de segmentación...")
    seg_nn = models.load_any(seg_model_path)

    print("[INFO] Cargando modelo de reconocimiento...")
    rec_nn = models.load_any(rec_model_path)

    # 3) Binarizar la imagen
    pil_img = Image.open(image_path).convert('L')
    bin_img = binarization.nlbin(pil_img)
    # nlbin aplica Otsu + deskew + limpieza interna que Kraken usa por defecto

    # 4) Segmentar en líneas usando el modelo de segmentación
    seg_result = pageseg.segment(
        im=bin_img,
        text_direction='horizontal-lr',
        scale=1.0,
        maxcolseps=0,  
        model=seg_nn    # ← Asignamos específicamente el modelo de segmentación
    )

    # Verificar si hay cajas detectadas
    if not hasattr(seg_result, 'boxes') or not seg_result.boxes:
        print("[WARNING] No se encontraron líneas. Revisa la calidad de la imagen y el modelo.")
        return ""

    print(f"Segmentación completa: {len(seg_result.boxes)} líneas encontradas.")

    # 5) Recortar las líneas detectadas
    lines_crops = []
    for box in seg_result['boxes']:
        line_crop = bin_img.crop(box)
        lines_crops.append(line_crop)

    # 6) Realizar el reconocimiento con el modelo de reconocimiento
    #    rpred.rpred admite una lista de PIL Images
    print("[INFO] Iniciando reconocimiento de texto...")
    preds = rpred.rpred(
        network=rec_nn,
        im=lines_crops,
        tags=[],
        bidi_reordering=False,  # Ajustar si idioma es RTL
        apply_box_threshold=False
    )

    # 7) Extraer el texto
    recognized_lines = []
    for pred in preds:
        # Cada pred puede tener varias salidas; típicamente la primera (pred.outputs[0]) es el texto
        recognized_lines.append(pred.outputs[0].strip())

    recognized_text = "\n".join(recognized_lines)
    return recognized_text

if __name__ == "__main__":
    # Rutas de ejemplo
    IMAGE_PATH = "image.jpg"
    SEG_MODEL_PATH = "sinai_sam_rec_v4_best.mlmodel"  # Modelo distinto para segmentar
    REC_MODEL_PATH = "McCATMuS_nfd_nofix_V1.mlmodel"  # Modelo para el reconocimiento final

    try:
        # Invocar la función de OCR
        resultado = ocr_handwritten_image(IMAGE_PATH, SEG_MODEL_PATH, REC_MODEL_PATH)
        print("Texto reconocido:\n")
        print(resultado)
    except Exception as e:
        print("Ocurrió un error:", e)
