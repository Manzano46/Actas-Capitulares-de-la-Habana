import json
import os
import matplotlib.pyplot as plt
import Levenshtein

# Definir rutas de archivos y carpetas
documents_folder = "scrapper/Filtered_Documents"

def load_json(file_path):
    """Carga un archivo JSON y lo devuelve como un diccionario o lista."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_cer(reference, hypothesis):
    """Calcula el Character Error Rate (CER) usando la distancia de Levenshtein."""
    return Levenshtein.distance(reference, hypothesis) / max(1, len(reference))

def compute_wer(reference, hypothesis):
    """Calcula el Word Error Rate (WER) usando la distancia de Levenshtein."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return Levenshtein.distance(" ".join(ref_words), " ".join(hyp_words)) / max(1, len(ref_words))

def get_reference_text(image_name):
    """Obtiene el texto crítico correspondiente a una imagen a partir de los JSON en filtered_documents."""
    doc_id = image_name.split("_")[0]  # Extraer el ID del documento
    doc_path = os.path.join(documents_folder, f"{doc_id}.json")
    
    if not os.path.exists(doc_path):
        print(f"Advertencia: No se encontró el JSON para {image_name}")
        return None
    
    document_data = load_json(doc_path)
    
    for page in document_data:
        if page["image_url"].endswith(image_name):
            return page["critical_text"]
    
    print(f"Advertencia: No se encontró el texto de referencia para {image_name}")
    return None