import os
import logging
from symspellpy import SymSpell, Verbosity
import spacy
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar claves de entorno para Gemini
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)


def capitalize_like(original: str, corrected: str) -> str:
    """ Ajusta la capitalización de 'corrected' para que coincida con 'original'. """
    if original.isupper():
        return corrected.upper()
    elif len(original) > 0 and original[0].isupper():
        return corrected.capitalize()
    else:
        return corrected


def correct_token(token_text: str, sym_spell: SymSpell, try_segmentation: bool = True) -> str:
    """
    Corrige un token usando SymSpell. Si no encuentra nada útil,
    intenta segmentación (split) en caso de que la palabra pueda ser en realidad dos o más.
    """
    original_text = token_text
    lowered_text = token_text.lower()

    # Lookup normal
    suggestions = sym_spell.lookup(
        lowered_text,
        Verbosity.CLOSEST,
        max_edit_distance=sym_spell._max_dictionary_edit_distance,
        include_unknown=True
    )
    
    if suggestions and suggestions[0].distance != 0:
        best_match = suggestions[0].term
    else:
        if try_segmentation:
            # Lookup compound
            compound_suggestions = sym_spell.lookup_compound(
                lowered_text,
                max_edit_distance=sym_spell._max_dictionary_edit_distance
            )
            if compound_suggestions and compound_suggestions[0].distance != 0:
                best_match = compound_suggestions[0].term
            else:
                # Word segmentation
                segmentation_result = sym_spell.word_segmentation(lowered_text)
                if segmentation_result and segmentation_result.corrected_string != lowered_text:
                    best_match = segmentation_result.corrected_string
                else:
                    best_match = lowered_text
        else:
            best_match = lowered_text

    # Ajustar la capitalización
    corrected_text = capitalize_like(original_text, best_match)
    return corrected_text


def correct_text(text: str, nlp, sym_spell, try_segmentation: bool = True) -> str:
    doc = nlp(text)
    corrected_tokens = []

    for token in doc:
        if token.is_alpha:
            corrected = correct_token(token.text, sym_spell, try_segmentation)
            corrected_tokens.append(corrected)
        else:
            corrected_tokens.append(token.text)

    return " ".join(corrected_tokens)


def load_corrector_model(spacy_model: str, frequency_dictionary_path: str, max_edit_distance: int = 3, prefix_length: int = 7):
    nlp = spacy.load(spacy_model)
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    
    if not sym_spell.load_dictionary(frequency_dictionary_path, 0, 1, separator=" ", encoding="utf-8"):
        raise ValueError(f"No se pudo cargar el diccionario {frequency_dictionary_path}")

    return nlp, sym_spell


def refine_text_with_gemini(original_text: str, corrected_text: str, gemini_model_name="gemini-2.0-flash-exp"):
    """
    Envía el texto corregido a Gemini para refinamiento.
    """
    # Crear el sistema de instrucciones para Gemini
    chat = genai.GenerativeModel(
        model_name=gemini_model_name,
        system_instruction=[
            """
            Eres un asistente que ayuda a refinar textos. Analiza el siguiente texto original y corregido,
            y reescríbelo en un formato limpio, coherente y gramaticalmente correcto. Los textos son documentos históricos
            extraídos mediante OCR, por lo que pueden contener errores típicos del procesamiento OCR. Los documentos son
            cartas en Cuba de los siglos XV y XVI.
            """
        ]
    ).start_chat(history=[])

    # Crear el mensaje de entrada para Gemini
    message = (
        f"Texto original (extraido del OCR):\n{original_text}\n\n"
        f"Texto corregido:\n{corrected_text}\n\n"
        f"Por favor reescribe el texto correcto en un formato limpio y refinado, manteniendo sentido y el contexto historico."
    )

    try:
        # Enviar el mensaje al modelo Gemini
        response = chat.send_message(message)
        return response.text.strip()
    except Exception as e:
        print(f"An error occurred with Gemini: {e}")
        return corrected_text  # Devuelve el texto corregido si hay un error


if __name__ == "__main__":
    spacy_model = "es_core_news_sm"
    frequency_dictionary_path = "spanish_frequency_dictionary.txt"

    # Cargar modelo SymSpell y Spacy
    nlp, sym_spell = load_corrector_model(
        spacy_model=spacy_model,
        frequency_dictionary_path=frequency_dictionary_path,
        max_edit_distance=3,
        prefix_length=7
    )

    # Texto de entrada
    text_ocr = ("Essta es una prueva de ectraccion de teexto. "
                "Connoscida cosa sea a todos los queesta carta uieren como yo "
                "don Fferrando por la gracia de dios hey de Castiella.")

    # Corregir texto usando SymSpell y Spacy
    corrected_text = correct_text(text_ocr, nlp, sym_spell, try_segmentation=True)

    print("Texto original : ", text_ocr)
    print("Texto corregido: ", corrected_text)

    # Refinar texto con Gemini
    final_text = refine_text_with_gemini(text_ocr, corrected_text)
    print("\n=== Versión final (refinada con Gemini) ===")
    print(final_text)
