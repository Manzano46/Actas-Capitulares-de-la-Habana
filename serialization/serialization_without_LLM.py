import spacy
from symspellpy import SymSpell, Verbosity
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer


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
    # 1. Guardar info sobre capitalización
    original_text = token_text
    lowered_text = token_text.lower()
    
    # 2. Primer intento: lookup normal (single-word)
    suggestions = sym_spell.lookup(
        lowered_text,
        Verbosity.CLOSEST,
        max_edit_distance=sym_spell._max_dictionary_edit_distance,
        include_unknown=True
    )
    
    if suggestions and suggestions[0].distance != 0:
        # Hubo una sugerencia de corrección distinta
        best_match = suggestions[0].term
    else:
        # 3. Si no hubo sugerencia o la sugerencia es la misma palabra,
        #    intentar lookup_compound o word_segmentation
        if try_segmentation:
            compound_suggestions = sym_spell.lookup_compound(
                lowered_text,
                max_edit_distance=sym_spell._max_dictionary_edit_distance
            )
            if compound_suggestions and compound_suggestions[0].distance != 0:
                # Usa la mejor sugerencia compuesta
                best_match = compound_suggestions[0].term
            else:
                # 4. También se puede intentar sym_spell.word_segmentation
                segmentation_result = sym_spell.word_segmentation(lowered_text)
                if segmentation_result and segmentation_result.corrected_string != lowered_text:
                    best_match = segmentation_result.corrected_string
                else:
                    # Dejar la palabra tal cual si no hay nada mejor
                    best_match = lowered_text
        else:
            best_match = lowered_text

    # 5. Ajustar la capitalización para que coincida con el token original
    corrected_text = capitalize_like(original_text, best_match)
    return corrected_text


def correct_text(text: str, nlp, sym_spell, try_segmentation: bool = True) -> str:
    doc = nlp(text)
    corrected_tokens = []

    for token in doc:
        if token.is_alpha:
            # Corrige solo si es alfabético
            corrected = correct_token(token.text, sym_spell, try_segmentation)
            corrected_tokens.append(corrected)
        else:
            # Dejar intacto puntuación, dígitos, etc.
            corrected_tokens.append(token.text)

    # Reconstruir la frase
    return " ".join(corrected_tokens)


def load_corrector_model(spacy_model: str,
                         frequency_dictionary_path: str,
                         max_edit_distance: int = 3,  # <-- prueba con 3
                         prefix_length: int = 7):
    nlp = spacy.load(spacy_model)
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                         prefix_length=prefix_length)

    if not sym_spell.load_dictionary(frequency_dictionary_path, 0, 1, separator=" ", encoding="utf-8"):
        raise ValueError(f"No se pudo cargar el diccionario {frequency_dictionary_path}")

    return nlp, sym_spell

def refine_text(original_text, corrected_text, tokenizer, model, max_length=256):
    """
    Envía al modelo un prompt que incluye:
    - El texto original.
    - El texto corregido por SymSpell.
    Para que genere una versión final más coherente.
    """
    # Prompt básico de ejemplo
    prompt = (
        f"original:\n{original_text}\n\n"
        f"corregido:\n{corrected_text}\n\n"
        "final:"
    )

    encoded = tokenizer(prompt, return_tensors="tf", padding=True)

    # encoded tiene, entre otros, las keys "input_ids" y "attention_mask"
    input_ids = encoded["input_ids"]         # shape => (batch_size, seq_length)
    attention_mask = encoded["attention_mask"]  # shape => (batch_size, seq_length)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_k=5,
        top_p=0.95,
        temperature=0.2,
        num_return_sequences=1,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )


    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_text

def load_llm(model_name="google/mt5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

if __name__ == "__main__":
    spacy_model = "es_core_news_sm"
    frequency_dictionary_path = "spanish_frequency_dictionary.txt"

    # Cargar modelo y diccionario
    nlp, sym_spell = load_corrector_model(
        spacy_model=spacy_model,
        frequency_dictionary_path=frequency_dictionary_path,
        max_edit_distance=3,  # Aumenta a 3
        prefix_length=7
    )

    text_ocr = ("Essta es una prueva de ectraccion de teexto. "
                "Connoscida cosa sea a todos los queesta carta uieren como yo "
                "don Fferrando por la gracia de dios hey de Castiella.")

    corrected_text = correct_text(text_ocr, nlp, sym_spell, try_segmentation=True)

    print("Texto original : ", text_ocr)
    print("Texto corregido: ", corrected_text)

    tokenizer, model = load_llm("google/mt5-small")

    final_text = refine_text(text_ocr, corrected_text, tokenizer, model)
    print("\n=== Versión final (refinada con GPT) ===")
    print(final_text)