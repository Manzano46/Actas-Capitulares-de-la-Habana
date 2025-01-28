import re
import unicodedata
import spacy
import tensorflow as tf
from symspellpy import SymSpell, Verbosity
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Tokenización y separación de palabras con spaCy
def split_words_with_spacy(text, nlp):
    doc = nlp(text)
    return " ".join([token.text for token in doc])

# Configurar SymSpell
def setup_symspell(dictionary_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=10)
    try:
        # Cargamos el diccionario (frecuencia de palabras en español)
        if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding="utf-8"):
            raise FileNotFoundError(f"El diccionario no se pudo cargar desde {dictionary_path}")
    except Exception as e:
        print(f"Error al cargar el diccionario: {e}")
        return None
    return sym_spell

# Separar palabras mal unidas basándose en reglas simples
def split_compound_words_with_rules(text):
    patterns = {
        "queesta": "que esta",
        "portodos": "por todos",
        "entodo": "en todo",
    }
    for compound, correction in patterns.items():
        text = re.sub(rf"\b{compound}\b", correction, text)
    return text

# Corrección específica para palabras mal reconocidas
def correct_specific_words(text):
    corrections = {
        "hey": "Rey",
        "aferrando": "Fernando"
    }
    for wrong, correct in corrections.items():
        text = re.sub(rf"\b{wrong}\b", correct, text)
    return text

# Corrección ortográfica con SymSpell
def correct_and_split_words(text, sym_spell):
    corrected_text = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_text.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected_text)

# Normalización de grafías antiguas (según diccionario personalizado)
def custom_lemmatizer(text, lemma_dict):
    words = text.split()
    return " ".join([lemma_dict.get(word, word) for word in words])

# Corrección de tildes según un mapa básico
def add_accents(text):
    accent_map = {
        "rey": "Rey",
        "leon": "León",
        "dios": "Dios",
        "murcia": "Murcia",
        "toledo": "Toledo",
        "connoscida": "conocida",
        "fablar": "hablar",
    }
    for word, accented in accent_map.items():
        text = re.sub(rf"\b{word}\b", accented, text, flags=re.IGNORECASE)
    return text

# Eliminar repeticiones en el texto final
def remove_repeated_phrases(text):
    seen = set()
    words = text.split()
    filtered = []
    for word in words:
        if word not in seen:
            filtered.append(word)
            seen.add(word)
    return " ".join(filtered)

# Refinar el texto con un modelo de lenguaje (LLM) usando TensorFlow
def refine_text_with_llm(text, tokenizer, model, max_length=200):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Normalizar Unicode (NFC) para evitar caracteres raros
    text = unicodedata.normalize("NFC", text)

    # Limpiar de símbolos poco comunes
    text = re.sub(r"[^\w\s.,¿?¡!]", "", text)

    inputs = tokenizer(
        text + " " + tokenizer.eos_token,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return remove_repeated_phrases(refined_text)

# Cargar el modelo y el tokenizador desde Hugging Face
def load_llm(model_name="mrm8488/spanish-gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# Pipeline principal de mejoras
def improved_pipeline_with_llm(text, sym_spell, lemma_dict, nlp, tokenizer, model):
    # Paso 1: Separar palabras mal unidas (según reglas específicas)
    text = split_compound_words_with_rules(text)

    # Paso 2: Corrección puntual de ciertas palabras
    text = correct_specific_words(text)

    # Paso 3: Corrección ortográfica con SymSpell
    text = correct_and_split_words(text, sym_spell)

    # Paso 4: Normalización de grafías antiguas según diccionario personalizado
    text = custom_lemmatizer(text, lemma_dict)

    # Paso 5: Corrección de tildes
    text = add_accents(text)

    # Paso 6: Tokenización con spaCy (para limpiar o re-separar)
    text = split_words_with_spacy(text, nlp)

    # Paso 7: Refinar texto con modelo de lenguaje
    text = refine_text_with_llm(text, tokenizer, model)

    return text

# Ejecución de prueba
if __name__ == "__main__":
    raw_text = """Connoscida cosa sea a todos los queesta carta uieren como yo don Fferrando por la gracia de dios hey de Castiella."""
    dictionary_path = "spanish_frequency_dictionary.txt"

    # Cargamos SymSpell
    sym_spell = setup_symspell(dictionary_path)
    if not sym_spell:
        raise SystemExit("No se pudo cargar SymSpell. Saliendo...")

    # Diccionario de palabras antiguas y sus lemas
    lemma_dict = {
        "fablar": "hablar",
        "connoscida": "conocida",
        "queesta": "que esta"
    }

    # Cargar modelo de spaCy
    nlp = spacy.load("es_core_news_sm")

    # Cargar modelo y tokenizador de lenguaje
    tokenizer, model = load_llm("mrm8488/spanish-gpt2")

    # Ejecutar el pipeline
    refined_text = improved_pipeline_with_llm(
        raw_text,
        sym_spell,
        lemma_dict,
        nlp,
        tokenizer,
        model
    )

    print("\n=== Texto Refinado ===")
    print(refined_text)


#flax-community/spanish-t5-small