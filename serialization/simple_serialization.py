model_name = "google/mt5-small"

from transformers import T5Tokenizer, TFT5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

raw_text = "Connoscida cosa sea a todos los queesta carta uieren..."

prompt = f"corrige el siguiente texto: {raw_text}"
input_ids = tokenizer.encode(prompt, return_tensors="tf")
outputs = model.generate(input_ids, max_length=64)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
