# mi_proyecto_ia_api_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import uvicorn

# -----------------------------
# Definir app y modelo
# -----------------------------
app = FastAPI(title="Traductor Inglés -> Español")

class Texto(BaseModel):
    texto: str

model_name = "facebook/mbart-large-50-many-to-many-mmt"
print("Cargando modelo, esto puede tardar unos segundos...")
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Modelo cargado en {device}")

# -----------------------------
# Función de traducción
# -----------------------------
def traducir_en_es(texto: str) -> str:
    tokenizer.src_lang = "en_XX"  # inglés
    inputs = tokenizer(texto, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],  # español
        max_length=256,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# -----------------------------
# Endpoint de traducción
# -----------------------------
@app.post("/traducir")
def traducir_api(data: Texto):
    traduccion = traducir_en_es(data.texto)
    return {"traduccion": traduccion}


    