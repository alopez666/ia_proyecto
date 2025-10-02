# app.py
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import gradio as gr

# -----------------------------
# Cargar modelo
# -----------------------------
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
    """
    Esta función toma un texto en inglés y lo traduce al español usando el modelo MBart.
    """
    if not texto.strip():
        return ""  # Evita procesar texto vacío
        
    tokenizer.src_lang = "en_XX"  # inglés
    inputs = tokenizer(texto, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],  # español
        max_length=1024,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# -----------------------------
# Interfaz de Gradio
# -----------------------------
iface = gr.Interface(
    fn=traducir_en_es,
    inputs=gr.Textbox(
        lines=5, 
        label="Texto en Inglés", 
        placeholder="Escribe aquí el texto que quieres traducir..."
    ),
    outputs=gr.Textbox(label="Traducción al Español"),
    title="🤖 Traductor IA (Inglés a Español)",
    description="Este es un traductor basado en el modelo MBart de Facebook. Escribe una frase en inglés para ver la magia.",
    examples=[
        ["The quick brown fox jumps over the lazy dog."],
        ["I am studying artificial intelligence at the university."],
        ["The workflow builds a Docker image and pushes it to Docker Hub."]
    ],
    allow_flagging="never"
)

# -----------------------------
# Lanzar la app (Gradio)
# -----------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)