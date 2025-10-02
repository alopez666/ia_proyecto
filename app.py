# app.py
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import gradio as gr
from langdetect import detect, LangDetectException

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
# MAPA COMPLETO DE IDIOMAS (langdetect -> mBART)
# -----------------------------
LANG_CODE_MAP = {
    'af': 'af_ZA', 'ar': 'ar_AR', 'az': 'az_AZ', 'bn': 'bn_IN',
    'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX',
    'et': 'et_EE', 'fa': 'fa_IR', 'fi': 'fi_FI', 'fr': 'fr_XX',
    'gl': 'gl_ES', 'gu': 'gu_IN', 'he': 'he_IL', 'hi': 'hi_IN',
    'hr': 'hr_HR', 'id': 'id_ID', 'it': 'it_IT', 'ja': 'ja_XX',
    'ka': 'ka_GE', 'kk': 'kk_KZ', 'km': 'km_KH', 'ko': 'ko_KR',
    'lt': 'lt_LT', 'lv': 'lv_LV', 'mk': 'mk_MK', 'ml': 'ml_IN',
    'mn': 'mn_MN', 'mr': 'mr_IN', 'my': 'my_MM', 'ne': 'ne_NP',
    'nl': 'nl_XX', 'pl': 'pl_PL', 'ps': 'ps_AF', 'pt': 'pt_XX',
    'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'sl': 'sl_SI',
    'sv': 'sv_SE', 'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN',
    'th': 'th_TH', 'tl': 'tl_XX', 'tr': 'tr_TR', 'uk': 'uk_UA',
    'ur': 'ur_PK', 'vi': 'vi_VN', 'zh-cn': 'zh_CN'
}

# -----------------------------
# Función de traducción
# -----------------------------
def traducir_en_es(texto: str) -> str:
    """
    Esta función detecta el idioma de entrada y lo traduce al español.
    """
    if not texto.strip():
        return ""

    try:
        # Detectar el idioma de entrada
        lang_code_short = detect(texto)
        # Convertir al formato de mBART (ej: 'fr' -> 'fr_XX')
        src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
    except LangDetectException:
        # Si la detección falla, se asume inglés por defecto
        src_lang_code = "en_XX"

    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(texto, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],  # Destino fijo: español
        max_length=1024,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# -----------------------------
# Interfaz de Gradio (MODIFICADA)
# -----------------------------
iface = gr.Interface(
    fn=traducir_en_es,
    inputs=gr.Textbox(
        lines=5,
        label="Texto en cualquier idioma",
        placeholder="Escribe aquí el texto que quieres traducir..."
    ),
    outputs=gr.Textbox(label="Traducción al Español"),
    title="🤖 Traductor IA Multilingüe a Español",
    description="Este traductor detecta automáticamente el idioma de entrada (entre 50 opciones) y lo traduce al español usando el modelo mBART de Facebook.",
    examples=[
        ["The quick brown fox jumps over the lazy dog."], # Inglés
        ["La vie est belle quand on poursuit ses rêves."], # Francés
        [" Künstliche Intelligenz verändert unsere Welt."], # Alemán
        ["人生はチョコレートの箱のようなものだ。"] # Japonés
    ],
    allow_flagging="never"
)

# -----------------------------
# Lanzar la app (Gradio)
# -----------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)
