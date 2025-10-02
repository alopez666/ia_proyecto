# app.py
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import gradio as gr
from langdetect import detect, LangDetectException

# -----------------------------
# Cargar modelo y tokenizer
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
# MAPA DE CÓDIGOS DE IDIOMA (langdetect -> mBART)
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

# Diccionario para mostrar nombres amigables de idiomas en la interfaz
LANG_NAME_MAP = {
    'en': 'Inglés', 'fr': 'Francés', 'de': 'Alemán', 'es': 'Español',
    'it': 'Italiano', 'pt': 'Portugués', 'ru': 'Ruso', 'ja': 'Japonés',
    'zh-cn': 'Chino', 'ar': 'Árabe', 'hi': 'Hindi', 'ko': 'Coreano',
}

# -----------------------------
# Función de traducción (MODIFICADA PARA EL NUEVO FORMATO)
# -----------------------------
def traducir_en_es(texto: str) -> str:
    if not texto.strip():
        return ""

    parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    # --- CAMBIO ---: Guardaremos bloques de texto ya formateados
    bloques_de_texto = []

    for parrafo in parrafos:
        try:
            lang_code_short = detect(parrafo)
            src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        except LangDetectException:
            lang_code_short = 'en'
            src_lang_code = "en_XX"

        tokenizer.src_lang = src_lang_code
        inputs = tokenizer(parrafo, return_tensors="pt").to(device)

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
            max_length=1024,
            num_beams=4,
            early_stopping=True
        )
        
        traduccion_parrafo = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, lang_code_short.upper())
        
        # --- CAMBIO PRINCIPAL ---
        # Creamos el bloque de texto con el formato exacto que pediste
        bloque_formateado = f"Idioma de origen: {nombre_idioma}.\n{traduccion_parrafo}"
        bloques_de_texto.append(bloque_formateado)

    # Unimos cada bloque con dos saltos de línea para dejar un espacio en blanco entre ellos
    return "\n\n".join(bloques_de_texto)

# -----------------------------
# Interfaz de Gradio (MODIFICADA)
# -----------------------------
iface = gr.Interface(
    fn=traducir_en_es,
    inputs=gr.Textbox(
        lines=10,
        label="Texto en cualquier idioma (puedes mezclar idiomas por párrafos)",
        placeholder="Escribe aquí el texto que quieres traducir..."
    ),
    # --- CAMBIO ---: Regresamos a un Textbox normal
    outputs=gr.Textbox(label="Traducción al Español"),
    title="🤖 Traductor IA Multilingüe a Español",
    description="Este traductor detecta el idioma de cada párrafo y lo traduce a español, indicando el idioma de origen.",
    examples=[
        ["The quick brown fox jumps over the lazy dog.\nLa vie est belle quand on poursuit ses rêves."]
    ],
    allow_flagging="never"
)

# -----------------------------
# Lanzar la app (Gradio)
# -----------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)