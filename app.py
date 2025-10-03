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
# MAPA DE CÓDIGOS DE IDIOMA
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

LANG_NAME_MAP = {
    'af': 'Afrikáans', 'ar': 'Árabe', 'az': 'Azerbaiyano', 'bn': 'Bengalí',
    'cs': 'Checo', 'de': 'Alemán', 'en': 'Inglés', 'es': 'Español',
    'et': 'Estonio', 'fa': 'Persa', 'fi': 'Finlandés', 'fr': 'Francés',
    'gl': 'Gallego', 'gu': 'Guyaratí', 'he': 'Hebreo', 'hi': 'Hindi',
    'hr': 'Croata', 'id': 'Indonesio', 'it': 'Italiano', 'ja': 'Japonés',
    'ka': 'Georgiano', 'kk': 'Kazajo', 'km': 'Jemer', 'ko': 'Coreano',
    'lt': 'Lituano', 'lv': 'Letón', 'mk': 'Macedonio', 'ml': 'Malayalam',
    'mn': 'Mongol', 'mr': 'Maratí', 'my': 'Birmano', 'ne': 'Nepalí',
    'nl': 'Neerlandés', 'pl': 'Polaco', 'ps': 'Pastún', 'pt': 'Portugués',
    'ro': 'Rumano', 'ru': 'Ruso', 'si': 'Cingalés', 'sl': 'Esloveno',
    'sv': 'Sueco', 'sw': 'Suajili', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Tailandés', 'tl': 'Tagalo', 'tr': 'Turco', 'uk': 'Ucraniano',
    'ur': 'Urdu', 'vi': 'Vietnamita', 'zh-cn': 'Chino'
}

# -----------------------------
# Función de traducción OPTIMIZADA
# -----------------------------
def traducir_en_es(texto: str) -> str:
    """
    Traduce un texto al español, detectando el idioma de cada párrafo
    y procesando cada uno de ellos en una sola llamada al modelo para mayor eficiencia.
    """
    if not texto or not texto.strip():
        return ""
    
    # 1. Dividir el texto en párrafos. Esto es útil para detectar diferentes idiomas
    # en un mismo texto y mantener la estructura.
    parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    traducciones_finales = []

    for parrafo in parrafos:
        # 2. Detectar el idioma del párrafo.
        try:
            # Usar una porción del párrafo para una detección más rápida y precisa.
            lang_code_short = detect(parrafo[:500])
            src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        except LangDetectException:
            # Si la detección falla, se asume inglés por defecto.
            lang_code_short = 'en'
            src_lang_code = "en_XX"

        tokenizer.src_lang = src_lang_code

        # 3. **OPTIMIZACIÓN CLAVE**: Tokenizar y traducir el PÁRRAFO COMPLETO de una sola vez.
        # Se elimina el bucle ineficiente que procesaba frase por frase.
        inputs = tokenizer(parrafo, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
            max_length=1024,  # Asegurarse de que haya espacio para el párrafo traducido
            num_beams=5,
            length_penalty=1.2,
            early_stopping=True
        )
        
        # 4. Decodificar la traducción completa del párrafo.
        traduccion_parrafo = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, lang_code_short.upper())
        traducciones_finales.append(f"Idioma de origen: {nombre_idioma}.\n{traduccion_parrafo}")

    # 5. Unir los párrafos traducidos con un doble salto de línea.
    return "\n\n".join(traducciones_finales)

# -------------------------------------------------------------
# Interfaz de Gradio (Sin cambios)
# -------------------------------------------------------------
custom_css = """
#main_container {
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    padding: 20px;
}
#app_title {
    text-align: center;
    color: #333;
    margin-bottom: 5px;
}
#app_description {
    text-align: center;
    color: #555;
    margin-bottom: 25px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as iface:
    with gr.Column(elem_id="main_container"):
        gr.Markdown("# 🤖 Traductor IA Multilingüe a Español", elem_id="app_title")
        gr.Markdown("Detecta automáticamente el idioma de cada párrafo y lo traduce al español. ¡Prueba a mezclar idiomas!", elem_id="app_description")
        with gr.Row():
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    lines=15,
                    label="Texto a Traducir",
                    placeholder="Escribe aquí el texto que quieres traducir...\n\nThe quick brown fox jumps over the lazy dog."
                )
                translate_button = gr.Button("Traducir 📝", variant="primary")
            with gr.Column(scale=1):
                output_textbox = gr.Textbox(
                    lines=15,
                    label="Resultado de la Traducción",
                    interactive=False,
                    show_copy_button=True
                )
        gr.Examples(
            examples=[
                "Every day is a new chance to start again.",
                "La vie est belle quand on apprend à l’apprécier.",
                "小さな努力が大きな結果につながります",
                "작은 시작이 큰 변화를 만들 수 있습니다."
            ],
            inputs=input_textbox,
            outputs=output_textbox,
            fn=traducir_en_es,
            cache_examples=True
        )
        gr.Markdown("<p style='text-align:center; color: #888;'>Proyecto IA-Traductor v1</p>")

    translate_button.click(
        fn=traducir_en_es,
        inputs=input_textbox,
        outputs=output_textbox,
        api_name="translate"
    )

# -----------------------------
# Lanzar la app (Sin cambios)
# -----------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)