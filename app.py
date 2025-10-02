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

# Se usa MBart50TokenizerFast para el tokenizer y MBartForConditionalGeneration para el modelo.
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Mover el modelo al dispositivo disponible (GPU si es posible, si no, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Poner el modelo en modo de evaluación
print(f"Modelo cargado en {device}")

# -----------------------------
# MAPA DE CÓDIGOS DE IDIOMA (Funcional)
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

# Diccionario para mostrar nombres amigables (VERSIÓN COMPLETA)
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
# Función de traducción
# -----------------------------
def traducir_en_es(texto: str) -> str:
    """
    Toma un texto, lo divide en párrafos, detecta el idioma de cada uno
    y lo traduce al español.
    """
    if not texto or not texto.strip():
        return "" # Devuelve vacío si la entrada está vacía

    # Divide el texto en párrafos y elimina los que estén vacíos
    parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    bloques_de_texto = []

    for parrafo in parrafos:
        try:
            # Detecta el idioma del párrafo
            lang_code_short = detect(parrafo)
            # Busca el código de idioma completo para el modelo mBART
            src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        except LangDetectException:
            # Si la detección falla, asume inglés por defecto
            lang_code_short = 'en'
            src_lang_code = "en_XX"

        # Prepara el texto para el modelo
        tokenizer.src_lang = src_lang_code
        inputs = tokenizer(parrafo, return_tensors="pt").to(device)

        # Genera la traducción, forzando la salida en español
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
            max_length=1024, # Límite de tokens para la traducción
            num_beams=4,      # Usa beam search para mejores resultados
            length_penalty=1.2,
            early_stopping=True
        )
        
        # Decodifica los tokens generados para obtener el texto final
        traduccion_parrafo = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, lang_code_short.upper())
        
        # Formatea la salida para indicar el idioma de origen
        bloque_formateado = f"Idioma de origen: {nombre_idioma}.\n{traduccion_parrafo}"
        bloques_de_texto.append(bloque_formateado)

    # Une todos los párrafos traducidos con un doble salto de línea
    return "\n\n".join(bloques_de_texto)

# -------------------------------------------------------------
# Interfaz de Gradio Mejorada con gr.Blocks
# -------------------------------------------------------------

# CSS personalizado para afinar detalles y darle un toque minimalista.
custom_css = """
/* Estilo general del contenedor */
#main_container {
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    padding: 20px;
}
/* Espaciado del título */
#app_title {
    text-align: center;
    color: #333;
    margin-bottom: 5px;
}
/* Espaciado de la descripción */
#app_description {
    text-align: center;
    color: #555;
    margin-bottom: 25px;
}
"""

# Usamos gr.Blocks para tener control total sobre el diseño.
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as iface:
    
    # 1. Título y Descripción usando Markdown
    with gr.Column(elem_id="main_container"):
        gr.Markdown("# 🤖 Traductor IA Multilingüe a Español", elem_id="app_title")
        gr.Markdown("Detecta automáticamente el idioma de cada párrafo y lo traduce al español. ¡Prueba a mezclar idiomas!", elem_id="app_description")

        # 2. Layout de dos columnas para entrada y salida
        with gr.Row():
            # Columna de la izquierda: Entrada de texto
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    lines=15,
                    label="Texto a Traducir",
                    placeholder="Escribe aquí el texto que quieres traducir...\n\nThe quick brown fox jumps over the lazy dog."
                )
                
                translate_button = gr.Button("Traducir 📝", variant="primary")

            # Columna de la derecha: Salida de la traducción
            with gr.Column(scale=1):
                output_textbox = gr.Textbox(
                    lines=15,
                    label="Resultado de la Traducción",
                    interactive=False, # El usuario no puede escribir aquí
                    show_copy_button=True # Botón para copiar el resultado fácilmente
                )
        
        # 3. Sección de Ejemplos Interactivos
        gr.Examples(
            examples=[
                "The quick brown fox jumps over the lazy dog.",
                "La vie est belle quand on poursuit ses rêves.",
                "こんにちは、世界！", # Japonés: Hola, mundo!
                "The rain in Spain stays mainly in the plain.\nDer Regen in Spanien bleibt hauptsächlich in der Ebene." # Inglés y Alemán
            ],
            inputs=input_textbox,
            outputs=output_textbox,
            fn=traducir_en_es,
            cache_examples=True # Acelera la ejecución de los ejemplos
        )
        
        # 4. Pie de página (Footer)
        gr.Markdown("<p style='text-align:center; color: #888;'>Proyecto IA-Traductor v1</p>")

    # Conectar el botón a la función de traducción
    translate_button.click(
        fn=traducir_en_es,
        inputs=input_textbox,
        outputs=output_textbox,
        api_name="translate" # Opcional: nombre para la API
    )

# -----------------------------
# Lanzar la app
# -----------------------------
if __name__ == "__main__":
    # Escucha en todas las interfaces de red (0.0.0.0) para que sea accesible desde Docker
    iface.launch(server_name="0.0.0.0", server_port=8000)