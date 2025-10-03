from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import gradio as gr
from langdetect import detect, LangDetectException
import re
import time

# -----------------------------
# Cargar modelo y tokenizer una sola vez
# -----------------------------
print("Cargando modelo y tokenizer, esto puede tardar unos segundos...")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Modelo '{MODEL_NAME}' cargado exitosamente en {DEVICE}")
except Exception as e:
    print(f"🚨 Error al cargar el modelo: {e}")
    exit()

# -----------------------------
# MAPA DE CÓDIGOS DE IDIOMA (sin cambios)
# -----------------------------
LANG_CODE_MAP = {
    'af': 'af_ZA', 'ar': 'ar_AR', 'az': 'az_AZ', 'bn': 'bn_IN', 'cs': 'cs_CZ', 
    'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'et': 'et_EE', 'fa': 'fa_IR', 
    'fi': 'fi_FI', 'fr': 'fr_XX', 'gl': 'gl_ES', 'gu': 'gu_IN', 'he': 'he_IL', 
    'hi': 'hi_IN', 'hr': 'hr_HR', 'id': 'id_ID', 'it': 'it_IT', 'ja': 'ja_XX', 
    'ka': 'ka_GE', 'kk': 'kk_KZ', 'km': 'km_KH', 'ko': 'ko_KR', 'lt': 'lt_LT', 
    'lv': 'lv_LV', 'mk': 'mk_MK', 'ml': 'ml_IN', 'mn': 'mn_MN', 'mr': 'mr_IN', 
    'my': 'my_MM', 'ne': 'ne_NP', 'nl': 'nl_XX', 'pl': 'pl_PL', 'ps': 'ps_AF', 
    'pt': 'pt_XX', 'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'sl': 'sl_SI', 
    'sv': 'sv_SE', 'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN', 'th': 'th_TH', 
    'tl': 'tl_XX', 'tr': 'tr_TR', 'uk': 'uk_UA', 'ur': 'ur_PK', 'vi': 'vi_VN', 
    'zh-cn': 'zh_CN', 'zh-tw': 'zh_CN'
}

LANG_NAME_MAP = {
    'af': 'Afrikáans', 'ar': 'Árabe', 'az': 'Azerbaiyano', 'bn': 'Bengalí', 'cs': 'Checo', 
    'de': 'Alemán', 'en': 'Inglés', 'es': 'Español', 'et': 'Estonio', 'fa': 'Persa', 
    'fi': 'Finlandés', 'fr': 'Francés', 'gl': 'Gallego', 'gu': 'Guyaratí', 'he': 'Hebreo', 
    'hi': 'Hindi', 'hr': 'Croata', 'id': 'Indonesio', 'it': 'Italiano', 'ja': 'Japonés', 
    'ka': 'Georgiano', 'kk': 'Kazajo', 'km': 'Jemer', 'ko': 'Coreano', 'lt': 'Lituano', 
    'lv': 'Letón', 'mk': 'Macedonio', 'ml': 'Malayalam', 'mn': 'Mongol', 'mr': 'Maratí', 
    'my': 'Birmano', 'ne': 'Nepalí', 'nl': 'Neerlandés', 'pl': 'Polaco', 'ps': 'Pastún', 
    'pt': 'Portugués', 'ro': 'Rumano', 'ru': 'Ruso', 'si': 'Cingalés', 'sl': 'Esloveno', 
    'sv': 'Sueco', 'sw': 'Suajili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Tailandés', 
    'tl': 'Tagalo', 'tr': 'Turco', 'uk': 'Ucraniano', 'ur': 'Urdu', 'vi': 'Vietnamita', 
    'zh-cn': 'Chino', 'zh-tw': 'Chino'
}


def limpiar_texto(texto: str) -> str:
    """Limpia el texto de entrada eliminando espacios extra."""
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def detectar_idioma_robusto(texto: str) -> tuple:
    """Detecta el idioma de forma robusta, devolviendo 'en' como fallback."""
    try:
        muestra = texto if len(texto) < 500 else texto[:500]
        lang_code_short = detect(muestra)
        src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, "Desconocido")
        return lang_code_short, src_lang_code, nombre_idioma
    except LangDetectException:
        return 'en', 'en_XX', 'Inglés (por defecto)'

# -----------------------------
# 🚀 FUNCIÓN DE TRADUCCIÓN ÚNICA Y OPTIMIZADA
# -----------------------------
def traducir_a_espanol(texto: str) -> str:
    """
    Traduce texto de múltiples idiomas al español con una configuración universal
    optimizada para calidad y velocidad.
    """
    if not texto or not texto.strip():
        return "⚠️ Por favor, ingresa un texto para traducir."
    
    inicio = time.time()
    texto_limpio = limpiar_texto(texto)
    
    # Detección de idioma
    lang_short, src_lang_code, nombre_idioma = detectar_idioma_robusto(texto_limpio)
    
    if lang_short == 'es':
        return f"✅ El texto ya está en Español:\n\n---\n{texto}\n---"

    # --- CONFIGURACIÓN DE GENERACIÓN UNIVERSAL Y ROBUSTA ---
    # Se usa una configuración equivalente al modo "Balanceado" para garantizar
    # alta calidad en todos los casos, incluyendo textos muy cortos.
    spanish_token_id = tokenizer.lang_code_to_id["es_XX"]
    
    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(texto_limpio, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    num_tokens = inputs.input_ids.shape[1]
    max_len = int(num_tokens * 3.0) + 10 # Margen generoso para la traducción
    
    gen_config = {
        "num_beams": 5,                 # Aumentado para máxima calidad
        "max_length": min(max_len, 1024),
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.5,
        "length_penalty": 1.0
    }

    # 🚀 GENERAR TRADUCCIÓN
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=spanish_token_id,
            **gen_config
        )
    
    traduccion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    traduccion_final = limpiar_texto(traduccion)

    if not traduccion_final:
        return "⚠️ Error: La traducción resultó en un texto vacío."

    tiempo_total = time.time() - inicio
    
    resultado = (
        f"🌐 {nombre_idioma} → Español\n"
        f"------------------------------------\n"
        f"{traduccion_final}\n"
        f"------------------------------------\n\n"
        f"⏱️ **Tiempo:** {tiempo_total:.2f} segundos"
    )
    
    return resultado

# -------------------------------------------------------------
# Interfaz de Gradio Rediseñada
# -------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        <div style='text-align: center;'>
            <h1>🌍 Traductor IA Ultrarrápido → Español</h1>
            <p>Escribe en cualquier idioma y obtén la traducción al instante.</p>
        </div>
        """
    )
    
    with gr.Row(equal_height=True):
        input_textbox = gr.Textbox(
            lines=10,
            label="📥 Texto Original (cualquier idioma)",
            placeholder="Escribe o pega aquí...\n\nPor ejemplo: Hello, how are you?",
            elem_id="input_textbox"
        )
        output_textbox = gr.Textbox(
            lines=10,
            label="📤 Traducción en Español",
            interactive=False,
            show_copy_button=True,
            elem_id="output_textbox"
        )
    
    with gr.Row():
        translate_button = gr.Button("🚀 Traducir Ahora", variant="primary", size="lg")
        clear_button = gr.Button("🗑️ Limpiar", variant="secondary", size="lg")
    
    with gr.Accordion("📋 Ejemplos para Probar", open=True):
        gr.Examples(
            examples=[
                ["Hello"],
                ["Good morning! How are you today?"],
                ["Guten Tag! Ich lerne Deutsch und es macht mir viel Spaß."],
                ["L'intelligence artificielle transforme notre monde à une vitesse incroyable."],
                ["人工智能正在改变我们的世界。"]
            ],
            inputs=input_textbox,
            outputs=output_textbox,
            fn=traducir_a_espanol,
            cache_examples=False
        )

# Event Handlers
translate_button.click(
    fn=traducir_a_espanol,
    inputs=input_textbox,
    outputs=output_textbox,
    api_name="translate"
)
clear_button.click(lambda: ("", ""), inputs=None, outputs=[input_textbox, output_textbox])

# -----------------------------
# Lanzar la aplicación
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LANZANDO LA APLICACIÓN DE TRADUCCIÓN IA (MODO OPTIMIZADO)")
    print("="*70 + "\n")
    iface.launch(server_name="0.0.0.0", server_port=8000)