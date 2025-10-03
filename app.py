from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import gradio as gr
from langdetect import detect, LangDetectException
import re

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
    'ur': 'ur_PK', 'vi': 'vi_VN', 'zh-cn': 'zh_CN', 'zh-tw': 'zh_CN'
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
    'ur': 'Urdu', 'vi': 'Vietnamita', 'zh-cn': 'Chino', 'zh-tw': 'Chino'
}

def limpiar_texto(texto: str) -> str:
    """Limpia el texto de entrada eliminando espacios extras y caracteres problemáticos."""
    texto = re.sub(r'\s+', ' ', texto)  # Normalizar espacios
    texto = texto.strip()
    return texto

def detectar_idioma_robusto(texto: str) -> tuple:
    """
    Detecta el idioma de forma más robusta, manejando textos cortos.
    Retorna: (código_corto, código_mbart, nombre_idioma)
    """
    try:
        # Para textos muy cortos, usar el texto completo
        muestra = texto if len(texto) < 100 else texto[:300]
        lang_code_short = detect(muestra)
        
        # Mapear el código detectado
        src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, lang_code_short.upper())
        
        return lang_code_short, src_lang_code, nombre_idioma
    except LangDetectException:
        # Si falla, asumir inglés
        return 'en', 'en_XX', 'Inglés'

# -----------------------------
# Función de traducción OPTIMIZADA Y CORREGIDA
# -----------------------------
def traducir_en_es(texto: str) -> str:
    """
    Traduce un texto al español, detectando el idioma de cada párrafo.
    Versión corregida para manejar textos cortos y largos correctamente.
    """
    if not texto or not texto.strip():
        return "⚠️ Por favor, ingresa un texto para traducir."
    
    texto = limpiar_texto(texto)
    
    # Si el texto es muy corto (menos de 3 palabras), advertir
    if len(texto.split()) < 3:
        return "⚠️ El texto es demasiado corto. Intenta con al menos una frase completa."
    
    # Dividir en párrafos
    parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    if not parrafos:
        return "⚠️ No se detectó texto válido."
    
    traducciones_finales = []

    for parrafo in parrafos:
        # Detectar idioma del párrafo
        lang_short, src_lang_code, nombre_idioma = detectar_idioma_robusto(parrafo)
        
        # Si ya está en español, no traducir
        if lang_short == 'es':
            traducciones_finales.append(f"✓ Ya está en español:\n{parrafo}")
            continue
        
        # Configurar idioma de origen
        tokenizer.src_lang = src_lang_code
        
        # Tokenizar con límites adaptativos
        longitud_texto = len(parrafo.split())
        max_input_length = min(512, max(128, longitud_texto * 2))
        max_output_length = min(512, max(64, longitud_texto * 3))
        
        inputs = tokenizer(
            parrafo, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_input_length
        ).to(device)
        
        # Generar traducción con parámetros optimizados
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
                max_length=max_output_length,
                min_length=max(10, longitud_texto // 2),  # Evitar traducciones muy cortas
                num_beams=3,  # Reducido de 5 a 3 para más variabilidad
                length_penalty=1.0,  # Reducido de 1.2 a 1.0
                early_stopping=True,
                no_repeat_ngram_size=3,  # Evitar repeticiones
                do_sample=False,  # Desactivar sampling para consistencia
                temperature=1.0
            )
        
        # Decodificar
        traduccion_parrafo = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        traduccion_parrafo = limpiar_texto(traduccion_parrafo)
        
        # Verificar que la traducción no esté vacía
        if not traduccion_parrafo or len(traduccion_parrafo) < 3:
            traduccion_parrafo = f"⚠️ No se pudo traducir este párrafo: {parrafo[:50]}..."
        
        traducciones_finales.append(
            f"🌐 Detectado: {nombre_idioma}\n📝 Traducción:\n{traduccion_parrafo}"
        )

    return "\n\n" + "─" * 50 + "\n\n".join(traducciones_finales)

# -------------------------------------------------------------
# Interfaz de Gradio
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
        gr.Markdown(
            "Detecta automáticamente el idioma y traduce al español con IA. "
            "¡Prueba con diferentes idiomas!",
            elem_id="app_description"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    lines=15,
                    label="📥 Texto a Traducir",
                    placeholder="Escribe aquí el texto que quieres traducir...\n\nEjemplo:\nThe quick brown fox jumps over the lazy dog."
                )
                translate_button = gr.Button("🔄 Traducir", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_textbox = gr.Textbox(
                    lines=15,
                    label="📤 Resultado de la Traducción",
                    interactive=False,
                    show_copy_button=True
                )
        
        gr.Examples(
            examples=[
                ["Hello world! How are you today?"],
                ["Bonjour, comment allez-vous? La vie est belle."],
                ["Guten Tag! Ich lerne Deutsch."],
                ["Ciao! Come stai? Sono felice di vederti."],
                ["こんにちは。今日はいい天気ですね。"],
                ["Olá! Como você está? Tenha um ótimo dia!"],
                ["안녕하세요! 오늘 날씨가 정말 좋네요."],
                ["Привет! Как дела? Надеюсь, у тебя всё хорошо."]
            ],
            inputs=input_textbox,
            outputs=output_textbox,
            fn=traducir_en_es,
            cache_examples=False,
            label="📋 Ejemplos de Prueba"
        )
        
        gr.Markdown(
            "<p style='text-align:center; color: #888; margin-top: 20px;'>"
            "💡 Consejo: Funciona mejor con frases completas. "
            "Soporta más de 50 idiomas."
            "</p>"
        )
        gr.Markdown("<p style='text-align:center; color: #888;'>Proyecto IA-Traductor v2.0</p>")

    translate_button.click(
        fn=traducir_en_es,
        inputs=input_textbox,
        outputs=output_textbox,
        api_name="translate"
    )

# -----------------------------
# Lanzar la app
# -----------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000)