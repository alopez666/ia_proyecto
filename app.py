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
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    return texto

def detectar_idioma_robusto(texto: str) -> tuple:
    """
    Detecta el idioma de forma más robusta, manejando textos cortos.
    Retorna: (código_corto, código_mbart, nombre_idioma)
    """
    try:
        muestra = texto if len(texto) < 100 else texto[:300]
        lang_code_short = detect(muestra)
        src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, lang_code_short.upper())
        return lang_code_short, src_lang_code, nombre_idioma
    except LangDetectException:
        return 'en', 'en_XX', 'Inglés'

# -----------------------------
# Función de traducción CORREGIDA
# -----------------------------
def traducir_en_es(texto: str) -> str:
    """
    Traduce un texto al español, detectando el idioma de cada párrafo.
    VERSIÓN CORREGIDA: Ahora fuerza correctamente el español como idioma de destino.
    """
    if not texto or not texto.strip():
        return "⚠️ Por favor, ingresa un texto para traducir."
    
    texto = limpiar_texto(texto)
    
    if len(texto.split()) < 2:
        return "⚠️ El texto es demasiado corto. Intenta con al menos una frase completa."
    
    parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
    if not parrafos:
        return "⚠️ No se detectó texto válido."
    
    traducciones_finales = []

    for parrafo in parrafos:
        lang_short, src_lang_code, nombre_idioma = detectar_idioma_robusto(parrafo)
        
        if lang_short == 'es':
            traducciones_finales.append(f"✓ Ya está en español:\n{parrafo}")
            continue
        
        # ⭐ CORRECCIÓN CRÍTICA: Configurar AMBOS idiomas explícitamente
        tokenizer.src_lang = src_lang_code
        
        # Tokenizar el texto de entrada
        inputs = tokenizer(
            parrafo, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # ⭐ SOLUCIÓN: Obtener el ID del token de español y forzarlo correctamente
        # El token "es_XX" debe estar al inicio de la secuencia generada
        spanish_token_id = tokenizer.convert_tokens_to_ids("es_XX")
        
        # Generar traducción CON CONFIGURACIÓN CORRECTA
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=spanish_token_id,  # ⭐ Forzar español como primer token
                max_length=512,
                num_beams=5,
                early_stopping=True,
                decoder_start_token_id=spanish_token_id  # ⭐ CRÍTICO: También configurar aquí
            )
        
        # ⭐ DECODIFICACIÓN CORRECTA: Configurar el idioma de destino antes de decodificar
        tokenizer.tgt_lang = "es_XX"
        traduccion_parrafo = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        traduccion_parrafo = limpiar_texto(traduccion_parrafo)
        
        if not traduccion_parrafo or len(traduccion_parrafo) < 2:
            traduccion_parrafo = f"⚠️ Error al traducir: {parrafo[:50]}..."
        
        traducciones_finales.append(
            f"🌐 Idioma detectado: {nombre_idioma} ({src_lang_code})\n"
            f"📝 Traducción al español:\n{traduccion_parrafo}"
        )

    return "\n\n" + "─" * 60 + "\n\n".join(traducciones_finales)

# -------------------------------------------------------------
# Interfaz de Gradio
# -------------------------------------------------------------
custom_css = """
#main_container {
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
#app_title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 5px;
    font-weight: bold;
}
#app_description {
    text-align: center;
    color: #34495e;
    margin-bottom: 25px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as iface:
    with gr.Column(elem_id="main_container"):
        gr.Markdown("# 🌍 Traductor IA Multilingüe → Español", elem_id="app_title")
        gr.Markdown(
            "**Traducción automática con detección de idioma**  \n"
            "Soporta más de 50 idiomas. ¡Prueba ahora!",
            elem_id="app_description"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    lines=12,
                    label="📥 Texto Original",
                    placeholder="Escribe o pega aquí el texto en cualquier idioma...\n\n"
                                "Ejemplo:\nGuten Tag! Ich lerne Deutsch.\n\n"
                                "El sistema detectará automáticamente el idioma y lo traducirá al español.",
                    max_lines=20
                )
                
                with gr.Row():
                    translate_button = gr.Button("🔄 Traducir a Español", variant="primary", size="lg")
                    clear_button = gr.Button("🗑️ Limpiar", variant="secondary")
                
            with gr.Column(scale=1):
                output_textbox = gr.Textbox(
                    lines=12,
                    label="📤 Traducción en Español",
                    interactive=False,
                    show_copy_button=True,
                    max_lines=20
                )
        
        gr.Markdown("### 📋 Ejemplos para probar:")
        
        gr.Examples(
            examples=[
                ["Hello world! How are you today? I hope you're having a great day!"],
                ["Guten Tag! Ich lerne Deutsch. Es macht mir viel Spaß!"],
                ["Bonjour! Comment allez-vous aujourd'hui? La vie est belle."],
                ["Ciao! Come stai? Sono molto felice di vederti oggi."],
                ["こんにちは！今日はとてもいい天気ですね。"],
                ["Olá! Como você está? Espero que tenha um ótimo dia!"],
                ["안녕하세요! 오늘 날씨가 정말 좋네요. 행복한 하루 되세요!"],
                ["Привет! Как дела? Надеюсь, у тебя всё хорошо сегодня."],
                ["مرحبا! كيف حالك اليوم؟ أتمنى لك يوماً سعيداً!"],
                ["你好！今天天气真好。祝你有美好的一天！"]
            ],
            inputs=input_textbox,
            outputs=output_textbox,
            fn=traducir_en_es,
            cache_examples=False
        )
        
        gr.Markdown(
            "<div style='text-align:center; margin-top: 20px; padding: 15px; "
            "background: rgba(255,255,255,0.8); border-radius: 8px;'>"
            "<p style='color: #2c3e50; margin: 5px;'>"
            "💡 <strong>Consejo:</strong> Funciona mejor con frases completas.</p>"
            "<p style='color: #7f8c8d; margin: 5px;'>"
            "🌐 Idiomas soportados: Alemán, Inglés, Francés, Italiano, Portugués, "
            "Chino, Japonés, Coreano, Árabe, Ruso, y 40+ más.</p>"
            "</div>",
            elem_id="tips"
        )
        
        gr.Markdown(
            "<p style='text-align:center; color: #95a5a6; margin-top: 15px;'>"
            "🤖 Traductor IA v2.1 | Powered by mBART-50</p>"
        )

    # Event handlers
    translate_button.click(
        fn=traducir_en_es,
        inputs=input_textbox,
        outputs=output_textbox,
        api_name="translate"
    )
    
    clear_button.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[input_textbox, output_textbox]
    )

# -----------------------------
# Lanzar la app
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Iniciando Traductor IA Multilingüe a Español")
    print("="*60)
    print(f"📱 Servidor: http://0.0.0.0:8000")
    print(f"💻 Dispositivo: {device}")
    print("="*60 + "\n")
    
    iface.launch(
        server_name="0.0.0.0", 
        server_port=8000,
        share=False
    )