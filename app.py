from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import gradio as gr
from langdetect import detect, LangDetectException
import re
import time

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
    """Limpia el texto de entrada."""
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    return texto

def detectar_idioma_robusto(texto: str) -> tuple:
    """Detecta el idioma de forma robusta."""
    try:
        muestra = texto if len(texto) < 100 else texto[:300]
        lang_code_short = detect(muestra)
        src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, lang_code_short.upper())
        return lang_code_short, src_lang_code, nombre_idioma
    except LangDetectException:
        return 'en', 'en_XX', 'Inglés'

# -----------------------------
# 🚀 FUNCIÓN DE TRADUCCIÓN ULTRARRÁPIDA
# -----------------------------
def traducir_en_es(texto: str, modo_velocidad: str = "⚡ Rápido (Recomendado)") -> str:
    """
    Traduce texto al español con 3 modos de velocidad:
    
    - ⚡ Rápido: Generación greedy (1-3 segundos) - Calidad buena
    - ⚖️ Balanceado: Beam search limitado (3-8 segundos) - Calidad muy buena  
    - 🎯 Preciso: Beam search completo (10-20 segundos) - Máxima calidad
    """
    if not texto or not texto.strip():
        return "⚠️ Por favor, ingresa un texto para traducir."
    
    inicio = time.time()
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
        
        # Configurar idioma de origen
        tokenizer.src_lang = src_lang_code
        
        # 📏 CÁLCULO INTELIGENTE DE LONGITUD
        # La traducción suele ser 0.8x - 1.5x la longitud del original
        num_palabras = len(parrafo.split())
        
        # Longitud máxima adaptativa (más eficiente)
        if num_palabras <= 10:  # Frase corta
            max_len = 30
        elif num_palabras <= 50:  # Párrafo pequeño
            max_len = 100
        elif num_palabras <= 150:  # Párrafo mediano
            max_len = 256
        else:  # Texto largo
            max_len = 512
        
        min_len = max(5, num_palabras // 2)  # Mínimo razonable
        
        # Tokenizar
        inputs = tokenizer(
            parrafo, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # Token español
        spanish_token_id = tokenizer.convert_tokens_to_ids("es_XX")
        
        # ⚡ CONFIGURACIÓN SEGÚN MODO DE VELOCIDAD
        if modo_velocidad == "⚡ Rápido (Recomendado)":
            # MODO GREEDY: Más rápido, buena calidad
            gen_config = {
                "max_length": max_len,
                "min_length": min_len,
                "num_beams": 1,  # ⚡ SIN beam search = 5x más rápido
                "do_sample": False,  # Determinístico
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,  # Evita repeticiones
            }
        
        elif modo_velocidad == "⚖️ Balanceado":
            # MODO BALANCEADO: Balance velocidad/calidad
            gen_config = {
                "max_length": max_len,
                "min_length": min_len,
                "num_beams": 3,  # ⚖️ Beam search limitado
                "do_sample": False,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
            }
        
        else:  # "🎯 Preciso (Más lento)"
            # MODO PRECISO: Máxima calidad
            gen_config = {
                "max_length": max_len,
                "min_length": min_len,
                "num_beams": 5,  # 🎯 Beam search completo
                "do_sample": False,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.2,
            }
        
        # 🚀 GENERAR TRADUCCIÓN
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=spanish_token_id,
                decoder_start_token_id=spanish_token_id,
                **gen_config  # Aplicar configuración elegida
            )
        
        # Decodificar
        tokenizer.tgt_lang = "es_XX"
        traduccion_parrafo = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        traduccion_parrafo = limpiar_texto(traduccion_parrafo)
        
        if not traduccion_parrafo or len(traduccion_parrafo) < 2:
            traduccion_parrafo = f"⚠️ Error al traducir"
        
        traducciones_finales.append(
            f"🌐 {nombre_idioma} → Español\n{traduccion_parrafo}"
        )
    
    # Calcular tiempo
    tiempo_total = time.time() - inicio
    
    resultado = "\n\n" + "─" * 60 + "\n\n".join(traducciones_finales)
    resultado += f"\n\n{'─' * 60}\n⏱️ Tiempo: {tiempo_total:.2f}s | Modo: {modo_velocidad}"
    
    return resultado

# -------------------------------------------------------------
# Interfaz de Gradio
# -------------------------------------------------------------
custom_css = """
#main_container {
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as iface:
    with gr.Column(elem_id="main_container"):
        gr.Markdown(
            "# 🌍 Traductor IA Ultrarrápido → Español\n"
            "### Traducción inteligente con detección automática de idioma",
            elem_classes="header"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    lines=10,
                    label="📥 Texto Original (cualquier idioma)",
                    placeholder="Escribe o pega aquí...\n\nEjemplo: Hello my name is Alan",
                    max_lines=20
                )
                
                modo_radio = gr.Radio(
                    choices=[
                        "⚡ Rápido (Recomendado)",
                        "⚖️ Balanceado", 
                        "🎯 Preciso (Más lento)"
                    ],
                    value="⚡ Rápido (Recomendado)",
                    label="⚙️ Modo de Traducción",
                    info="⚡ Rápido = 1-3s | ⚖️ Balanceado = 3-8s | 🎯 Preciso = 10-20s"
                )
                
                with gr.Row():
                    translate_button = gr.Button(
                        "🚀 Traducir Ahora", 
                        variant="primary", 
                        size="lg"
                    )
                    clear_button = gr.Button("🗑️ Limpiar", variant="secondary")
                
            with gr.Column(scale=1):
                output_textbox = gr.Textbox(
                    lines=10,
                    label="📤 Traducción en Español",
                    interactive=False,
                    show_copy_button=True,
                    max_lines=20
                )
        
        with gr.Accordion("📋 Ejemplos de Prueba", open=False):
            gr.Examples(
                examples=[
                    ["Hello my name is Alan", "⚡ Rápido (Recomendado)"],
                    ["Good morning! How are you today?", "⚡ Rápido (Recomendado)"],
                    ["Guten Tag! Ich lerne Deutsch und es macht mir viel Spaß.", "⚖️ Balanceado"],
                    ["Bonjour! Comment allez-vous? J'espère que vous passez une excellente journée.", "⚖️ Balanceado"],
                    ["The quick brown fox jumps over the lazy dog. This is a test sentence to check translation speed.", "⚡ Rápido (Recomendado)"],
                    ["人工智能正在改变我们的世界。这项技术将在未来发挥重要作用。", "🎯 Preciso (Más lento)"]
                ],
                inputs=[input_textbox, modo_radio],
                outputs=output_textbox,
                fn=traducir_en_es,
                cache_examples=False
            )
        
        gr.Markdown(
            """
            <div style='background: white; padding: 15px; border-radius: 8px; margin-top: 20px;'>
            
            ### 💡 Guía de Modos de Traducción:
            
            | Modo | Velocidad | Calidad | Mejor Para |
            |------|-----------|---------|------------|
            | ⚡ **Rápido** | 1-3 seg | ⭐⭐⭐⭐ | Frases cortas, chats, uso diario |
            | ⚖️ **Balanceado** | 3-8 seg | ⭐⭐⭐⭐⭐ | Párrafos, emails, textos importantes |
            | 🎯 **Preciso** | 10-20 seg | ⭐⭐⭐⭐⭐+ | Documentos formales, traducciones profesionales |
            
            **Recomendación:** Usa el modo **⚡ Rápido** para el 90% de tus traducciones.
            
            </div>
            """,
            elem_classes="info-box"
        )
        
        gr.Markdown(
            "<p style='text-align:center; color: white; margin-top: 15px;'>"
            "🤖 Traductor IA v3.0 | Optimizado para velocidad | 50+ idiomas soportados</p>"
        )

    # Event handlers
    translate_button.click(
        fn=traducir_en_es,
        inputs=[input_textbox, modo_radio],
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
    print("\n" + "="*70)
    print("🚀 TRADUCTOR IA ULTRARRÁPIDO - Iniciando...")
    print("="*70)
    print(f"💻 Dispositivo: {str(device).upper()}")
    print(f"📱 URL Local: http://localhost:8000")
    print(f"🌐 URL Red: http://0.0.0.0:8000")
    print("="*70)
    print("\n⚡ MODOS DISPONIBLES:")
    print("  • Rápido (1-3s) - Generación Greedy")
    print("  • Balanceado (3-8s) - Beam Search x3")
    print("  • Preciso (10-20s) - Beam Search x5")
    print("\n✨ Consejo: Usa el modo RÁPIDO para textos cortos\n")
    print("="*70 + "\n")
    
    iface.launch(
        server_name="0.0.0.0", 
        server_port=8000,
        share=False
    )