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
    model.eval()  # Poner el modelo en modo de evaluación
    print(f"✅ Modelo '{MODEL_NAME}' cargado exitosamente en {DEVICE}")
except Exception as e:
    print(f"🚨 Error al cargar el modelo: {e}")
    # Salir si el modelo no se puede cargar
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
        # Usar una muestra más grande para mayor precisión
        muestra = texto if len(texto) < 500 else texto[:500]
        lang_code_short = detect(muestra)
        src_lang_code = LANG_CODE_MAP.get(lang_code_short, "en_XX")
        nombre_idioma = LANG_NAME_MAP.get(lang_code_short, "Desconocido")
        return lang_code_short, src_lang_code, nombre_idioma
    except LangDetectException:
        # Si la detección falla, asumimos inglés como idioma por defecto
        return 'en', 'en_XX', 'Inglés (detectado por defecto)'

# -----------------------------
# 🚀 FUNCIÓN DE TRADUCCIÓN MEJORADA
# -----------------------------
def traducir_a_espanol(texto: str, modo_velocidad: str = "⚡ Rápido (Recomendado)") -> str:
    """
    Traduce texto de múltiples idiomas al español con 3 modos de velocidad optimizados.
    """
    if not texto or not texto.strip():
        return "⚠️ Por favor, ingresa un texto para traducir."
    
    inicio = time.time()
    texto_limpio = limpiar_texto(texto)
    
    if len(texto_limpio.split()) < 2:
        return "⚠️ El texto es demasiado corto. Intenta con una frase más larga."
    
    # Detección de idioma del texto completo para evitar procesar párrafo por párrafo
    lang_short, src_lang_code, nombre_idioma = detectar_idioma_robusto(texto_limpio)
    
    # Si el texto ya está en español, no hacemos nada
    if lang_short == 'es':
        return f"✅ El texto ya está en Español:\n\n---\n{texto}\n---"

    # --- CONFIGURACIÓN DE GENERACIÓN ---
    # Usamos la forma canónica para obtener el ID del token de idioma
    spanish_token_id = tokenizer.lang_code_to_id["es_XX"]
    
    # Tokenizar el texto de entrada
    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(
        texto_limpio, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024  # Aumentamos un poco por si acaso
    ).to(DEVICE)

    # --- AJUSTE DINÁMICO DE PARÁMETROS ---
    num_tokens = inputs.input_ids.shape[1]
    max_len = int(num_tokens * 2.0)  # Permitir que la traducción sea más larga
    min_len = int(num_tokens * 0.5)  # O más corta

    # Asegurar límites razonables
    max_len = min(max(max_len, 32), 1024)
    min_len = max(min_len, 10)

    # Parámetros base para todos los modos
    base_gen_config = {
        "max_length": max_len,
        "min_length": min_len,
        "early_stopping": True,
        "no_repeat_ngram_size": 3, # Evita la repetición de secuencias de 3 palabras
        "repetition_penalty": 1.5, # Penaliza palabras repetidas
    }

    # Configuración específica por modo
    if modo_velocidad == "⚡ Rápido (Recomendado)":
        # LA SOLUCIÓN CLAVE: Usar un beam search mínimo (2) en lugar de greedy (1).
        # Es casi igual de rápido pero mucho más robusto contra errores de repetición.
        gen_config = {"num_beams": 2, **base_gen_config}
    
    elif modo_velocidad == "⚖️ Balanceado":
        gen_config = {"num_beams": 4, "length_penalty": 1.0, **base_gen_config}
        
    else:  # "🎯 Preciso (Más lento)"
        gen_config = {"num_beams": 6, "length_penalty": 1.2, **base_gen_config}

    # 🚀 GENERAR TRADUCCIÓN
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=spanish_token_id, # Indica al modelo el idioma de salida
            **gen_config
        )
    
    # Decodificar el resultado
    traduccion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    # Post-procesamiento final
    traduccion_final = limpiar_texto(traduccion)

    if not traduccion_final:
        return "⚠️ Error: La traducción resultó en un texto vacío."

    tiempo_total = time.time() - inicio
    
    # Formatear el resultado final
    resultado = (
        f"🌐 {nombre_idioma} → Español\n\n"
        f"------------------------------------\n"
        f"{traduccion_final}\n"
        f"------------------------------------\n\n"
        f"⏱️ **Tiempo:** {tiempo_total:.2f}s | ⚙️ **Modo:** {modo_velocidad}"
    )
    
    return resultado


# -------------------------------------------------------------
# Interfaz de Gradio (sin cambios mayores, solo la función llamada)
# -------------------------------------------------------------
custom_css = """
#main_container {
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as iface:
    with gr.Column(elem_id="main_container"):
        gr.Markdown(
            "# 🌍 Traductor IA Ultrarrápido → Español\n"
            "### Traduce más de 50 idiomas al español con detección automática."
        )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_textbox = gr.Textbox(
                    lines=10,
                    label="📥 Texto Original (cualquier idioma)",
                    placeholder="Escribe o pega aquí...\n\nEjemplo: The quick brown fox jumps over the lazy dog.",
                )
                
                modo_radio = gr.Radio(
                    choices=[
                        "⚡ Rápido (Recomendado)",
                        "⚖️ Balanceado", 
                        "🎯 Preciso (Más lento)"
                    ],
                    value="⚡ Rápido (Recomendado)",
                    label="⚙️ Modo de Traducción",
                    info="Rápido es ideal para la mayoría de los casos."
                )
                
                with gr.Row():
                    translate_button = gr.Button("🚀 Traducir Ahora", variant="primary")
                    clear_button = gr.Button("🗑️ Limpiar", variant="secondary")
                
            with gr.Column(scale=1):
                output_textbox = gr.Textbox(
                    lines=12,
                    label="📤 Traducción en Español",
                    interactive=False,
                    show_copy_button=True,
                )
        
        with gr.Accordion("📋 Ejemplos para Probar", open=False):
            gr.Examples(
                examples=[
                    ["Hello my name is Alan.", "⚡ Rápido (Recomendado)"],
                    ["Good morning! How are you today?", "⚡ Rápido (Recomendado)"],
                    ["Guten Tag! Ich lerne Deutsch und es macht mir viel Spaß.", "⚖️ Balanceado"],
                    ["L'intelligence artificielle transforme notre monde à une vitesse incroyable.", "⚖️ Balanceado"],
                    ["人工智能正在改变我们的世界。这项技术将在未来发挥重要作用。", "🎯 Preciso (Más lento)"]
                ],
                inputs=[input_textbox, modo_radio],
                outputs=output_textbox,
                fn=traducir_a_espanol,
                cache_examples=False
            )
        
        gr.Markdown(
            """
            <div style='padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #E0E0E0;'>
            
            ### 💡 Guía de Modos:
            | Modo | Velocidad | Calidad | Ideal Para |
            |:---:|:---:|:---:|:---|
            | ⚡ **Rápido** | ~1-3 seg | ⭐⭐⭐⭐ | Frases, chats, uso general. |
            | ⚖️ **Balanceado** | ~3-8 seg | ⭐⭐⭐⭐⭐ | Párrafos, emails, textos importantes. |
            | 🎯 **Preciso** | ~10-20 seg | ⭐⭐⭐⭐⭐+ | Documentos, máxima fidelidad. |
            
            </div>
            """
        )

    translate_button.click(
        fn=traducir_a_espanol,
        inputs=[input_textbox, modo_radio],
        outputs=output_textbox,
        api_name="translate"
    )
    
    clear_button.click(lambda: ("", ""), inputs=None, outputs=[input_textbox, output_textbox])

# -----------------------------
# Lanzar la aplicación
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LANZANDO LA APLICACIÓN DE TRADUCCIÓN IA")
    print("="*70 + "\n")
    iface.launch(server_name="0.0.0.0", server_port=8000)