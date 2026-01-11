import tkinter as tk
from tkinter import messagebox
import joblib
import os
import sys

# ==============================================================================
# 1. CONFIGURACIÓN ROBUSTA DE RUTAS (Para que funcione limpieza y models)
# ==============================================================================
# Detectamos dónde está este archivo y añadimos la carpeta superior al sistema
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)                
sys.path.append(parent_dir)

import limpieza  # Ahora sí importará sin errores

# Ruta del modelo (ajustada para que funcione siempre)
model_path = os.path.join(parent_dir, '..', 'models', 'modelo_final_balanced.pkl')

print("⏳ Cargando modelo inteligente...")
try:
    pipeline = joblib.load(model_path)
    print("✅ ¡Modelo cargado correctamente!")
except FileNotFoundError:
    try:
        # Intento alternativo por si la ruta varia
        model_path = 'models/modelo_final_balanced.pkl'
        pipeline = joblib.load(model_path)
    except:
        messagebox.showerror("Error Crítico", f"No encuentro el modelo en:\n{model_path}")
        sys.exit()

# ==============================================================================
# 2. LÓGICA DE PREDICCIÓN
# ==============================================================================
def analizar_sentimiento(event=None): # 'event' permite usar la tecla Enter
    texto_usuario = entry_texto.get()
    
    if not texto_usuario.strip():
        lbl_resultado.config(text="⚠️ Escribe algo...", fg="#e67e22") # Naranja aviso
        return

    # Limpieza y Predicción
    texto_limpio = limpieza.clean_text(texto_usuario)
    prediccion_num = pipeline.predict([texto_limpio])[0]
    probs = pipeline.predict_proba([texto_limpio])[0]
    
    # Configuración de Resultados (Textos y Colores Modernos)
    etiquetas = {0: "NEGATIVO 😡", 1: "NEUTRAL 😐", 2: "POSITIVO 😄"}
    # Colores más pasteles y modernos
    colores = {0: "#c0392b", 1: "#7f8c8d", 2: "#27ae60"} 
    
    resultado_texto = etiquetas[prediccion_num]
    confianza = probs[prediccion_num] * 100
    
    # Actualizar Interfaz
    lbl_resultado.config(text=resultado_texto, fg=colores[prediccion_num])
    lbl_confianza.config(text=f"Confianza : {confianza:.1f}%")

# ==============================================================================
# 3. INTERFAZ GRÁFICA
# ==============================================================================
root = tk.Tk()
root.title("🤖 AC-13 Sentiment Analyzer")
root.geometry("600x450")
root.configure(bg="#dce4f2") # Fondo de ventana (Azul Grisáceo Suave)

# --- 1. Marco Central (Estilo Tarjeta Blanca) ---
# Esto crea el efecto de "caja flotante"
card = tk.Frame(root, bg="white", bd=0)
card.place(relx=0.5, rely=0.5, anchor="center", width=450, height=350)

# --- 2. Título (Negro y Elegante) ---
lbl_titulo = tk.Label(card, text="Detector de Sentimientos", 
                      font=("Segoe UI", 20, "bold"), # Fuente moderna de Windows
                      bg="white", fg="black")
lbl_titulo.pack(pady=(30, 5)) # Padding arriba y abajo



# --- 3. Caja de Texto (Estilizada) ---
frame_entry = tk.Frame(card, bg="white")
frame_entry.pack(pady=10)

# Borde decorativo para la caja de texto
lbl_instruccion = tk.Label(frame_entry, text="Tu frase:", bg="white", fg="#2c3e50", font=("Segoe UI", 10, "bold"))
lbl_instruccion.pack(anchor="w", padx=5)

entry_texto = tk.Entry(frame_entry, width=35, font=("Segoe UI", 13), 
                       bd=2, relief="groove", justify="center")
entry_texto.pack(ipady=5)
entry_texto.focus()

# --- 4. Botón Moderno ---
# Función para cambiar color al pasar el mouse (Hover Effect)
def on_enter(e):
    btn_analizar['bg'] = '#5a4fcf' # Color más oscuro al tocar
def on_leave(e):
    btn_analizar['bg'] = '#6c5ce7' # Color original

btn_analizar = tk.Button(card, text="ANALIZAR", command=analizar_sentimiento,
                         font=("Segoe UI", 11, "bold"), 
                         bg="#6c5ce7", fg="white", # Morado moderno
                         activebackground="#5a4fcf", activeforeground="white",
                         bd=0, cursor="hand2", padx=20, pady=5)
btn_analizar.pack(pady=15)

# Vincular eventos de mouse para efecto hover
btn_analizar.bind("<Enter>", on_enter)
btn_analizar.bind("<Leave>", on_leave)

# --- 5. Resultados ---
lbl_resultado = tk.Label(card, text="...", font=("Segoe UI", 24, "bold"), bg="white", fg="#dfe6e9")
lbl_resultado.pack(pady=(5, 0))

lbl_confianza = tk.Label(card, text="", font=("Segoe UI", 10), bg="white", fg="#95a5a6")
lbl_confianza.pack(pady=5)

# Permitir Enter para enviar
root.bind('<Return>', analizar_sentimiento)

# Pie de página (Créditos)
lbl_footer = tk.Label(root, text="Proyecto AC-13 ", font=("Arial", 8), bg="#dce4f2", fg="#7f8c8d")
lbl_footer.pack(side="bottom", pady=10)

root.mainloop()