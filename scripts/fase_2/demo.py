import tkinter as tk
from tkinter import messagebox
import joblib
import os
import sys

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS (TRUCO PARA ENCONTRAR LIMPIEZA Y EL MODELO)
# ==============================================================================
# Obtenemos la ruta de donde está este archivo
current_dir = os.path.dirname(os.path.abspath(__file__)) # carpeta fase_2
parent_dir = os.path.dirname(current_dir)                # carpeta scripts
sys.path.append(parent_dir)                              # Añadimos scripts al path

# Ahora podemos importar limpieza sin error
import limpieza

# Ruta del modelo (subimos 2 niveles hasta la raíz y entramos en models)
model_path = os.path.join(parent_dir, '..', 'models', 'modelo_final_balanced.pkl')

# ==============================================================================
# 2. CARGA DEL MODELO
# ==============================================================================
print("⏳ Cargando modelo inteligente...")
try:
    pipeline = joblib.load(model_path)
    print("✅ ¡Modelo cargado correctamente!")
except FileNotFoundError:
    messagebox.showerror("Error", f"No encuentro el modelo en:\n{model_path}\n\n¿Seguro que ejecutaste el entrenamiento?")
    sys.exit()

# ==============================================================================
# 3. LÓGICA DE PREDICCIÓN
# ==============================================================================
def analizar_sentimiento():
    # 1. Coger el texto de la caja
    texto_usuario = entry_texto.get()
    
    if not texto_usuario.strip():
        lbl_resultado.config(text="⚠️ Escribe algo primero...", fg="orange")
        return

    # 2. Limpiar el texto (IGUAL QUE EN EL ENTRENAMIENTO)
    texto_limpio = limpieza.clean_text(texto_usuario)
    
    # 3. Predecir
    prediccion_num = pipeline.predict([texto_limpio])[0]
    probs = pipeline.predict_proba([texto_limpio])[0]
    
    # 4. Traducir número a texto y color
    etiquetas = {0: "NEGATIVO 😡", 1: "NEUTRAL 😐", 2: "POSITIVO 😄"}
    colores = {0: "#e74c3c", 1: "#95a5a6", 2: "#2ecc71"} # Rojo, Gris, Verde
    
    resultado_texto = etiquetas[prediccion_num]
    confianza = probs[prediccion_num] * 100
    
    # 5. Actualizar la interfaz
    lbl_resultado.config(text=f"{resultado_texto}", fg=colores[prediccion_num])
    lbl_confianza.config(text=f"Confianza del modelo: {confianza:.1f}%")

# ==============================================================================
# 4. INTERFAZ GRÁFICA (LA VENTANITA)
# ==============================================================================
root = tk.Tk()
root.title("🤖 Detector de Sentimientos IA (AC-13)")
root.geometry("500x350")
root.configure(bg="#f0f2f5")

# Título
lbl_titulo = tk.Label(root, text="Analizador de Sentimientos", font=("Helvetica", 18, "bold"), bg="#f0f2f5", fg="#333")
lbl_titulo.pack(pady=20)

# Caja de texto
lbl_instruccion = tk.Label(root, text="Escribe una frase (Juegos, Aerolíneas, Vida...):", bg="#f0f2f5", font=("Arial", 10))
lbl_instruccion.pack()

entry_texto = tk.Entry(root, width=40, font=("Arial", 12))
entry_texto.pack(pady=10, ipady=5)
entry_texto.focus() # Poner el cursor ahí directamente

# Botón
btn_analizar = tk.Button(root, text="✨ ANALIZAR ✨", command=analizar_sentimiento, 
                         font=("Arial", 11, "bold"), bg="#3498db", fg="white", cursor="hand2")
btn_analizar.pack(pady=10, ipadx=10, ipady=5)

# Resultado (Grande)
lbl_resultado = tk.Label(root, text="Esperando...", font=("Helvetica", 24, "bold"), bg="#f0f2f5", fg="#bdc3c7")
lbl_resultado.pack(pady=20)

# Confianza (Pequeño)
lbl_confianza = tk.Label(root, text="", font=("Arial", 9), bg="#f0f2f5", fg="#7f8c8d")
lbl_confianza.pack()

# Permitir pulsar "Enter" para enviar
root.bind('<Return>', lambda event: analizar_sentimiento())

# Arrancar la app
root.mainloop()