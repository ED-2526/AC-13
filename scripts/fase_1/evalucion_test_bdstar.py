import pandas as pd
import scripts.fase_2.limpieza as limpieza
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    print("--- EVALUACIÓN FINAL CON JSON PROPIO (CORREGIDO) ---")
    
    # 1. CARGAR DATOS
    archivo = 'validation.json'
    print(f"-> Leyendo {archivo}...")
    
    try:
        df = pd.read_json(archivo)
    except ValueError:
        print("ERROR: El JSON no tiene el formato correcto. Asegúrate de que empieza con '[' y termina con ']'.")
        return

    # 2. MAPEO DE ETIQUETAS A NÚMEROS (CRUCIAL)
    # Tu modelo fue entrenado con 0, 1, 2. Debemos convertir el texto del JSON a eso.
    # El JSON tiene: "neutral", "negative", "positive"
    map_to_numbers = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
        'Negative': 0, # Por si acaso viene en mayúscula
        'Neutral': 1,
        'Positive': 2
    }
    
    print("-> Convirtiendo etiquetas de texto a números (0, 1, 2)...")
    # Convertimos a minúsculas primero para asegurar que coincida, luego mapeamos
    df['label_num'] = df['label'].str.lower().map(map_to_numbers)
    
    # Eliminamos si alguno no se pudo mapear
    df = df.dropna(subset=['label_num'])
    df['label_num'] = df['label_num'].astype(int)

    # 3. LIMPIEZA DE TEXTO
    print("-> Limpiando los textos...")
    # Usamos tu función limpieza.clean_text (que ya arreglamos antes)
    df['text_clean'] = df['text'].astype(str).apply(limpieza.clean_text)
    
    # 4. CARGAR TU MEJOR MODELO
    ruta_modelo = 'resultados_rf_250_depth/modelo_rf_250_depth.pkl'
    print(f"-> Cargando modelo: {ruta_modelo}")
    try:
        model = joblib.load(ruta_modelo)
    except FileNotFoundError:
        print("ERROR: No encuentro el modelo .pkl. Verifica la ruta.")
        return

    # 5. PREDICCIÓN
    print("-> Realizando predicciones...")
    y_pred = model.predict(df['text_clean']) # Esto devuelve números [0, 1, 2...]
    y_true = df['label_num']                 # Esto ahora son números [0, 1, 2...]

    # 6. RESULTADOS
    acc = accuracy_score(y_true, y_pred)
    print(f"\n" + "="*40)
    print(f"   ACCURACY FINAL: {acc:.4f}")
    print(f"="*40 + "\n")
    
    # Nombres para que el reporte sea legible
    target_names = ['Negative', 'Neutral', 'Positive']
    
    print("Reporte de Clasificación:")
    # Ahora sí comparamos peras con peras (números con números)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 7. MATRIZ DE CONFUSIÓN
    print("-> Generando matriz...")
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title('Matriz de Confusión: Custom JSON')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('matriz_custom_json.png')
    print("-> Gráfica guardada: 'matriz_custom_json.png'")

if __name__ == "__main__":
    main()