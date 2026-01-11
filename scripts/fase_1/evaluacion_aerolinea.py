import pandas as pd
import scripts.fase_2.limpieza as limpieza
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    print("--- EVALUACIÓN FINAL: DATASET AEROLÍNEAS ---")
    
    archivo = 'Tweets_aerolinea.csv'
    print(f"-> Leyendo {archivo}...")
    
    try:
        # Cargamos el CSV. Este dataset tiene cabeceras, así que no hace falta 'header=None'
        df = pd.read_csv(archivo)
    except FileNotFoundError:
        print(f"ERROR: No encuentro '{archivo}'. Descárgalo de Kaggle y ponlo en la carpeta.")
        return

    # 1. MAPEO DE ETIQUETAS
    # Este dataset usa: "negative", "neutral", "positive"
    # Tu modelo usa números: 0, 1, 2
    map_to_numbers = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    print("-> Traduciendo etiquetas...")
    # La columna en este dataset se llama 'airline_sentiment'
    df['label_num'] = df['airline_sentiment'].map(map_to_numbers)
    
    # 2. LIMPIEZA
    print("-> Limpiando textos (esto tarda un poco)...")
    # La columna de texto se llama 'text'
    df['text_clean'] = df['text'].astype(str).apply(limpieza.clean_text)
    
    # 3. CARGAR MODELO
    ruta_modelo = 'resultados_rf_250_depth/modelo_rf_250_depth.pkl'
    print(f"-> Cargando modelo: {ruta_modelo}")
    try:
        model = joblib.load(ruta_modelo)
    except:
        print("ERROR: No encuentro el modelo.")
        return

    # 4. PREDICCIÓN
    print("-> Realizando predicciones...")
    y_pred = model.predict(df['text_clean'])
    y_true = df['label_num']

    # 5. RESULTADOS
    acc = accuracy_score(y_true, y_pred)
    print(f"\n" + "="*40)
    print(f"   ACCURACY (AEROLÍNEAS): {acc:.4f}")
    print(f"="*40 + "\n")
    
    target_names = ['Negative', 'Neutral', 'Positive']
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 6. MATRIZ
    print("-> Generando matriz...")
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión: Aerolíneas')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('matriz_aerolineas.png')
    print("-> Gráfica guardada: 'matriz_aerolineas.png'")

if __name__ == "__main__":
    main()