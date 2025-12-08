# Script para cargar y analizar el dataset
import pandas as pd
import os

# --- Función para cargar el dataset ---
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

<<<<<<< Updated upstream
#FUNCION PARWA CARGAR EL DATASET
def load_dataset():
    # cambiar con la ruta donde tenemos el csv
    path = r"Twitter_Data.csv"
=======
    print(f"Cargando dataset desde: {path}")
>>>>>>> Stashed changes

    try:
        df = pd.read_csv(path, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", header=None)

    # Asignamos nombres de columnas manualmente
    df.columns = ["id", "topic", "sentiment", "text"]

    return df


# --- Función de análisis ---
def analyze_dataset(df):
    print("\n===== COLUMNAS =====")
    print(df.columns.tolist())

    print("\n===== CLASES DE SENTIMIENTO =====")
    print(df['sentiment'].value_counts(dropna=False))
    print("\nProporción:")
    print(df['sentiment'].value_counts(normalize=True, dropna=False))

    print("\n===== LONGITUD DE TEXTO =====")
    df['text_len'] = df['text'].astype(str).apply(len)
    print(df['text_len'].describe())

    print("\n===== EJEMPLOS =====")
    print(df.sample(5))

    n_clases = df['sentiment'].nunique()
    print(f"\nNúmero de clases: {n_clases}")
    print("Multiclass" if n_clases > 2 else "Binario")

    print("\n===== DISTRIBUCIÓN =====")
    print(df['sentiment'].value_counts(normalize=True))


# --- Ejecución principal ---
if __name__ == "__main__":
    path = r"C:\Users\simpl\Desktop\UNIVERSIDAD\cuarto\Aprenentatge Computacional (1er Semestre)\AC Projecte\2526\twitter_training.csv\twitter_training.csv"
    df = load_dataset(path)
    analyze_dataset(df)




