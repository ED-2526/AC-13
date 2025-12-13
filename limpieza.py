import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios si no están (por seguridad)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Función para limpiar una sola frase.
    Se ha renombrado de '_clean_text_logic' a 'clean_text' 
    para poder usarla desde otros scripts.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text) # URLs
    text = re.sub(r'\@\w+|\#', '', text)       # Menciones
    text = re.sub(r'[^a-zA-Z]', ' ', text)     # Solo letras
    
    tokens = text.split()
    # Lematización y Stopwords
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(filtered)

def load_and_clean_data(filepath, balance=None):
    """
    Carga el CSV, elimina irrelevantes, mapea etiquetas y limpia el texto.
    Retorna: DataFrame limpio ready para entrenar.
    """
    
    print("--- CARGANDO Y LIMPIANDO DATOS ---")
    
    # 1. Cargar
    col_names = ['id', 'entity', 'sentiment', 'text']
    # header=None porque el dataset original no suele traer cabecera
    df = pd.read_csv(filepath, names=col_names, header=None)
    
    # 2. Filtrar nulos e Irrelevantes
    df = df.dropna(subset=['text'])
    df = df[df['sentiment'] != 'Irrelevant']
    
    # 3. Mapeo a 0, 1, 2
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label'] = df['sentiment'].map(label_map)
    
    if balance:
        # 4. Balanceo (Recortar dataset)
        counts = df['label'].value_counts()
        min_count = counts.min() # El número de la clase más pequeña
        
        print(f"Recortando clases para balancear. Mínimo encontrado: {min_count} muestras por clase.")
        
        # Cogemos aleatoriamente 'min_count' filas de cada clase
        df_0 = df[df['label'] == 0].sample(n=min_count, random_state=42)
        df_1 = df[df['label'] == 1].sample(n=min_count, random_state=42)
        df_2 = df[df['label'] == 2].sample(n=min_count, random_state=42)
        
        # Juntamos y mezclamos (shuffle) para que no queden ordenados
        df = pd.concat([df_0, df_1, df_2], axis=0)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset balanceado: {len(df)} filas en total ({min_count} por clase).")

    # 5. Limpieza de texto (Usamos la función renombrada clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    
    print(f"Datos procesados: {len(df)} filas.")
    return df

if __name__ == "__main__":
    
    # Si lo importas desde otro archivo, esto se ignora (CORRECTO)
    
    filepath = 'twitter_training.csv'

    print("--- PRUEBA DE LIMPIEZA ---")
    df_1 = load_and_clean_data(filepath, balance=False) # Prueba sin balancear
    muestras = df_1.sample(5)
    for index, row in muestras.iterrows():
        print("ORIGINAL: ", row['text'])
        print("LIMPIO:   ", row['text_clean'])
        print("-" * 80) 
    

    print("\n--- GENERANDO GRÁFICA DE BALANCEO ---")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')

    # Cargamos BALANCEADO para ver la gráfica bonita
    df = load_and_clean_data(filepath, balance=True) 

    plt.figure(figsize=(8, 6))
    sns.countplot(
        data=df, 
        x='sentiment', 
        order=['Negative', 'Neutral', 'Positive'], 
        palette='viridis',     
        edgecolor='black'      
    )

    plt.title('Distribución de Clases (Tras el Balanceo)', fontsize=14)
    plt.xlabel('Sentimiento', fontsize=12)
    plt.ylabel('Cantidad de Tweets', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5) 

    filename = "imagenes/eda_1.2_distribucion_clases_balanceada.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 

    print(f"-> Gráfica guardada correctamente en: {filename}")