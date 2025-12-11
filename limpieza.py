
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_and_clean_data(filepath):
    """
    Carga el CSV, elimina irrelevantes, mapea etiquetas y limpia el texto.
    Retorna: DataFrame limpio ready para entrenar.
    """
    print("--- CARGANDO Y LIMPIANDO DATOS ---")
    
    # 1. Cargar
    col_names = ['id', 'entity', 'sentiment', 'text']
    df = pd.read_csv(filepath, names=col_names, header=None)
    
    # 2. Filtrar nulos e Irrelevantes
    df = df.dropna(subset=['text'])
    df = df[df['sentiment'] != 'Irrelevant']
    
    # 3. Mapeo a 0, 1, 2
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label'] = df['sentiment'].map(label_map)
    
    # 4. Limpieza de texto (lema i stopwprd)
    df['text_clean'] = df['text'].apply(_clean_text_logic)
    
    print(f"Datos procesados: {len(df)} filas.")
    return df

def _clean_text_logic(text):
    """Función interna para limpiar una sola frase"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text) # URLs
    text = re.sub(r'\@\w+|\#', '', text)       # Menciones
    text = re.sub(r'[^a-zA-Z]', ' ', text)     # Solo letras
    
    tokens = text.split()
    # Lematización y Stopwords
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(filtered)

filepath = 'twitter_training.csv'

df_1 = load_and_clean_data(filepath)

muestras = df_1.sample(5)

for index, row in muestras.iterrows():
    print("ORIGINAL: ", row['text'])
    print("LIMPIO:   ", row['text_clean'])
    print("-" * 80) 