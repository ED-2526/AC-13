import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración estética
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("--- INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---")
    
    
    # no tiene cabecera
    col_names = ['id', 'entity', 'sentiment', 'text']
    df = pd.read_csv('twitter_training.csv', names=col_names, header=None)
    
    # Información básica
    print("\n[INFO GENERAL]")
    print(f"Total de filas: {len(df)}")
    print(f"Columnas: {df.columns.tolist()}")
    
    #DETECCIÓN DE NULOS
    print("\n[VALORES NULOS]")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0])
    
    # Limpiamos nulos 
    df = df.dropna(subset=['text'])
    print(f"Filas tras eliminar nulos: {len(df)}")

    # (BIAS)
    print("\n[DISTRIBUCIÓN DE CLASES]")
    class_counts = df['sentiment'].value_counts()
    print(class_counts)
    
    #  Barras de distribución
    plt.figure()
    sns.countplot(data=df, x='sentiment', order=class_counts.index, palette='viridis')
    plt.title('Distribución de Sentimientos')
    plt.ylabel('Cantidad de Tuits')
    plt.savefig('eda_1_distribucion_clases.png')
    print("-> Gráfica guardada: eda_1_distribucion_clases.png")

    #  ANÁLISIS DE LONGITUD DEL TEXTO
    # Creamos una columna nueva contando palabras
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    
    print("\n[ESTADÍSTICAS DE LONGITUD (PALABRAS)]")
    stats = df.groupby('sentiment')['word_count'].describe()
    print(stats[['count', 'mean', 'std', 'max']])
    
    #Boxplot de longitud por sentimiento
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='sentiment', y='word_count', palette='Set2')
    plt.title('Longitud de los Tuits por Sentimiento')
    plt.ylim(0, 100) # Limitamos eje Y para ver mejor (quitamos outliers extremos)
    plt.savefig('eda_2_boxplot_longitud.png')
    print("-> Gráfica guardada: eda_2_boxplot_longitud.png")
    
    #Histograma comparativo
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='word_count', hue='sentiment', kde=True, element="step")
    plt.title('Histograma de Palabras por Sentimiento')
    plt.xlim(0, 80)
    plt.savefig('eda_3_histograma_longitud.png')
    print("-> Gráfica guardada: eda_3_histograma_longitud.png")

    #  MUESTRA DE DATOS
    print("\n[EJEMPLOS DE TUITS]")
    print("--- Positivo ---")
    print(df[df['sentiment']=='Positive']['text'].iloc[0])
    print("\n--- Negativo ---")
    print(df[df['sentiment']=='Negative']['text'].iloc[0])
    print("\n--- Neutro ---")
    print(df[df['sentiment']=='Neutral']['text'].iloc[0])

if __name__ == "__main__":
    main()
