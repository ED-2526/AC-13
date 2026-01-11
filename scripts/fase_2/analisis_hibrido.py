import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import os
import limpieza # Tu script de limpieza

def main():
    print("--- 📂 ANÁLISIS MASTER: HÍBRIDO + CARPETA ORGANIZADA 📂 ---")
    
    # ==============================================================================
    # 0. CREAR CARPETA DE SALIDA
    # ==============================================================================
    folder_name = "results\Fase_2_hibrido\hibrido_analisis"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"\n[0/6] ✅ Carpeta creada: '{folder_name}/'")
    else:
        print(f"\n[0/6] ℹ️ La carpeta '{folder_name}/' ya existe. Guardando ahí.")

    # ==============================================================================
    # 1. CARGA Y PREPARACIÓN
    # ==============================================================================
    print("\n[1/6] Cargando y fusionando datasets...")
    try:
        # A) ORIGINAL (Sin Irrelevant)
        df_orig_raw = pd.read_csv('data/twitter_training.csv', header=None, names=['id', 'entity', 'sentiment', 'text'])
        df_orig = df_orig_raw[df_orig_raw['sentiment'] != 'Irrelevant'].copy()
        df_orig['label'] = df_orig['sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
        df_orig['text_clean'] = df_orig['text'].astype(str).apply(limpieza.clean_text)
        df_orig['Origen'] = 'Juegos (Original)'
        
        # B) AEROLÍNEAS
        df_air = pd.read_csv('data/Tweets_aerolinea.csv')
        df_air['label'] = df_air['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air['Origen'] = 'Aerolíneas'
        
        # C) VIDA COTIDIANA
        df_life = pd.read_json('data/validation.json')
        df_life['label'] = df_life['label'].str.lower().map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life['Origen'] = 'Vida Cotidiana'

        # FUSIÓN
        df_total = pd.concat([
            df_orig[['text_clean', 'label', 'Origen']], 
            df_air[['text_clean', 'label', 'Origen']].dropna(), 
            df_life[['text_clean', 'label', 'Origen']].dropna()
        ], ignore_index=True)
        
        print(f"   ✅ Dataset Híbrido Total: {len(df_total)} tweets.")

    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    # ==============================================================================
    # 2. DISTRIBUCIONES BÁSICAS
    # ==============================================================================
    print("\n[2/6] Generando distribuciones...")

    # 2.1 Distribución Global con Porcentajes
    plt.figure(figsize=(10, 6))
    colores = ['#e74c3c', '#95a5a6', '#2ecc71'] # Rojo, Gris, Verde
    ax = sns.countplot(x='label', data=df_total, palette=colores)
    plt.title('Distribución Global de Sentimientos', fontsize=14, fontweight='bold')
    plt.xticks([0, 1, 2], ['Negativo', 'Neutral', 'Positivo'])
    plt.xlabel('Sentimiento')
    plt.ylabel('Cantidad')
    
    total = len(df_total)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = f'{100 * count / total:.1f}%'
        ax.annotate(f'{count}\n({percentage})', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '1_distribucion_global.png'))

    # 2.2 Desglose por Origen
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_total, x='Origen', hue='label', palette='viridis')
    plt.title('Sentimientos por Fuente de Datos')
    plt.legend(title='Sentimiento', labels=['Negativo', 'Neutral', 'Positivo'])
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '2_desglose_origen.png'))

    # ==============================================================================
    # 3. ANÁLISIS AVANZADO: VIOLIN PLOT (Longitud vs Sentimiento)
    # ==============================================================================
    print("\n[3/6] Analizando longitud vs. sentimiento...")
    # Calculamos número de palabras
    df_total['num_words'] = df_total['text_clean'].apply(lambda x: len(str(x).split()))
    
    # Eliminamos outliers extremos para que la gráfica se vea bien (tweets > 60 palabras)
    df_clean_len = df_total[df_total['num_words'] < 60]

    plt.figure(figsize=(10, 6))
    # El Violin Plot muestra la densidad (si es gordo, hay muchos datos ahí)
    sns.violinplot(data=df_clean_len, x='label', y='num_words', palette=colores)
    plt.title('Densidad de Longitud del Tweet por Sentimiento', fontsize=14)
    plt.xlabel('Sentimiento')
    plt.ylabel('Número de Palabras')
    plt.xticks([0, 1, 2], ['Negativo', 'Neutral', 'Positivo'])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '3_violin_longitud.png'))
    print("   -> (NUEVO) 🎻 Violin plot guardado. Muestra si los enfadados escriben más.")

    # ==============================================================================
    # 4. ANÁLISIS CIENTÍFICO: COBERTURA DE VOCABULARIO (Justificación TF-IDF)
    # ==============================================================================
    print("\n[4/6] Generando curva de cobertura de vocabulario...")
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_total['text_clean'])
    word_counts = np.array(X.sum(axis=0)).flatten()
    # Ordenamos de mayor a menor frecuencia
    sorted_counts = np.sort(word_counts)[::-1]
    cumulative_counts = np.cumsum(sorted_counts)
    total_counts = cumulative_counts[-1]
    cumulative_percentage = cumulative_counts / total_counts

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_percentage, color='blue', linewidth=2)
    
    # Dibujamos línea en 5000 (tu max_features)
    limit = 8000
    coverage = cumulative_percentage[limit] if limit < len(cumulative_percentage) else 1.0
    
    plt.axvline(x=limit, color='red', linestyle='--', label=f'Corte en 8,000 palabras')
    plt.axhline(y=coverage, color='green', linestyle=':', label=f'Cobertura: {coverage:.1%}')
    
    plt.title('Curva de Cobertura del Vocabulario (Ley de Zipf)')
    plt.xlabel('Número de Palabras (Ranking Frecuencia)')
    plt.ylabel('Porcentaje del Texto Total Cubierto')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20000) # Hacemos zoom en las primeras 20k palabras
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '4_cobertura_vocabulario.png'))
    print(f"   -> (NUEVO) 📈 Curva guardada. Demuestra que 8000 palabras cubren el {coverage:.1%} del texto.")

    # ==============================================================================
    # 5. BIGRAMAS POR SENTIMIENTO (Contexto)
    # ==============================================================================
    print("\n[5/6] Analizando Bigramas (Parejas de palabras)...")
    
    def get_top_bigrams(corpus, n=10):
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    # Bigramas Negativos
    neg_text = df_total[df_total['label'] == 0]['text_clean']
    top_neg = get_top_bigrams(neg_text)
    
    # Bigramas Positivos
    pos_text = df_total[df_total['label'] == 2]['text_clean']
    top_pos = get_top_bigrams(pos_text)

    # Plot Doble
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Negativo
    x_neg, y_neg = zip(*top_neg)
    sns.barplot(x=list(y_neg), y=list(x_neg), ax=axes[0], palette='Reds_r')
    axes[0].set_title('Top 10 Bigramas NEGATIVOS')
    
    # Positivo
    x_pos, y_pos = zip(*top_pos)
    sns.barplot(x=list(y_pos), y=list(x_pos), ax=axes[1], palette='Greens_r')
    axes[1].set_title('Top 10 Bigramas POSITIVOS')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '5_top_bigramas.png'))

    # ==============================================================================
    # 6. WORDCLOUDS (Estética)
    # ==============================================================================
    print("\n[6/6] Generando Nubes de Palabras...")
    
    # Nube Negativa
    wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(" ".join(neg_text))
    wc_neg.to_file(os.path.join(folder_name, '6_cloud_negativo.png'))
    
    # Nube Positiva
    wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(" ".join(pos_text))
    wc_pos.to_file(os.path.join(folder_name, '6_cloud_positivo.png'))

    print(f"\n✅ ¡PROCESO TERMINADO! Revisa la carpeta '{folder_name}' 📂")

if __name__ == "__main__":
    main()