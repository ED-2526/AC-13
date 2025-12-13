import pandas as pd
import limpieza
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def balancear_dataset(df, nombre):
    """
    Toma un dataframe y devuelve 3 sub-dataframes (Neg, Neu, Pos)
    y el conteo de cada uno.
    """
    df_neg = df[df['label'] == 0]
    df_neu = df[df['label'] == 1]
    df_pos = df[df['label'] == 2]
    
    counts = {
        'neg': len(df_neg),
        'neu': len(df_neu),
        'pos': len(df_pos)
    }
    print(f"   -> {nombre}: Neg={counts['neg']}, Neu={counts['neu']}, Pos={counts['pos']}")
    
    return df_neg, df_neu, df_pos, counts

def main():
    print("--- ⚖️ CREANDO DATASET HÍBRIDO BALANCEADO + GRÁFICA ⚖️ ---")

    # ==============================================================================
    # 1. CARGA DE DATOS
    # ==============================================================================
    
    # A) ORIGINAL
    print("\n[1/5] Cargando Original...")
    try:
        df_orig = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
        df_orig = df_orig[['text_clean', 'label']]
        neg_orig, neu_orig, pos_orig, c_orig = balancear_dataset(df_orig, "Original")
    except Exception as e:
        print(f"Error cargando Original: {e}")
        return

    # B) AEROLÍNEAS
    print("\n[2/5] Cargando Aerolíneas...")
    try:
        df_air = pd.read_csv('Tweets_aerolinea.csv')
        map_air = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_air['label'] = df_air['airline_sentiment'].map(map_air)
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air = df_air[['text_clean', 'label']].dropna()
        neg_air, neu_air, pos_air, c_air = balancear_dataset(df_air, "Aerolíneas")
    except Exception as e:
        print(f"Error cargando Aerolíneas: {e}")
        return

    # C) VIDA COTIDIANA
    print("\n[3/5] Cargando Vida Cotidiana...")
    try:
        df_life = pd.read_json('validation.json')
        map_life = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_life['label'] = df_life['label'].str.lower().map(map_life)
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life = df_life[['text_clean', 'label']].dropna()
        neg_life, neu_life, pos_life, c_life = balancear_dataset(df_life, "Vida")
    except Exception as e:
        print(f"Error cargando JSON: {e}")
        return

    # ==============================================================================
    # 2. CÁLCULO DEL LÍMITE
    # ==============================================================================
    min_global = min(
        c_orig['neg'], c_orig['neu'], c_orig['pos'],
        c_air['neg'], c_air['neu'], c_air['pos'],
        c_life['neg'], c_life['neu'], c_life['pos']
    )
    
    print(f"\n📢 EL LÍMITE DE BALANCEO ES: {min_global} muestras por clase/dataset.")

    # ==============================================================================
    # 3. CREACIÓN DE DATASETS BALANCEADOS
    # ==============================================================================
    def crear_balanceado(neg, neu, pos, n, nombre_origen):
        d1 = neg.sample(n=n, random_state=42)
        d2 = neu.sample(n=n, random_state=42)
        d3 = pos.sample(n=n, random_state=42)
        df_bal = pd.concat([d1, d2, d3]).sample(frac=1, random_state=42)
        df_bal['Origen'] = nombre_origen # Etiqueta para la gráfica
        return df_bal

    df_final_orig = crear_balanceado(neg_orig, neu_orig, pos_orig, min_global, "Juegos (Original)")
    df_final_air = crear_balanceado(neg_air, neu_air, pos_air, min_global, "Aerolíneas")
    df_final_life = crear_balanceado(neg_life, neu_life, pos_life, min_global, "Vida Cotidiana")

    # ==============================================================================
    # 4. GENERACIÓN DE GRÁFICA INTERMEDIA (LO NUEVO)
    # ==============================================================================
    print("\n[4/5] Generando imagen de distribución final...")
    
    # Juntamos solo para pintar
    df_viz = pd.concat([df_final_orig, df_final_air, df_final_life])
    
    plt.figure(figsize=(10, 6))
    # Countplot nos cuenta las filas automáticamente
    ax = sns.countplot(data=df_viz, x='Origen', hue='label', palette='viridis')
    
    plt.title(f'Distribución FINAL del Dataset Híbrido (Balanceado a {min_global})')
    plt.ylabel('Cantidad de Tweets')
    plt.xlabel('Fuente de Datos')
    
    # Leyenda bonita
    plt.legend(title='Sentimiento', labels=['Negativo', 'Neutral', 'Positivo'])
    
    # Poner los numeritos encima de las barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                   textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('distribucion_final_entrenamiento.png')
    print("   -> Gráfica guardada: 'distribucion_final_entrenamiento.png'")

    # ==============================================================================
    # 5. SPLIT Y ENTRENAMIENTO
    # ==============================================================================
    # Quitamos la columna 'Origen' antes de entrenar para no ensuciar, aunque RF la ignoraría si solo le pasamos texto
    # Pero mejor limpiar.
    
    # Split individual para reservar Test puro de cada uno
    orig_train, orig_test = train_test_split(df_final_orig, test_size=0.2, random_state=42, stratify=df_final_orig['label'])
    air_train, air_test = train_test_split(df_final_air, test_size=0.2, random_state=42, stratify=df_final_air['label'])
    life_train, life_test = train_test_split(df_final_life, test_size=0.2, random_state=42, stratify=df_final_life['label'])

    # Juntamos los Train
    df_total_train = pd.concat([orig_train, air_train, life_train], ignore_index=True).sample(frac=1, random_state=42)
    
    print(f"\n[5/5] Entrenando con {len(df_total_train)} muestras...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=250, max_depth=250, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(df_total_train['text_clean'], df_total_train['label'])
    
    # Guardar
    joblib.dump(pipeline, 'modelo_hibrido_balanceado.pkl')
    
    # Evaluar
    print("\n" + "="*50)
    print("   RESULTADOS (BALANCEO PERFECTO)")
    print("="*50)
    
    acc_orig = accuracy_score(orig_test['label'], pipeline.predict(orig_test['text_clean']))
    print(f"1. Original (Juegos): {acc_orig:.4f}")
    
    acc_air = accuracy_score(air_test['label'], pipeline.predict(air_test['text_clean']))
    print(f"2. Aerolíneas:        {acc_air:.4f}")
    
    acc_life = accuracy_score(life_test['label'], pipeline.predict(life_test['text_clean']))
    print(f"3. Vida Cotidiana:    {acc_life:.4f}")

    # Global
    total_test = pd.concat([orig_test, air_test, life_test])
    acc_global = accuracy_score(total_test['label'], pipeline.predict(total_test['text_clean']))
    print("-" * 30)
    print(f"🌎 ACCURACY GLOBAL:   {acc_global:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
    
    