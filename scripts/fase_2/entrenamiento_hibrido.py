import pandas as pd
import scripts.fase_2.limpieza as limpieza
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

def main():
    print("--- 🚀 ENTRENAMIENTO HÍBRIDO TOTAL (3 DATASETS SPLIT 80/20) ---")
    
    # ==============================================================================
    # 1. CARGA Y SPLIT: DATASET ORIGINAL (Marcas/Videojuegos)
    # ==============================================================================
    print("\n[1/5] Procesando Dataset Original (twitter_training.csv)...")
    try:
        # Cargamos todo
        df_base = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
        df_base = df_base[['text_clean', 'label']]
        
        # DIVIDIMOS 80/20 (Igual que haremos con los otros)
        base_train, base_test = train_test_split(df_base, test_size=0.2, random_state=42, stratify=df_base['label'])
        
        print(f"   -> Total Original: {len(df_base)}")
        print(f"   -> Train (80%): {len(base_train)}")
        print(f"   -> Test Reservado (20%): {len(base_test)}")
    except Exception as e:
        print(f"ERROR cargando original: {e}")
        return

    # ==============================================================================
    # 2. CARGA Y SPLIT: AEROLÍNEAS (Tweets_aerolinea.csv)
    # ==============================================================================
    archivo_air = 'Tweets_aerolinea.csv' # <--- NOMBRE CORREGIDO
    print(f"\n[2/5] Procesando Dataset Aerolíneas ({archivo_air})...")
    try:
        df_air = pd.read_csv(archivo_air)
        # Mapeo específico
        map_air = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_air['label'] = df_air['airline_sentiment'].map(map_air)
        # Limpieza
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air = df_air[['text_clean', 'label']].dropna()
        
        # DIVIDIMOS 80/20
        air_train, air_test = train_test_split(df_air, test_size=0.2, random_state=42, stratify=df_air['label'])
        
        print(f"   -> Total Aerolíneas: {len(df_air)}")
        print(f"   -> Train (80%): {len(air_train)}")
        print(f"   -> Test Reservado (20%): {len(air_test)}")
    except Exception as e:
        print(f"ERROR cargando Aerolíneas: {e}")
        return

    # ==============================================================================
    # 3. CARGA Y SPLIT: VIDA COTIDIANA (validation.json)
    # ==============================================================================
    print("\n[3/5] Procesando Dataset Vida Cotidiana (validation.json)...")
    try:
        df_life = pd.read_json('validation.json')
        # Mapeo específico
        map_life = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_life['label'] = df_life['label'].str.lower().map(map_life)
        # Limpieza
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life = df_life[['text_clean', 'label']].dropna()
        
        # DIVIDIMOS 80/20
        life_train, life_test = train_test_split(df_life, test_size=0.2, random_state=42, stratify=df_life['label'])
        
        print(f"   -> Total Vida: {len(df_life)}")
        print(f"   -> Train (80%): {len(life_train)}")
        print(f"   -> Test Reservado (20%): {len(life_test)}")
    except Exception as e:
        print(f"ERROR cargando JSON: {e}")
        return

    # ==============================================================================
    # 4. FUSIÓN DE TRAINS Y ENTRENAMIENTO
    # ==============================================================================
    print("\n[4/5] Fusionando los 3 conjuntos de TRAIN y Entrenando...")
    
    # Juntamos SOLO las partes de entrenamiento
    df_total_train = pd.concat([base_train, air_train, life_train], ignore_index=True)
    
    # Mezclamos (Shuffle)
    df_total_train = df_total_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   -> DATASET DE ENTRENAMIENTO FINAL: {len(df_total_train)} muestras.")
    print("   -> Entrenando Random Forest (250 trees, 250 depth)...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)), 
        ('clf', RandomForestClassifier(
            n_estimators=250, 
            max_depth=250, 
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(df_total_train['text_clean'], df_total_train['label'])
    
    # Guardamos el modelo
    joblib.dump(pipeline, 'modelo_hibrido_final.pkl')
    print("   -> Modelo guardado: 'modelo_hibrido_final.pkl'")

    # ==============================================================================
    # 5. EVALUACIÓN FINAL (LOS 3 EXÁMENES)
    # ==============================================================================
    print("\n" + "="*60)
    print("RESULTADOS FINALES POR DOMINIO (TEST SETS RESERVADOS)")
    print("="*60)
    
    target_names = ['Negative', 'Neutral', 'Positive']
    
    # --- EXAMEN 1: ORIGINAL (Marcas) ---
    print("\n--- 1. DOMINIO ORIGINAL (Marcas/Juegos) ---")
    y_pred_base = pipeline.predict(base_test['text_clean'])
    acc_base = accuracy_score(base_test['label'], y_pred_base)
    print(f"   ACCURACY: {acc_base:.4f}")
    
    # --- EXAMEN 2: AEROLÍNEAS ---
    print("\n--- 2. DOMINIO SERVICIOS (Aerolíneas) ---")
    y_pred_air = pipeline.predict(air_test['text_clean'])
    acc_air = accuracy_score(air_test['label'], y_pred_air)
    print(f"   ACCURACY: {acc_air:.4f}")

    # --- EXAMEN 3: VIDA COTIDIANA ---
    print("\n--- 3. DOMINIO GENERAL (Vida Cotidiana) ---")
    y_pred_life = pipeline.predict(life_test['text_clean'])
    acc_life = accuracy_score(life_test['label'], y_pred_life)
    print(f"   ACCURACY: {acc_life:.4f}")

    # --- MATRIZ GLOBAL (RESUMEN) ---
    print("\n--- RESUMEN GLOBAL ---")
    # Juntamos todos los tests para la foto final
    final_test_x = pd.concat([base_test['text_clean'], air_test['text_clean'], life_test['text_clean']])
    final_test_y = pd.concat([base_test['label'], air_test['label'], life_test['label']])
    
    y_pred_final = pipeline.predict(final_test_x)
    acc_global = accuracy_score(final_test_y, y_pred_final)
    print(f"   ACCURACY PROMEDIO TOTAL: {acc_global:.4f}")
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(final_test_y, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz Final Híbrida Global (Acc: {acc_global:.2f})')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('matriz_final_hibrida_global.png')
    print("   -> Gráfica guardada: 'matriz_final_hibrida_global.png'")

if __name__ == "__main__":
    main()