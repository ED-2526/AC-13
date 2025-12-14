import pandas as pd
import scripts.fase_2.limpieza as limpieza
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def main():
    print("--- 📉 GRÁFICA COMPLETA: TRAIN vs TEST (50 -> 550) 📉 ---")
    
    # 1. CARGA RÁPIDA
    print("1. Preparando datos...")
    try:
        df_base = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
        
        df_air = pd.read_csv('Tweets_aerolinea.csv')
        df_air['label'] = df_air['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        
        df_life = pd.read_json('validation.json')
        df_life['label'] = df_life['label'].str.lower().map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        
        df = pd.concat([
            df_base[['text_clean', 'label']], 
            df_air[['text_clean', 'label']].dropna(), 
            df_life[['text_clean', 'label']].dropna()
        ], ignore_index=True).sample(frac=1, random_state=42)
        
        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        print(f"   -> Train: {len(X_train)} | Test: {len(X_test)}")
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    # 2. BUCLE DE PRUEBAS
    profundidades = [50, 100, 200, 300, 400, 500, 550, 600, 650]
    res_train = [] # Guardamos notas de entrenamiento
    res_test = []  # Guardamos notas de examen
    
    print(f"\n2. Calculando curvas de aprendizaje...")

    for depth in profundidades:
        print(f"   Testing max_depth={depth}...", end=" ")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', RandomForestClassifier(
                n_estimators=50,     # Rápido
                max_depth=depth,     
                n_jobs=-1, 
                random_state=42
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Calculamos LAS DOS notas
        acc_train = accuracy_score(y_train, pipeline.predict(X_train)) # ¿Cuánto memorizó?
        acc_test = accuracy_score(y_test, pipeline.predict(X_test))    # ¿Cuánto aprendió?
        
        print(f"Train: {acc_train:.3f} | Test: {acc_test:.3f}")
        
        res_train.append(acc_train)
        res_test.append(acc_test)

    # 3. GENERAR GRÁFICA DOBLE
    print("\n3. Generando imagen comparativa...")
    
    plt.figure(figsize=(10, 6))
    
    # Línea de Entrenamiento (Azul punteada)
    plt.plot(profundidades, res_train, marker='o', linestyle='--', color='blue', alpha=0.6, label='Entrenamiento (Train)')
    
    # Línea de Test (Roja sólida - LA IMPORTANTE)
    plt.plot(profundidades, res_test, marker='o', linestyle='-', color='red', linewidth=2, label='Validación (Test)')
    
    # Decoración
    plt.title('Curva de Validación: Profundidad del Árbol')
    plt.xlabel('Profundidad Máxima (max_depth)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Anotamos el mejor valor de Test
    max_acc = max(res_test)
    idx_max = res_test.index(max_acc)
    plt.annotate(f"Mejor: {max_acc:.1%}", 
                 (profundidades[idx_max], res_test[idx_max]), 
                 xytext=(0,-15), textcoords='offset points', 
                 ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('grafica_train_vs_test.png')
    
    print(f"   -> ✅ Gráfica guardada: 'grafica_train_vs_test.png'")

if __name__ == "__main__":
    main()