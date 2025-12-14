import pandas as pd
import limpieza
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def main():
    print("--- 🏆 TORNEO FINAL V3 (SIN KNN): TF-IDF vs Bag of Words (BoW) 🏆 ---")
    
    # ==============================================================================
    # 1. PREPARACIÓN DE DATOS
    # ==============================================================================
    print("\n[1/3] Preparando datos...")
    
    try:
        # Carga Original
        df_base = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
        base_train, base_test = train_test_split(df_base[['text_clean', 'label']], test_size=0.2, random_state=42, stratify=df_base['label'])
        
        # Carga Aerolíneas
        df_air = pd.read_csv('Tweets_aerolinea.csv')
        map_air = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_air['label'] = df_air['airline_sentiment'].map(map_air)
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air = df_air[['text_clean', 'label']].dropna()
        air_train, air_test = train_test_split(df_air, test_size=0.2, random_state=42, stratify=df_air['label'])

        # Carga JSON
        df_life = pd.read_json('validation.json')
        map_life = {'negative': 0, 'neutral': 1, 'positive': 2}
        df_life['label'] = df_life['label'].str.lower().map(map_life)
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life = df_life[['text_clean', 'label']].dropna()
        life_train, life_test = train_test_split(df_life, test_size=0.2, random_state=42, stratify=df_life['label'])
        
        # Fusión
        df_train_total = pd.concat([base_train, air_train, life_train], ignore_index=True).sample(frac=1, random_state=42)
        df_test_total = pd.concat([base_test, air_test, life_test], ignore_index=True)
        
        print(f"   -> Train: {len(df_train_total)} | Test: {len(df_test_total)}")

    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    # ==============================================================================
    # 2. DEFINICIÓN DE MODELOS (SIN KNN)
    # ==============================================================================
    
    tfidf = TfidfVectorizer(max_features=5000)
    bow = CountVectorizer(max_features=5000)
    
    modelos = [
        {
            'nombre': 'Logistic Regression',
            'pipeline': Pipeline([
                ('vect', tfidf), 
                ('clf', LogisticRegression(max_iter=2000, random_state=42))
            ]),
            'params': {
                'vect': [tfidf, bow],       # TF-IDF vs BoW
                'clf__C': [1, 10],          
                'clf__solver': ['liblinear']
            }
        },
        {
            'nombre': 'Linear SVC (SVM)',
            'pipeline': Pipeline([
                ('vect', tfidf),
                ('clf', LinearSVC(dual='auto', random_state=42, max_iter=3000))
            ]),
            'params': {
                'vect': [tfidf, bow],       # TF-IDF vs BoW
                'clf__C': [0.1, 1, 10]
            }
        },
        # KNN ELIMINADO POR MEMORY ERROR
        {
            'nombre': 'Random Forest (El Jefe)',
            'pipeline': Pipeline([
                ('vect', tfidf),
                ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))
            ]),
            'params': {
                'vect': [tfidf, bow],       # TF-IDF vs BoW
                'clf__n_estimators': [200], 
                'clf__max_depth': [None]
            }
        }
    ]

    # ==============================================================================
    # 3. EL COMBATE
    # ==============================================================================
    print("\n[2/3] Buscando la mejor combinación (Modelo + Vectorizador)...")
    
    resultados = []

    for m in modelos:
        print(f"\n🔹 Optimizando: {m['nombre']}...")
        start_time = time.time()
        
        grid = GridSearchCV(m['pipeline'], m['params'], cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(df_train_total['text_clean'], df_train_total['label'])
        
        mejor_modelo = grid.best_estimator_
        mejor_params = grid.best_params_
        
        vect_nombre = "TF-IDF" if isinstance(mejor_params['vect'], TfidfVectorizer) else "BoW (Counts)"
        
        y_pred = mejor_modelo.predict(df_test_total['text_clean'])
        acc = accuracy_score(df_test_total['label'], y_pred)
        
        elapsed = time.time() - start_time
        
        print(f"   ✅ Ganador: {vect_nombre}")
        print(f"   ⚙️ Config: {mejor_params}")
        print(f"   🎯 ACCURACY: {acc:.4f} (Tiempo: {elapsed:.2f}s)")
        
        resultados.append({
            'Modelo': m['nombre'], 
            'Vectorizador': vect_nombre,
            'Accuracy': acc, 
            'Tiempo': elapsed
        })

    # ==============================================================================
    # 4. RESULTADOS FINALES
    # ==============================================================================
    print("\n" + "="*60)
    print("   🏆 PODIO FINAL (SIN KNN) 🏆")
    print("="*60)
    
    resultados.sort(key=lambda x: x['Accuracy'], reverse=True)
    
    for i, res in enumerate(resultados):
        print(f"{i+1}. {res['Modelo']} usando [{res['Vectorizador']}]")
        print(f"   Nota: {res['Accuracy']:.4f}")
        print("-" * 30)

    ganador = resultados[0]
    print(f"\n💡 CONCLUSIÓN FINAL: Debes usar {ganador['Modelo']} con {ganador['Vectorizador']}.")

if __name__ == "__main__":
    main()