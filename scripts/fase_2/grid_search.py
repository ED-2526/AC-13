import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') 
# Eliminamos matplotlib y seaborn si solo vamos a generar el CSV
# import matplotlib.pyplot as plt
# import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN DE RUTAS E IMPORTACIÓN ROBUSTA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Subimos un nivel (scripts/fase_2 -> scripts) y lo añadimos al path
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

import data_loader # <-- ¡IMPORTAMOS EL NUEVO MÓDULO!

def main():
    print("--- 🕵️‍♀️ INICIANDO GRID SEARCH (BUSCANDO EL MODELO PERFECTO) ---")
    
 # ==============================================================================
    # 1. CARGA DE DATOS Y PREPARACIÓN PARA GRID SEARCH (CÓDIGO CORREGIDO)
    # ==============================================================================
    
    # 1. Carga, limpieza y Fusión (utilizando la función que SÍ existe)
    # Ya incluimos el muestreo sample_frac=0.3 dentro de esta función.
    df_total = data_loader.load_and_merge_data(
        sample_frac=1, 
        data_folder_path='data'
    )
    
    if df_total.empty:
        print("❌ Terminando la ejecución porque no hay datos de entrenamiento.")
        return
        
    
    X_total = df_total['text_clean']
    y_total = df_total['label']
    
    # Reservamos el 20% para Test (aunque no lo usemos en Grid Search)
    X_train_grid, _, y_train_grid, _ = train_test_split(
        X_total, 
        y_total, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_total # Mantenemos estratificación simple por sentimiento
    )
    # El Grid Search se ajusta sobre el conjunto de entrenamiento (X_train_grid, y_train_grid)
    X = X_train_grid
    y = y_train_grid

    print(f"   -> Datos listos. Usando {len(X)} ejemplos de entrenamiento para la búsqueda.")
    
   
    param_grid = {
        'clf__n_estimators': [100, 200], 
        'clf__max_depth': [300, 650, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 3, 5], 
        'tfidf__max_features': [5000,6500,8000]
    }

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])

    # ==============================================================================
    # 3. EJECUTAR LA BÚSQUEDA
    # ==============================================================================
    print("\n🚀 Iniciando Grid Search (Esto puede tardar unos minutos)...")
    total_combinations = (len(param_grid['clf__n_estimators']) * len(param_grid['clf__max_depth']) * len(param_grid['clf__min_samples_leaf']) *
                          len(param_grid['clf__min_samples_split']) *
                          len(param_grid['tfidf__max_features']))
                          
    print(f"   Probando {total_combinations} combinaciones (Total de fits: {total_combinations * 3})...")


    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    
    grid_search.fit(X, y)

    # ==============================================================================
    # 4. RESULTADOS (CSV COMPLETO)
    # ==============================================================================
    print("\n✨ ¡BÚSQUEDA COMPLETADA! ✨")
    print("--------------------------------------------------")
    print(f"🏆 MEJOR ACCURACY: {grid_search.best_score_:.4%}")
    print(f"💎 MEJORES PARÁMETROS: {grid_search.best_params_}")
    print("--------------------------------------------------")

    # Guardamos los resultados en un CSV con TODAS las columnas
    results_df = pd.DataFrame(grid_search.cv_results_)
    cols_to_keep = [
        'param_clf__max_depth', 
        'param_clf__min_samples_leaf', 
        'param_clf__min_samples_split', 
        'param_tfidf__max_features', 
        'param_clf__n_estimators', 
        'mean_test_score', 
        'rank_test_score'
    ]
    
    # Renombrar para que sea legible
    results_df = results_df[cols_to_keep].rename(columns={
        'param_clf__max_depth': 'Profundidad',
        'param_clf__min_samples_leaf': 'Min_Samples_Hoja', 
        'param_clf__min_samples_split': 'Min_Samples_Split', 
        'param_tfidf__max_features': 'Max_Features_TFIDF',
        'param_clf__n_estimators': 'Arboles',
        'mean_test_score': 'Accuracy',
        'rank_test_score': 'Ranking'
    })
    
    # Ordenar por mejor resultado
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    
    output_folder = 'results\Fase_2_hibrido\grid_search' # He corregido la ruta
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    results_path = f'{output_folder}/tabla_grid_search_completa.csv'
    results_df.to_csv(results_path, index=False)
    print(f"📝 Tabla de resultados guardada en: {results_path}")

if __name__ == "__main__":
    main()