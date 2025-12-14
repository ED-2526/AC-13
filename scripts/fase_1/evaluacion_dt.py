import pandas as pd
import limpieza
import visu
import joblib
import matplotlib
matplotlib.use('Agg') # Backend para evitar errores de ventana
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer # <--- OJO: Usamos BoW
from sklearn.pipeline import Pipeline

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print("--- EVALUACIÓN FINAL DEL MODELO (DECISION TREE - BoW) ---")
    
    # 1. Cargar datos (Sin balancear, igual que el RF)
    df = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
    
    # 2. SPLIT (Misma semilla random_state=42 para que sea comparable al RF)
    print("Separando Train (80%) y Test (20%) con random_state=42...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    classes = sorted(df['label'].unique())

    # 3. Configurar el Modelo DEFINITIVO
    # - Vectorizador: CountVectorizer (BoW)
    # - Modelo: DecisionTree
    # - Profundidad: 70 (Basado en tu gráfica de validación donde se aplana)
    print(f"Entrenando Decision Tree con max_depth=70 y BoW...")
    
    final_pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=3000)), # BoW
        ('clf', DecisionTreeClassifier(
            max_depth=250,     # <--- Límite para controlar el overfitting
            random_state=42, 
            criterion='gini'
        ))
    ])

    # 4. Entrenar con el 80% completo
    final_pipeline.fit(X_train, y_train)

    # 5. Evaluación Final con el TEST (20% virgen)
    print("Evaluando sobre el conjunto de Test...")
    y_pred = final_pipeline.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> ACCURACY FINAL EN TEST (DT): {acc:.4f} <<<")
    
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes])
    print("\nReporte de Clasificación:")
    print(report)

    # 6. Guardar resultados
    out_dir = 'resultados_dt_250_depth'
    ensure_dir(out_dir)
    
    # Guardar reporte en texto
    with open(f'{out_dir}/dt_classification_report.txt', 'w') as f:
        f.write(f"Modelo: Decision Tree (BoW)\n")
        f.write(f"Hiperparámetros: max_depth=70\n")
        f.write(f"Accuracy Final: {acc:.4f}\n\n")
        f.write(report)

    # --- GRÁFICAS ---
    print("Generando gráficas finales...")
    
    # Matriz de Confusión
    visu.plot_conf_matrix(final_pipeline, X_test, y_test, prefix=f'{out_dir}/dt_final')
    
    # Curva ROC (Por clase)
    visu.plot_roc_per_class(final_pipeline, X_test, y_test, classes, out_dir=out_dir)
    
    # Curva Precision-Recall (Por clase)
    visu.plot_pr_per_class(final_pipeline, X_test, y_test, classes, out_dir=out_dir)

    # Guardar el modelo
    joblib.dump(final_pipeline, f'{out_dir}/modelo_dt_bow.pkl')
    
    print(f"\nPROCESO COMPLETADO.")
    print(f"Resultados guardados en: '{out_dir}/'")

if __name__ == "__main__":
    main()