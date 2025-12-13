import matplotlib
matplotlib.use('Agg') # Backend sin ventana

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import limpieza
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def plot_validation_curve(train_scores, test_scores, param_range, param_name, filename):
    """
    Pinta la curva de aprendizaje: Entrenamiento vs Validación.
    """
    # Calculamos la media de los 3 folds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    
    # Línea NARANJA: Qué tan bien se sabe los datos de memoria (Training)
    plt.plot(param_range, train_mean, label="Training score", color="darkorange", lw=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
    
    # Línea AZUL: Qué tan bien acierta en datos nuevos (Cross-Validation)
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy", lw=2)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")

    plt.title(f"Curva de Validación: Random Forest ({param_name})", fontsize=14)
    plt.xlabel(f"Profundidad del Árbol ({param_name})", fontsize=12)
    plt.ylabel("Accuracy (Precisión)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"-> Gráfica guardada: {filename}")

def main():
    print("--- ANÁLISIS DE OVERFITTING EN RANDOM FOREST ---")
    
    # 1. Cargar datos
    # Preguntamos si queremos usar el dataset balanceado o el normal
    # Para ver overfitting, a veces es mejor usar el normal para tener más datos, 
    # pero usa el que prefieras. Aquí pongo balance=False por defecto.
    df = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 3Configurar el Pipeline 
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=3000)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    # 4. Definir el rango de 'max_depth' a probar
    
    param_range = np.arange(15, 515, 15) 
    
    print(f"Calculando Curva de Validación para max_depth = {param_range}...")
    print("Usando 200 árboles. Esto va a tardar un poco...")

    # 5. Calcular la curva (Validation Curve)
    # OJO: Aquí pasamos 'X_train' y 'y_train'. 
    train_scores, test_scores = validation_curve(
        pipeline, 
        X_train, 
        y_train, 
        param_name="clf__max_depth", 
        param_range=param_range,
        cv=5, 
        scoring="accuracy", 
        n_jobs=-1, 
        verbose=1
    )

    # 6. Graficar
    ensure_dir('imagenes')
    plot_validation_curve(train_scores, test_scores, param_range, "max_depth", "rf_validation_curve_max_depth.png")

    print("\nANÁLISIS COMPLETADO.")
   
if __name__ == "__main__":
    main()