import matplotlib
matplotlib.use('Agg') # Backend sin ventana

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import limpieza
import os

# Importamos CountVectorizer para BoW
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import Pipeline

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def plot_validation_curve(train_scores, test_scores, param_range, param_name, filename):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    
    # Línea Naranja (Entrenamiento)
    plt.plot(param_range, train_mean, label="Training score", color="darkorange", lw=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
    
    # Línea Azul (Validación Cruzada)
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy", lw=2)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")

    plt.title(f"Curva de Validación: Decision Tree (BoW) - {param_name}", fontsize=14)
    plt.xlabel(f"Profundidad del Árbol ({param_name})", fontsize=12)
    plt.ylabel("Accuracy (Precisión)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"-> Gráfica guardada: {filename}")

def main():
    print("--- ANÁLISIS DE OVERFITTING EN DECISION TREE (BoW) ---")
    
    # 1. Cargar datos
    df = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
    
    print("Separando Train (80%) y Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 2. Configuración para DECISION TREE con BoW
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=3000)), # <--- AQUI ESTÁ EL CAMBIO A BoW
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    # 3. Rango AUMENTADO
    # Antes llegabas a 50 y seguía subiendo.
    # Vamos a probar hasta 150 para ver dónde choca con el techo.
    param_range = np.arange(15, 415, 15) 
    
    print(f"Calculando curva para max_depth = {param_range}...")

    # 4. Calcular curva
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

    # 5. Guardar
    
    plot_validation_curve(train_scores, test_scores, param_range, "max_depth", "dt_bow_validation_curve_155.png")

    print("\nANÁLISIS COMPLETADO.")
    

if __name__ == "__main__":
    main()