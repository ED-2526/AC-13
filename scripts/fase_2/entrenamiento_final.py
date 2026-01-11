import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import limpieza
from itertools import cycle
import data_loader
def main():
    print("--- 🚀 ENTRENAMIENTO FINAL (DOMINIOS + ROC + PR + FEATURES) 🚀 ---")
    
    # ==============================================================================
    # 0. PREPARAR CARPETA DE RESULTADOS
    # ==============================================================================
    output_folder = 'results\Fase_2_hibrido'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Carpeta creada: {output_folder}")
    else:
        print(f"📁 Guardando en: {output_folder}")

    # ==============================================================================
    # 1. CARGA Y SPLIT INDIVIDUAL
 # 1. CARGA Y FUSIÓN (Paso 1)
    df_total = data_loader.load_and_merge_data(
        sample_frac=1.0, data_folder_path='data'
    )
    
    if df_total.empty:
        return
        
    # 2. SPLIT INDIVIDUALIZADO (Paso 2: Reserva 20% de cada uno)
    X_train_total, X_test_global, y_train_total, y_test_global, df_test_completo = data_loader.split_by_domain(
        df_total, test_size=0.2, random_state=42
    )

    if X_train_total is None:
        return
    
    # ==============================================================================
    # 3. ENTRENAMIENTO (MODELO FINAL BALANCEADO)
    # ==============================================================================
    print("\n[3/8] Entrenando Random Forest (Balanced + Depth None)...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000)),
        ('clf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            class_weight='balanced', 
            random_state=42, 
            min_samples_leaf=1,
            min_samples_split=2,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train_total, y_train_total)
    
    # Guardar modelo
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(pipeline, 'models/modelo_final_balanced.pkl', compress=3)
    print(f"   -> 💾 Modelo guardado en models/")

    # ==============================================================================
    # 4. GRÁFICA DE RENDIMIENTO POR DOMINIO
    # ==============================================================================
    print("\n[4/8] Evaluando acierto por cada dominio...")
    # JUEGOS (Base)
    df_test_base = df_test_completo[df_test_completo['domain'] == 'Juegos']
    X_test_base = df_test_base['text_clean']
    y_test_base = df_test_base['label']
    
    # AEROLÍNEAS (Air)
    df_test_air = df_test_completo[df_test_completo['domain'] == 'Aerolineas']
    X_test_air = df_test_air['text_clean']
    y_test_air = df_test_air['label']
    
    # VIDA COTIDIANA (Life)
    df_test_life = df_test_completo[df_test_completo['domain'] == 'Vida']
    X_test_life = df_test_life['text_clean']
    y_test_life = df_test_life['label']
    
    acc_base = accuracy_score(y_test_base, pipeline.predict(X_test_base))
    acc_air = accuracy_score(y_test_air, pipeline.predict(X_test_air))
    acc_life = accuracy_score(y_test_life, pipeline.predict(X_test_life))
    
    X_test_global = pd.concat([X_test_base, X_test_air, X_test_life])
    y_test_global = pd.concat([y_test_base, y_test_air, y_test_life])
    acc_global = accuracy_score(y_test_global, pipeline.predict(X_test_global))
    
    dominios = ['Juegos (Original)', 'Aerolíneas', 'Vida Cotidiana', 'GLOBAL']
    scores = [acc_base, acc_air, acc_life, acc_global]
    colores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(dominios, scores, color=colores)
    plt.ylim(0, 1.1)
    plt.title('Capacidad de Generalización del Modelo Híbrido')
    plt.ylabel('Accuracy (Precisión)')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1%}', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'{output_folder}/final_rendimiento_dominios.png')
    print(f"   -> 📊 Gráfica guardada: final_rendimiento_dominios.png")

    # ==============================================================================
    # 5. MATRIZ DE CONFUSIÓN GLOBAL (BALANCEADA)
    # ==============================================================================
    print("\n[5/8] Generando Matriz de Confusión Global...")
    
    y_pred_global = pipeline.predict(X_test_global)
    cm = confusion_matrix(y_test_global, y_pred_global)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f'Matriz de Confusión Global (Acc: {acc_global:.2%})')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/final_matriz_confusion.png')

    # ==============================================================================
    # 6. CURVAS ROC MULTICLASE
    # ==============================================================================
    print("\n[6/8] Generando Curvas ROC...")
    
    y_prob = pipeline.predict_proba(X_test_global)
    y_test_bin = label_binarize(y_test_global, classes=[0, 1, 2])
    n_classes = 3
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['red', 'blue', 'green'])
    clases = ['Negativo', 'Neutral', 'Positivo']
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC {clases[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC por Sentimiento')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_folder}/final_curvas_roc.png')

    # ==============================================================================
    # 7. CURVAS PRECISION-RECALL (¡NUEVO!)
    # ==============================================================================
    print("\n[7/8] Generando Curvas Precision-Recall...")
    
    plt.figure(figsize=(10, 8))
    # Usamos los mismos colores para ser consistentes
    colors = cycle(['red', 'blue', 'green'])
    clases = ['Negativo', 'Neutral', 'Positivo']

    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
        average_precision = average_precision_score(y_test_bin[:, i], y_prob[:, i])
        
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'P-R {clases[i]} (AP = {average_precision:.2f})')

    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precision (Pureza)')
    plt.title('Curvas Precision-Recall por Sentimiento')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f'{output_folder}/final_curvas_pr.png')
    print(f"   -> 🎯 Gráfica guardada: final_curvas_pr.png")

    # ==============================================================================
    # 8. FEATURE IMPORTANCE
    # ==============================================================================
    print("\n[8/8] Analizando Feature Importance...")

    vectorizer = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    
    df_imp = pd.DataFrame({'palabra': feature_names, 'importancia': importances})
    df_imp = df_imp.sort_values(by='importancia', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importancia', y='palabra', data=df_imp, palette='viridis')
    plt.title('Top 20 Palabras Más Importantes para el Modelo')
    plt.xlabel('Importancia (Gini Impurity)')
    plt.ylabel('Palabra')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/final_feature_importance.png')
    print(f"   -> 🧠 Gráfica guardada: final_feature_importance.png")

    print(f"\n✅ ¡PROCESO COMPLETADO! Revisa la carpeta '{output_folder}'")

if __name__ == "__main__":
    main()