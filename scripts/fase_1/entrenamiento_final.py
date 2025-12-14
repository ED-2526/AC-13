import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo
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
from itertools import cycle
import limpieza
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
    # ==============================================================================
    print("\n[1/8] Cargando datos desde 'data/'...")

    # A) ORIGINAL (Juegos)
    try:
        df_base_raw = pd.read_csv('data/twitter_training.csv', header=None, names=['id', 'entity', 'sentiment', 'text'])
        df_base = df_base_raw[df_base_raw['sentiment'] != 'Irrelevant'].copy()
        df_base['label'] = df_base['sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
        df_base['text_clean'] = df_base['text'].astype(str).apply(limpieza.clean_text)
        df_base = df_base.dropna()
        
        X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
            df_base['text_clean'], df_base['label'], test_size=0.2, random_state=42, stratify=df_base['label']
        )
    except Exception as e:
        print(f"❌ Error cargando Juegos: {e}")
        return

    # B) AEROLÍNEAS
    try:
        df_air = pd.read_csv('data/Tweets_aerolinea.csv')
        df_air['label'] = df_air['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air = df_air[['text_clean', 'label']].dropna()
        
        X_train_air, X_test_air, y_train_air, y_test_air = train_test_split(
            df_air['text_clean'], df_air['label'], test_size=0.2, random_state=42, stratify=df_air['label']
        )
    except Exception as e:
        print(f"❌ Error cargando Aerolíneas: {e}")
        return

    # C) VIDA COTIDIANA
    try:
        df_life = pd.read_json('data/validation.json')
        df_life['label'] = df_life['label'].str.lower().map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life = df_life[['text_clean', 'label']].dropna()
        
        X_train_life, X_test_life, y_train_life, y_test_life = train_test_split(
            df_life['text_clean'], df_life['label'], test_size=0.2, random_state=42, stratify=df_life['label']
        )
    except Exception as e:
        print(f"❌ Error cargando Vida Cotidiana: {e}")
        return

    # ==============================================================================
    # 2. FUSIÓN DE TRAINS
    # ==============================================================================
    print("\n[2/8] Fusionando datasets de entrenamiento...")
    
    X_train_total = pd.concat([X_train_base, X_train_air, X_train_life])
    y_train_total = pd.concat([y_train_base, y_train_air, y_train_life])
    
    train_indices = np.random.permutation(len(X_train_total))
    X_train_total = X_train_total.iloc[train_indices]
    y_train_total = y_train_total.iloc[train_indices]
    
    print(f"   -> Entrenando con {len(X_train_total)} tweets.")

    # ==============================================================================
    # 3. ENTRENAMIENTO (MODELO FINAL BALANCEADO)
    # ==============================================================================
    print("\n[3/8] Entrenando Random Forest (Balanced + Depth None)...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train_total, y_train_total)
    
    # Guardar modelo
    if not os.path.exists('../models'): os.makedirs('../models')
    joblib.dump(pipeline, '../models/modelo_final_balanced.pkl')
    print(f"   -> 💾 Modelo guardado en ../models/")

    # ==============================================================================
    # 4. GRÁFICA DE RENDIMIENTO POR DOMINIO
    # ==============================================================================
    print("\n[4/8] Evaluando acierto por cada dominio...")
    
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