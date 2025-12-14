import pandas as pd
import limpieza
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg') # Guardar sin mostrar ventana
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def main():
    print("--- 🎓 ENTRENAMIENTO FINAL PRO MAX (ROC + FEATURE IMPORTANCE) 🎓 ---")
    
    # ==============================================================================
    # 1. CARGA Y SPLIT (80% TRAIN / 20% TEST RESERVADO)
    # ==============================================================================
    print("\n[1/7] Preparando los datos...")

    # A) ORIGINAL
    try:
        df_base = limpieza.load_and_clean_data('twitter_training.csv', balance=False)
        base_train, base_test = train_test_split(df_base[['text_clean', 'label']], test_size=0.2, random_state=42, stratify=df_base['label'])
    except: return

    # B) AEROLÍNEAS
    try:
        df_air = pd.read_csv('Tweets_aerolinea.csv')
        df_air['label'] = df_air['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air = df_air[['text_clean', 'label']].dropna()
        air_train, air_test = train_test_split(df_air, test_size=0.2, random_state=42, stratify=df_air['label'])
    except: return

    # C) VIDA COTIDIANA
    try:
        df_life = pd.read_json('validation.json')
        df_life['label'] = df_life['label'].str.lower().map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life = df_life[['text_clean', 'label']].dropna()
        life_train, life_test = train_test_split(df_life, test_size=0.2, random_state=42, stratify=df_life['label'])
    except: return

    # ==============================================================================
    # 2. FUSIÓN DE TRAINS
    # ==============================================================================
    print("\n[2/7] Fusionando datasets...")
    df_train_total = pd.concat([base_train, air_train, life_train], ignore_index=True).sample(frac=1, random_state=42)
    print(f"   -> Train Total: {len(df_train_total)} muestras.")

    # ==============================================================================
    # 3. ENTRENAMIENTO (CONFIGURACIÓN GANADORA)
    # ==============================================================================
    print("\n[3/7] Entrenando Random Forest (TF-IDF + 200 Est + Depth None)...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=600, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(df_train_total['text_clean'], df_train_total['label'])
    joblib.dump(pipeline, 'modelo_final_rfb.pkl')
    print("   -> 💾 Modelo guardado.")

    # ==============================================================================
    # 4. EVALUACIÓN GLOBAL
    # ==============================================================================
    print("\n[4/7] Evaluando rendimiento global...")
    
    # Juntamos todo el test para las gráficas globales
    final_test_x = pd.concat([base_test['text_clean'], air_test['text_clean'], life_test['text_clean']])
    final_test_y = pd.concat([base_test['label'], air_test['label'], life_test['label']])
    
    y_pred = pipeline.predict(final_test_x)
    # IMPORTANTE: Para ROC necesitamos probabilidades, no solo etiquetas
    y_prob = pipeline.predict_proba(final_test_x)
    
    acc_global = accuracy_score(final_test_y, y_pred)
    print(f"🏆 ACCURACY GLOBAL: {acc_global:.4f}")
    print(classification_report(final_test_y, y_pred, target_names=['Neg', 'Neu', 'Pos']))

    # ==============================================================================
    # 5. GRÁFICA 1: MATRIZ DE CONFUSIÓN
    # ==============================================================================
    print("\n[5/7] Generando Matriz de Confusión...")
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(final_test_y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f'Matriz de Confusión Global (Acc: {acc_global:.2%})')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('final_matriz.png')

    # ==============================================================================
    # 6. GRÁFICA 2: CURVAS ROC MULTI-CLASE (¡NUEVO!)
    # ==============================================================================
    print("\n[6/7] Generando Curvas ROC...")
    
    # Binarizamos las etiquetas reales (convertimos 0,1,2 en columnas separadas)
    y_test_bin = label_binarize(final_test_y, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['red', 'blue', 'green']) # Colores para Neg, Neu, Pos
    clases = ['Negativo', 'Neutral', 'Positivo']
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'Curva ROC {clases[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Línea diagonal (azar)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curvas ROC Multi-clase por Sentimiento')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('final_curvas_roc.png')
    print("   -> 📈 Gráfica guardada: 'final_curvas_roc.png'")

    # ==============================================================================
    # 7. GRÁFICA 3: FEATURE IMPORTANCE (LAS PALABRAS CLAVE) (¡NUEVO!)
    # ==============================================================================
    print("\n[7/7] Analizando qué palabras aprendió el modelo...")
    
    # Sacamos el vectorizador y el modelo
    vectorizer = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    # Obtenemos nombres de características e importancias
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    
    # Creamos un DataFrame para ordenar
    df_imp = pd.DataFrame({'palabra': feature_names, 'importancia': importances})
    df_imp = df_imp.sort_values(by='importancia', ascending=False).head(20) # Top 20
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importancia', y='palabra', data=df_imp, palette='viridis')
    plt.title('Top 20 Palabras Más Importantes para el Modelo')
    plt.xlabel('Importancia (Gini Impurity)')
    plt.ylabel('Palabra')
    plt.tight_layout()
    plt.savefig('final_feature_importance.png')
    print("   -> 🧠 Gráfica guardada: 'final_feature_importance.png'")

    print("\n¡PROCESO FINALIZADO! Ya tienes todas las evidencias para tu TFG/TFM. 🚀")

if __name__ == "__main__":
    main()