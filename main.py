
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Modelos
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier


import limpieza
import visu

def main():
    #CARGA DE DATOS
    df = limpieza.load_and_clean_data('twitter_training.csv')
    
    #  si va lento (descomenta para pruebas rápidas)
    # df = df.sample(10000, random_state=42)

    # SPLIT (Dataset Nuevo vs Antiguo)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Preparar datos para métricas ROC (binarizar etiquetas)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # CONFIGURACIÓN DE MODELOS
    vectorizers = {
        'BoW': CountVectorizer(max_features=3000),
        'TF-IDF': TfidfVectorizer(max_features=3000)
    }
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'LogReg': OneVsRestClassifier(LogisticRegression(max_iter=2000)),
        'SVM': CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000)),
        'Decision Tree': DecisionTreeClassifier(max_depth=15),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss'),
        'AdaBoost': AdaBoostClassifier(n_estimators=50),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    print("\n--- INICIANDO ENTRENAMIENTO COMPARATIVO ---")

    # BUCLE DE ENTRENAMIENTO
    best_auc = 0
    best_pipeline = None
    best_info = ""

    for v_name, vect in vectorizers.items():
        for m_name, model in models.items():
            print(f"Entrenando {m_name} con {v_name}...")
            
            pipe = Pipeline([('vect', vect), ('clf', model)])
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)
            
            # Métricas
            acc = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            
            results.append({
                'Vectorizer': v_name, 'Model': m_name, 
                'Accuracy': acc, 'AUC': auc_score,
                'y_prob': y_prob
            })
            
            # Guardar el mejor
            if auc_score > best_auc:
                best_auc = auc_score
                best_pipeline = pipe
                best_info = f"{m_name} ({v_name})"

    # hacer RESULTADOS i GRÁFICAS generales
    print("\n--- RESULTADOS ---")
    results_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
    print(results_df)
    
    # Guardar gráfica comparativa
    visu.plot_model_comparison(results, filename="1_comparativa_modelos.png")

    # Guardar gráficas ROC comparativas
    visu.plot_roc_comparison_top_n(results, y_test_bin, top_n=5, filename="3_comparativa_roc_top5.png")
    visu.plot_roc_comparison_all(results, y_test_bin, filename="4_comparativa_roc_all.png")

    print(f"\n🏆 MEJOR MODELO: {best_info} con AUC: {best_auc:.4f}")

    # 6. TUNEADO DE HIPERPARÁMETROS (Del ganador)
    print("\n--- TUNING DEL MEJOR MODELO ---")
    
    # Detectamos qué modelo ganó para configurar el grid
    model_step = best_pipeline.named_steps['clf']
    
    param_grid = {}
    print(f"Configurando grid para el modelo ganador: {type(model_step).__name__}")

    # Lógica de grid según el modelo ganador
    if isinstance(model_step, OneVsRestClassifier) and hasattr(model_step.estimator, 'C'): # Para LogisticRegression
        param_grid = {'clf__estimator__C': [0.1, 1, 10]}
    elif isinstance(model_step, CalibratedClassifierCV) and hasattr(model_step.base_estimator, 'C'): # Para SVM
        param_grid = {'clf__base_estimator__C': [0.1, 1, 10]}
    elif isinstance(model_step, MultinomialNB):
        param_grid = {'clf__alpha': [0.1, 0.5, 1.0]}
    elif isinstance(model_step, KNeighborsClassifier):
        param_grid = {'clf__n_neighbors': [3, 5, 7, 9]}
    elif isinstance(model_step, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)):
        param_grid = {'clf__max_depth': [15, 25, 35]}
    
    if not param_grid:
        print("El modelo ganador no tiene un grid de tuning predefinido en este script.")

    # Si hay grid, ejecutamos
    if param_grid:
        grid = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='roc_auc_ovr_weighted')
        grid.fit(X_train, y_train)
        print(f"Mejores parámetros: {grid.best_params_}")
        final_model = grid.best_estimator_
    else:
        print("Modelo no requiere tuning complejo, usamos el base.")
        final_model = best_pipeline

    #EVALUACIÓN POR CLASE
    print("\n--- EVALUACIÓN FINAL POR CLASE ---")
    y_prob_final = final_model.predict_proba(X_test)
    y_pred_final = final_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_final, target_names=['Negativo', 'Neutro', 'Positivo']))
    
    #Gráfica ROC 
    class_labels = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
    visu.plot_roc_curve_multiclass(
        y_test_bin, y_prob_final, class_labels, 
        title=f"ROC por Clase - {best_info} Optimizado",
        filename="2_roc_por_clase.png"
    )

    

if __name__ == "__main__":
    main()