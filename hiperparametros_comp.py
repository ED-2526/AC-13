import os
import limpieza
import visu
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report

# Modelos
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb

# ------------------- HELPERS -------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)



def main():
    print('--- CARGANDO Y LIMPIANDO DATOS ---')
    df = limpieza.load_and_clean_data('twitter_training.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    classes = sorted(df['label'].unique())

    vectorizers = {
        'BoW': CountVectorizer(max_features=3000),
        'TF-IDF': TfidfVectorizer(max_features=3000)
    }

    models = {
        'Naive Bayes': MultinomialNB(),
        'LogReg': OneVsRestClassifier(LogisticRegression(max_iter=2000)),
        'SVM': CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000)),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'KNN': KNeighborsClassifier()
    }

    grids = {
        'Naive Bayes': {'clf__alpha': [0.1, 0.5, 1.0]},
        'LogReg': {'clf__estimator__C': [0.1, 1, 10]},
        'SVM': {'clf__estimator__C': [0.1, 1, 10]},
        'Decision Tree': {'clf__max_depth': [10, 20, None]},
        'Random Forest': {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20, None]},
        'XGBoost': {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 6]},
        'AdaBoost': {'clf__n_estimators': [50, 100]},
        'Gradient Boosting': {'clf__n_estimators': [100, 150], 'clf__max_depth': [3, 5]},
        'KNN': {'clf__n_neighbors': [3, 5, 7]}
    }

    ensure_dir('results_hiperparams')
    best_models_results = []
    
    print('--- INICIANDO BÚSQUEDA DE HIPERPARÁMETROS ---')
    for v_name, vect in vectorizers.items():
   
        for m_name, model in models.items():
            print(f'\n--- Procesando {m_name} con {v_name} ---')
            pipe = Pipeline([('vect', vect), ('clf', model)])
            grid = grids.get(m_name)

            if not grid:
                print(f"No se encontró grid para {m_name}, saltando.")
                continue

            out_dir = f'results_hiperparams/{m_name}_{v_name}'
            ensure_dir(out_dir)

            n_jobs = -1 if m_name not in ['KNN', 'SVM'] else 1 # Algunos modelos son muy pesados

            gcv = GridSearchCV(pipe, grid, cv=3, scoring='roc_auc_ovr_weighted', # cv=3 para agilizar
                               n_jobs=n_jobs, verbose=1, return_train_score=True)
            
            # Usar un subset para KNN para no agotar memoria
            if m_name == 'KNN':
                X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42, stratify=y_train)
                gcv.fit(X_train_small, y_train_small)
            else:
                gcv.fit(X_train, y_train)


            best_estimator = gcv.best_estimator_
            
            # Imprimir mejores parámetros
            print(f"Mejores parámetros para {m_name} ({v_name}): {gcv.best_params_}")
            print(f"Mejor puntuación (AUC): {gcv.best_score_:.4f}")

            # Guardar resultados para la comparativa final
            best_models_results.append({
                'name': m_name,
                'vectorizer': v_name,
                'estimator': best_estimator,
                'best_score': gcv.best_score_,
                'best_params': gcv.best_params_
            })
            
            # Guardar matriz de confusión y reporte de clasificación
            prefix = f'{out_dir}/best'
            visu.plot_conf_matrix(best_estimator, X_test, y_test, prefix=prefix)

            report = classification_report(y_test, best_estimator.predict(X_test), target_names=[str(c) for c in classes])
            with open(f'{prefix}_classification_report.txt', 'w') as f:
                f.write(f"Best params: {gcv.best_params_}\n")
                f.write(f"Best CV score (roc_auc_ovr_weighted): {gcv.best_score_:.4f}\n\n")
                f.write(report)
            
            # Guardar el mejor modelo
            joblib.dump(best_estimator, f'{prefix}_model.pkl')

            # --- Generar y guardar gráficas específicas del modelo ---
            print("Generando gráficas específicas del modelo...")
            
            # Gráfica ROC por clase
            visu.plot_roc_per_class(best_estimator, X_test, y_test, classes, out_dir=out_dir)
            
            # Gráfica PR por clase
            visu.plot_pr_per_class(best_estimator, X_test, y_test, classes, out_dir=out_dir)
            
            # Gráficas de resultados del GridSearch
            visu.plot_cv_results(gcv, out_dir=out_dir)


    # --- GENERAR GRÁFICOS COMPARATIVOS ---
    print("\n--- Generando gráficos comparativos de los mejores modelos ---")
    
    # Ordenar resultados por la mejor puntuación
    best_models_results.sort(key=lambda x: x['best_score'], reverse=True)
    
    # Crear carpeta para imágenes si no existe
    ensure_dir('imagenes')

   
    visu.plot_comparison_roc(best_models_results, X_test, y_test, classes, 
                             filename='imagenes/comparativa_roc_mejores_modelos.png', 
                             average='micro')
    
    visu.plot_comparison_pr(best_models_results, X_test, y_test, classes, 
                            filename='imagenes/comparativa_pr_mejores_modelos.png', 
                            average='micro')
    

    print("\n--- PROCESO COMPLETADO ---")
    print("Se han guardado los siguientes artefactos:")
    for res in best_models_results:
        print(f"  - Modelo: {res['name']} ({res['vectorizer']}) -> AUC: {res['best_score']:.4f}")
        print(f"    Mejores Parámetros: {res['best_params']}")
        print(f"    Archivos guardados en: results_hiperparams/{res['name']}_{res['vectorizer']}/")

    print("\nGráficos comparativos guardados en la carpeta 'imagenes/'.")


if __name__ == '__main__':
    main()





