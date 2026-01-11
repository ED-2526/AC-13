import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import scripts.fase_2.limpieza as limpieza
import scripts.fase_2.visu as visu

def main():
    # --- CARGA Y PREPARACIÓN DE DATOS ---
    df = limpieza.load_and_clean_data('twitter_training.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    classes = sorted(df['label'].unique())

    # --- CONFIGURACIÓN DE MODELOS Y VECTORIZADORES ---
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
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False),
        'AdaBoost': AdaBoostClassifier(n_estimators=50),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    # --- BUCLE DE ENTRENAMIENTO ---
    results = []
    best_auc = 0
    best_pipeline = None
    best_info = ""
    print("\n--- INICIANDO ENTRENAMIENTO COMPARATIVO ---")
    for v_name, vect in vectorizers.items():
        for m_name, model in models.items():
            print(f"Entrenando {m_name} con {v_name}...")
            pipe = Pipeline([('vect', vect), ('clf', model)])
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            y_prob = visu._get_probas(pipe, X_test)
            
            acc = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            
            results.append({
                'vectorizer': v_name, 
                'name': m_name, 
                'accuracy': acc, 
                'best_score': auc_score,
                'estimator': pipe
            })
            
            if auc_score > best_auc:
                best_auc = auc_score
                best_pipeline = pipe
                best_info = f"{m_name} ({v_name})"

    # --- RESULTADOS Y GRÁFICAS GENERALES ---
    print("\n--- RESULTADOS ---")
    results_df = pd.DataFrame(results).sort_values(by='best_score', ascending=False)
    print(results_df[['name', 'vectorizer', 'best_score', 'accuracy']])
    
    visu.plot_model_comparison(results, filename="imagenes/1_comparativa_modelos.png")
    visu.plot_comparison_roc(results, X_test, y_test, classes, 
                             filename="imagenes/4_comparativa_roc_all_macro.png",
                             average='macro')
    visu.plot_comparison_roc(results, X_test, y_test, classes, 
                             filename="imagenes/3_comparativa_roc_top5_macro.png",
                             top_n=5, average='macro')
    print(f"\n🏆 MEJOR MODELO: {best_info} con AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
