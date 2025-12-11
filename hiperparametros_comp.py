import os
import limpieza
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report

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

def plot_conf_matrix(estimator, X_test, y_test, prefix=''):
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion Matrix {prefix}')
    plt.tight_layout()
    plt.savefig(f'{prefix}_confmatrix.png')
    plt.close()

def plot_roc_pr(estimator, X_test, y_test, classes=[0,1,2], prefix=''):
    # Algunos modelos como SVC no tienen predict_proba, se calibra para ello
    try:
        y_prob = estimator.predict_proba(X_test)
    except:
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_test)
            # Escalar a [0,1] para multiclass
            y_prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        else:
            y_prob = np.zeros((X_test.shape[0], len(classes)))

    y_bin = label_binarize(y_test, classes=classes)

    # ROC
    plt.figure(figsize=(6,5))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve {prefix}'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_roc.png')
    plt.close()

    # PR
    plt.figure(figsize=(6,5))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f'{classes[i]} (AP={ap:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curve {prefix}'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_pr.png')
    plt.close()

def plot_param_grid(gridcv, param_name, out_png=None):
    results = gridcv.cv_results_
    x = list(results[f'param_{param_name}'])
    y = results['mean_test_score']
    yerr = results['std_test_score']

    plt.figure(figsize=(6,4))
    plt.errorbar(x, y, yerr=yerr, fmt='-o')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'{param_name} vs Mean CV Score')
    plt.grid(True)
    if out_png:
        plt.savefig(out_png)
    plt.close()

# ------------------- MAIN -------------------

def main():
    print('--- CARGANDO Y LIMPIANDO DATOS ---')
    df = limpieza.load_and_clean_data('twitter_training.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

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
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss'),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'KNN': KNeighborsClassifier()
    }

    grids = {
        'Naive Bayes': {'clf__alpha': [0.1, 0.5, 1.0]},
        'LogReg': {'clf__estimator__C': [0.01, 0.1, 1, 10]},
        'SVM': {'clf__estimator__C': [0.01, 0.1, 1, 10]},
        'Decision Tree': {'clf__max_depth': [5, 10, 15, None]},
        'Random Forest': {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, 15, None]},
        'XGBoost': {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 6]},
        'AdaBoost': {'clf__n_estimators': [50, 100]},
        'Gradient Boosting': {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5]},
        'KNN': {'clf__n_neighbors': [3,5,7]}  # Reduce número de combinaciones para no agotar memoria
    }

    ensure_dir('results')

    for v_name, vect in vectorizers.items():
        for m_name, model in models.items():
            print(f'--- Procesando {m_name} con {v_name} ---')
            pipe = Pipeline([('vect', vect), ('clf', model)])
            grid = grids.get(m_name, None)
            out_dir = f'results/{m_name}_{v_name}'
            ensure_dir(out_dir)

            if grid:
                # Para KNN evitar usar todos los cores y reducir RAM
                n_jobs = -1 if m_name != 'KNN' else 1

                gcv = GridSearchCV(pipe, grid, cv=5, scoring='roc_auc_ovr_weighted',
                                   n_jobs=n_jobs, verbose=1, return_train_score=True)
                # Si es KNN podemos usar un subset para no saturar memoria
                if m_name == 'KNN':
                    X_train_small = X_train.sample(10000, random_state=42)
                    y_train_small = y_train.loc[X_train_small.index]
                    gcv.fit(X_train_small, y_train_small)
                else:
                    gcv.fit(X_train, y_train)

                # Graficar hiperparámetros
                for param in grid.keys():
                    plot_param_grid(gcv, param, out_png=f'{out_dir}/{param}_cv_score.png')

                # ROC, PR y matriz de confusión del mejor modelo
                plot_roc_pr(gcv.best_estimator_, X_test, y_test, prefix=f'{out_dir}/best')
                plot_conf_matrix(gcv.best_estimator_, X_test, y_test, prefix=f'{out_dir}/best')

                # Reporte
                report = classification_report(y_test, gcv.best_estimator_.predict(X_test))
                with open(f'{out_dir}/classification_report.txt', 'w') as f:
                    f.write(report)

            else:
                pipe.fit(X_train, y_train)
                plot_roc_pr(pipe, X_test, y_test, prefix=f'{out_dir}/model')
                plot_conf_matrix(pipe, X_test, y_test, prefix=f'{out_dir}/model')

if __name__ == '__main__':
    main()


