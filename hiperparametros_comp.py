import os
import limpieza
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Modelos
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb

# ------------------- HELPERS -------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def plot_param_grid(gridcv, param_name, out_png=None):
    results = gridcv.cv_results_
    x = list(results[f'param_{param_name}'])
    y = results['mean_test_score']
    yerr = results['std_test_score']

    plt.figure(figsize=(6,4))
    plt.errorbar(x, y, yerr=yerr, fmt='-o')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV AUC')
    plt.title(f'{param_name} vs Mean CV AUC')
    plt.grid(True)
    if out_png:
        plt.savefig(out_png)
    plt.show()


def plot_roc_pr(estimator, X_test, y_test, classes=[0,1,2], prefix=''):
    y_prob = estimator.predict_proba(X_test)
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
    plt.show()

    # PR
    plt.figure(figsize=(6,5))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f'{classes[i]} (AP={ap:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curve {prefix}'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_pr.png')
    plt.show()

# ------------------- MAIN -------------------

def main():
    # --- Cargar dataset limpio ---
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
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'KNN': KNeighborsClassifier()
    }

    grids = {
        'Naive Bayes': {'clf__alpha': [0.1, 0.5, 1.0, 2.0]},
        'LogReg': {'clf__estimator__C': [0.01, 0.1, 1, 10]},
        'SVM': {'clf__estimator__C': [0.01, 0.1, 1, 10]},  # Ajustado para CalibratedClassifierCV
        'Decision Tree': {'clf__max_depth': [5, 10, 15, 25, None]},
        'Random Forest': {'clf__n_estimators': [20, 50, 100], 'clf__max_depth': [10, 15, 25, None]},
        'XGBoost': {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 6, 9]},
        'AdaBoost': {'clf__n_estimators': [20, 50, 100]},
        'Gradient Boosting': {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5]},
        'KNN': {'clf__n_neighbors': [1, 3, 5, 7, 9]}
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
                gcv = GridSearchCV(pipe, grid, cv=3, scoring='roc_auc_ovr_weighted', n_jobs=-1, verbose=1, return_train_score=True)
                gcv.fit(X_train, y_train)

                # Graficar hiperparámetros
                for param in grid.keys():
                    plot_param_grid(gcv, param, out_png=f'{out_dir}/{param}_cv_auc.png')

                # ROC y PR del mejor modelo
                plot_roc_pr(gcv.best_estimator_, X_test, y_test, prefix=f'{out_dir}/best')

            else:
                # Modelo sin grid
                pipe.fit(X_train, y_train)
                plot_roc_pr(pipe, X_test, y_test, prefix=f'{out_dir}/model')

if __name__ == '__main__':
    main()