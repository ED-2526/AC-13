

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

def _get_probas(estimator, X_test):
    """Helper to get prediction probabilities from various estimator types."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        if len(scores.shape) == 1: # Binary case
            scores = np.vstack([-scores, scores]).T
        return np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    return np.zeros((X_test.shape[0], len(estimator.classes_)))

def _save_plot(title, xlabel, ylabel, filename, legend=True, tight_layout=True):
    """Helper to finalize and save a plot."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend: plt.legend()
    plt.grid(True)
    if tight_layout: plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica guardada: {filename}")

def plot_conf_matrix(estimator, X_test, y_test, prefix=''):
    """Generates and saves a confusion matrix."""
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=estimator.classes_, yticklabels=estimator.classes_)
    _save_plot(f'Confusion Matrix', 'Predicted', 'Actual', f'{prefix}_confmatrix.png', legend=False)

def plot_curve_per_class(estimator, X_test, y_test, classes, out_dir, curve_type='roc'):
    """Generates and saves a ROC or PR curve for each class."""
    y_bin = label_binarize(y_test, classes=classes)
    y_prob = _get_probas(estimator, X_test)
    plt.figure(figsize=(12, 10))

    if curve_type == 'roc':
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'Clase {class_name} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        title, xlabel, ylabel = 'ROC Curve per Class', 'False Positive Rate', 'True Positive Rate'
    elif curve_type == 'pr':
        for i, class_name in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            plt.plot(recall, precision, lw=2, label=f'Clase {class_name} (AP={average_precision_score(y_bin[:, i], y_prob[:, i]):.2f})')
        title, xlabel, ylabel = 'Precision-Recall Curve per Class', 'Recall', 'Precision'
    else:
        raise ValueError("curve_type must be 'roc' or 'pr'")
    _save_plot(title, xlabel, ylabel, f'{out_dir}/best_{curve_type}_per_class.png')

def plot_comparison_curve(results, X_test, y_test, classes, filename, curve_type, average='micro', top_n=None):
    """Calculates and plots a comparative ROC or PR curve for multiple models."""
    if top_n:
        results = sorted(results, key=lambda x: x['best_score'], reverse=True)[:top_n]
    y_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(12, 10))

    for res in results:
        y_prob = _get_probas(res['estimator'], X_test)
        model_label = f"{res['name']} ({res['vectorizer']})"
        if curve_type == 'roc' and average == 'micro':
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
            plt.plot(fpr, tpr, lw=2, label=f'{model_label} (Micro-AUC={auc(fpr, tpr):.3f})')
        elif curve_type == 'pr' and average == 'micro':
            precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
            plt.plot(recall, precision, lw=2, label=f'{model_label} (Micro-AP={average_precision_score(y_bin.ravel(), y_prob.ravel()):.3f})')

    title_prefix = "ROC" if curve_type == 'roc' else "Precision-Recall"
    xlabel = "False Positive Rate" if curve_type == 'roc' else "Recall"
    ylabel = "True Positive Rate" if curve_type == 'roc' else "Precision"
    if curve_type == 'roc': plt.plot([0, 1], [0, 1], 'k--', lw=2)
    _save_plot(f'Curva {title_prefix} Comparativa ({average.capitalize()}-Average)',
               f'{xlabel} ({average}-average)', f'{ylabel} ({average}-average)', filename)

def plot_cv_results(gcv, out_dir):
    """Plots the relationship between hyperparameters and CV score (1D or 2D)."""
    cv_results = pd.DataFrame(gcv.cv_results_)
    params = list(gcv.param_grid.keys())
    s_params = [p.replace('clf__', '').replace('estimator__', '') for p in params]

    if len(params) == 1:
        p_name, s_name = params[0], s_params[0]
        p_vals = cv_results[f'param_{p_name}'].apply(str)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=p_vals, y=cv_results['mean_test_score'], marker='o')
        plt.xticks(rotation=30, ha='right')
        _save_plot(f'CV Score vs {s_name}', s_name, 'Mean Test Score (AUC)', f"{out_dir}/{s_name}_cv_auc.png")
    elif len(params) == 2:
        p1, p2 = params
        s_p1, s_p2 = s_params
        pivot_df = cv_results.astype({f'param_{p1}': str, f'param_{p2}': str})
        pivot = pivot_df.pivot_table('mean_test_score', f'param_{p1}', f'param_{p2}')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap='viridis')
        _save_plot(f'CV Score Heatmap: {s_p1} vs {s_p2}', s_p2, s_p1, f"{out_dir}/{s_p1}_vs_{s_p2}_cv_auc.png", legend=False)
    else:
        print(f"No se genera gráfica de tuning para {len(params)} hiperparámetros.")

# --- Aliases for compatibility ---
def plot_roc_per_class(estimator, X_test, y_test, classes, out_dir):
    plot_curve_per_class(estimator, X_test, y_test, classes, out_dir, curve_type='roc')

def plot_pr_per_class(estimator, X_test, y_test, classes, out_dir):
    plot_curve_per_class(estimator, X_test, y_test, classes, out_dir, curve_type='pr')

def plot_comparison_roc(*args, **kwargs):
    plot_comparison_curve(*args, curve_type='roc', **kwargs)

def plot_comparison_pr(*args, **kwargs):
    plot_comparison_curve(*args, curve_type='pr', **kwargs)