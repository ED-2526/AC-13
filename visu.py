
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pandas as pd
import numpy as np

def plot_model_comparison(results, filename="comparativa_modelos.png"):
    """Genera un gráfico de barras comparando Accuracy y AUC"""
    df_res = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_res, x='AUC', y='Model', hue='Vectorizer', palette='viridis')
    plt.title('Comparativa de Modelos (AUC Score)')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Gráfica guardada: {filename}")
    plt.close()

def plot_roc_curve_multiclass(y_test_bin, y_prob, class_labels, title, filename):
    """Genera la curva ROC por clase (0, 1, 2)"""
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    print(f"Gráfica guardada: {filename}")
    plt.close()

def plot_roc_comparison_top_n(results, y_test_bin, top_n=5, filename="3_comparativa_roc_top5.png"):
    """
    Calcula y grafica la curva ROC macro-promediada para los N mejores modelos.
    """
    n_classes = y_test_bin.shape[1]
    
    # Ordenar resultados y coger los N mejores
    df_res = pd.DataFrame(results).sort_values(by='AUC', ascending=False).head(top_n)

    plt.figure(figsize=(11, 9))
    
    for index, row in df_res.iterrows():
        y_prob = row['y_prob']
        model_name = f"{row['Model']} ({row['Vectorizer']})"
        
        #average ROC
        all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]))
        
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            
        mean_tpr /= n_classes
        
        macro_auc = auc(all_fpr, mean_tpr)
       

        plt.plot(all_fpr, mean_tpr, lw=2, label=f'{model_name} (Macro-AUC = {macro_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate (Promedio)')
    plt.ylabel('True Positive Rate (Promedio)')
    plt.title(f'Curva ROC Comparativa (Macro-Average) - Top {top_n} Modelos')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Gráfica guardada: {filename}")
    plt.close()

def plot_roc_comparison_all(results, y_test_bin, filename="4_comparativa_roc_all.png"):
    """
    Calcula y grafica la curva ROC macro-promediada para todos los modelos.
    """
    n_classes = y_test_bin.shape[1]
    
    df_res = pd.DataFrame(results).sort_values(by='AUC', ascending=False)

    plt.figure(figsize=(12, 10))
    
    for index, row in df_res.iterrows():
        y_prob = row['y_prob']
        model_name = f"{row['Model']} ({row['Vectorizer']})"
        
        
        all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]))
        
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            
        mean_tpr /= n_classes
        
        macro_auc = auc(all_fpr, mean_tpr)
        # --- Fin Lógica ---

        plt.plot(all_fpr, mean_tpr, lw=2, label=f'{model_name} (Macro-AUC = {macro_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate (Promedio)')
    plt.ylabel('True Positive Rate (Promedio)')
    plt.title('Curva ROC Comparativa (Macro-Average) - Todos los Modelos')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.savefig(filename)
    print(f"Gráfica guardada: {filename}")
    plt.close()