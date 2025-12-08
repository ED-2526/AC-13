
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
        'LogReg': LogisticRegression(multi_class='ovr', max_iter=1000),
        'SVM': CalibratedClassifierCV(LinearSVC(dual=False)),
        'Decision Tree': DecisionTreeClassifier(max_depth=15),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False),
        'AdaBoost': AdaBoostClassifier(n_estimators=50),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3),
        'KNN': KNeighborsClassifier(n_neighbors=5)
        

    }

    