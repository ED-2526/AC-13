# -*- coding: utf-8 -*-
"""
Script Principal - Projecte Twitter Sentiment Analysis
"""
from data.load_dataset import load_dataset, analyze_dataset
from data.preprocess import TwitterDataProcessor
from data.split_data import split_dataset
from models.baseline_model import baseline_model
from utils.cross_validation import prepare_cross_validation

def main():
    print(" INICIANT PROJECTE TWITTER SENTIMENT ANALYSIS")
    print("="*70)
    
    # 1. Carregar i analitzar dataset
    print("\n1.  CARREGANT DATASET...")
    df = load_dataset()
    analyze_dataset(df)
    
    # 2. Neteja i processament
    print("\n2. 🧹 NETEGANT I PROCESSANT DADES...")
    processor = TwitterDataProcessor()
    df_netejat = processor.netejar_dataset(df)
    
    # 3. Dividir dades
    print("\n3.  DIVIDINT DADES...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df_netejat)
    
    # 4. Configurar Cross-Validation
    print("\n4.  CONFIGURANT CROSS-VALIDATION...")
    cv_strategy = prepare_cross_validation(X_train)
    
    # 5. Model Baseline
    print("\n5.  ENTRENANT MODEL BASELINE...")
    model = baseline_model(X_train, y_train, X_val, y_val)
    
    print("\n PROJECTE INICIALITZAT CORRECTAMENT!")
    print(" Estructura modular creada i funcionant")

if __name__ == "__main__":
    main()
