import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split # Necesario para el split

# --- CONFIGURACIÓN DE IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import limpieza # Necesita importar tu módulo de limpieza

def load_and_merge_data(sample_frac=1.0, data_folder_path='/data'):
    """
    Carga, limpia y fusiona los tres datasets (Juegos, Aerolíneas, Vida Cotidiana).

    Args:
        sample_frac (float): Fracción (entre 0.0 y 1.0) de los datos a usar. 
        data_folder_path (str): Ruta relativa a la carpeta 'data'.

    Returns:
        pd.DataFrame: DataFrame único con las columnas 'text_clean', 'label' y 'domain'.
    """
    print("⏳ Iniciando la carga, limpieza y fusión de datasets...")
    
    # --- RUTAS DE ARCHIVOS (relativas al directorio raíz del proyecto) ---
    try:
        # A) Juegos
        df_base = pd.read_csv(os.path.join(data_folder_path, 'twitter_training.csv'), header=None, names=['id', 'entity', 'sentiment', 'text'])
        df_base = df_base[df_base['sentiment'] != 'Irrelevant'].copy()
        df_base['label'] = df_base['sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
        df_base['text_clean'] = df_base['text'].astype(str).apply(limpieza.clean_text)
        df_base['domain'] = 'Juegos' # <--- AÑADIDA COLUMNA DOMINIO
        df_base = df_base[['text_clean', 'label', 'domain']].dropna()

        # B) Aerolíneas
        df_air = pd.read_csv(os.path.join(data_folder_path, 'Tweets_aerolinea.csv'))
        df_air['label'] = df_air['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_air['text_clean'] = df_air['text'].astype(str).apply(limpieza.clean_text)
        df_air['domain'] = 'Aerolineas' # <--- AÑADIDA COLUMNA DOMINIO
        df_air = df_air[['text_clean', 'label', 'domain']].dropna()

        # C) Vida Cotidiana
        df_life = pd.read_json(os.path.join(data_folder_path, 'validation.json'))
        df_life['label'] = df_life['label'].str.lower().map({'negative': 0, 'neutral': 1, 'positive': 2})
        df_life['text_clean'] = df_life['text'].astype(str).apply(limpieza.clean_text)
        df_life['domain'] = 'Vida' # <--- AÑADIDA COLUMNA DOMINIO
        df_life = df_life[['text_clean', 'label', 'domain']].dropna()
        
        # Fusión final
        df_total = pd.concat([df_base, df_air, df_life])
        df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True) # Barajar

        # Aplicar muestreo si es necesario
        if sample_frac < 1.0:
            df_total = df_total.sample(frac=sample_frac, random_state=42)
        
        print(f"✅ Carga y limpieza completada. Total de ejemplos: {len(df_total)}")
        return df_total

    except FileNotFoundError as e:
        print(f"❌ ERROR: No se encuentra el archivo de datos. Revisar ruta: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERROR inesperado durante la carga: {e}")
        return pd.DataFrame()

# ==============================================================================
# NUEVA FUNCIÓN: SPLIT INDIVIDUALIZADO (80/20 de cada dominio)
# ==============================================================================
def split_by_domain(df, test_size=0.2, random_state=42):
    # ... (inicio de la función, manejo de errores, etc.) ...
    
    train_dfs = []
    test_dfs = []

    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            domain_df['text_clean'],
            domain_df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=domain_df['label']
        )
        
        # GUARDAMOS EL DOMINIO EN LOS DATAFRAMES
        df_train_d = pd.DataFrame({'text_clean': X_train_d, 'label': y_train_d, 'domain': domain})
        df_test_d = pd.DataFrame({'text_clean': X_test_d, 'label': y_test_d, 'domain': domain})
        
        train_dfs.append(df_train_d)
        test_dfs.append(df_test_d)
        
        print(f"   -> {domain} | Train: {len(X_train_d)} | Test: {len(X_test_d)}")

    # Fusión final de todos los conjuntos de Train y Test
    df_train = pd.concat(train_dfs).sample(frac=1, random_state=random_state)
    df_test = pd.concat(test_dfs).sample(frac=1, random_state=random_state)
    
    print(f"✅ Split completado. Train Total: {len(df_train)} | Test Total: {len(df_test)}")

    # DEVOLVEMOS EL DATAFRAME DE TEST COMPLETO
    return df_train['text_clean'], df_test['text_clean'], df_train['label'], df_test['label'], df_test # <-- AÑADIMOS df_test

if __name__ == '__main__':
    # --- Test del Módulo ---
    print("--- Test de carga y split ---")
    
    # 1. Cargar una muestra
    # Nota: Aquí la ruta es '../data' porque se ejecuta desde scripts/
    df_total = load_and_merge_data(sample_frac=1, data_folder_path='data')
    
    # 2. Dividir esa muestra
    if not df_total.empty:
        X_train, X_test, y_train, y_test = split_by_domain(df_total, test_size=0.2)
        print(f"\nResultado final del Test: Train Size: {len(X_train)} | Test Size: {len(X_test)}")