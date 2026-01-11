import pandas as pd

# Versión simplificada - asume estructura conocida
def combine_datasets_simple():
    # Cargar dataset 1 (sentiment140)
    df1 = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)
    df1.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    # Cargar dataset 2 (Twitter_Data.csv)
    df2 = pd.read_csv('Twitter_Data.csv', encoding='latin-1')
    
    # Renombrar columnas del segundo dataset si es necesario
    if 'clean_text' in df2.columns:
        df2 = df2.rename(columns={'clean_text': 'text'})
    
    # Generar IDs para el segundo dataset
    start_id = int(df1['ids'].max()) + 1
    df2['ids'] = range(start_id, start_id + len(df2))
    
    # Crear DataFrames finales
    df1_final = df1[['ids', 'text']].copy()
    df2_final = df2[['ids', 'text']].copy()
    
    # Renombrar columnas
    df1_final.columns = ['target_id', 'text']
    df2_final.columns = ['target_id', 'text']
    
    # Combinar
    combined_df = pd.concat([df1_final, df2_final], ignore_index=True)
    
    # Guardar
    combined_df.to_csv('combined_tweets.csv', index=False)
    
    print(f"Dataset combinado creado con {len(combined_df)} registros")
    return combined_df

# Ejecutar
combine_datasets_simple()
