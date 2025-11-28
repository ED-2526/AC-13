#Script con el que cargaremos el dataset de twitter
import pandas as pd


#FUNCION PARWA CARGAR EL DATASET
def load_dataset():
    # cambiar con la ruta donde tenemos el csv
    path = r"training.1600000.processed.noemoticon.csv"

    cols = ["sentiment", "id", "date", "query", "user", "text"]

    print(f"Cargando dataset desde: {path}") 
    df = pd.read_csv(path, encoding="latin-1", names=cols)

    return df

#FUNCION DE ANALISIS DEL DATASET - para situarnos en el problema, luego lo saco
def analyze_dataset(df):
    print("\n===== TIPUS DE DATASET =====")
    print(type(df))

    print("\n===== VARIABLES =====")
    print(df.columns)

    print("\n===== CLASSES (sentiment) =====")
    print(df['sentiment'].value_counts())
    print("\nPercentatge:")
    print(df['sentiment'].value_counts(normalize=True)) #contamos con valoraciones de o 0 o 4 (comentarlo para saber si debemos ampliar)

    
    print("\n===== SIZE DE LES REVIEWS =====") 
    df['text_len'] = df['text'].apply(len)
    print(df['text_len'].describe())

    print("\n===== Exemples =====")
    print(df.sample(5))

    #Es multiclass?
    n_clases = df['sentiment'].nunique()
    print(f"\nNúmero de classes: {n_clases}")
    if n_clases > 2:
        print("Problema Multiclass")
    else:
        print("Problema Binari")

    #Esta desbalanceado?
    print("\nComprobació biaix:")
    print(df['sentiment'].value_counts(normalize=True))


if __name__ == "__main__":
    df = load_dataset()
    analyze_dataset(df)


