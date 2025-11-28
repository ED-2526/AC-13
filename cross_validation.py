from sklearn.model_selection import KFold,  StratifiedKFold
from load_dataset import load_dataset

def prepare_cross_validation(X, y):
    print("Configurando 5-fold cross validation...\n")

    #kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #para mejor balance de clases


    for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
        print("\n======================================================")
        print("        ANALISIS DELS FOLDS (DISTRIBUCIO) ")
        print("======================================================\n")
        
        print(f"Fold {fold}:")
        print(f"  Train size = {len(train_idx)}")
        print(f"  Val size   = {len(val_idx)}\n")
        
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

 
        print("Fold", fold)
        print("\nTrain counts:", y_train.value_counts())
        print("\nVal counts:", y_val.value_counts())
            

    return kf


if __name__ == "__main__":
    df = load_dataset()
    prepare_cross_validation(df["text"], df['sentiment'])
