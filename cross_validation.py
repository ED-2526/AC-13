from sklearn.model_selection import KFold
from load_dataset import load_dataset

def prepare_cross_validation(X):
    print("Configurando 5-fold cross validation...\n")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold}:")
        print(f"  Train size = {len(train_idx)}")
        print(f"  Val size   = {len(val_idx)}\n")

    return kf


if __name__ == "__main__":
    df = load_dataset()
    prepare_cross_validation(df["text"])
