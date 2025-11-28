import pandas as pd
from sklearn.model_selection import train_test_split
from load_dataset import load_dataset

def split_dataset(df):
    X = df["text"]
    y = df["sentiment"]

    print("Creació splits (80% train, 10% val, 10% test)...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("==Size==")
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    df = load_dataset()
    split_dataset(df)
