from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from load_dataset import load_dataset
from split_data import split_dataset

def baseline_model(X_train, y_train, X_val, y_val):
    print("\nEntrenando modelo baseline (TF-IDF + Logistic Regression)...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    
    #Training
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    #Validation
    acc = accuracy_score(y_val, preds)
    print(f"Accuracy en validación: {acc:.4f}")
    
    return model


if __name__ == "__main__":
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)

    model = baseline_model(X_train, y_train, X_val, y_val)
    
    #Test por probar el model
    preds_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, preds_test)
    print(f"Accuracy en test: {acc_test:.4f}")
