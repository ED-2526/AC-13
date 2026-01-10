import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import spacy

sns.set_style("whitegrid")

# ---------------- CONFIG ----------------
MODEL_PATH = "C:\\Users\\simpl\\Documents\\GitHub\\AC-13\\models\\modelo_final_balanced.pkl"
DATA_PATH = "C:\\Users\\simpl\\Desktop\\UNIVERSIDAD\\cuarto\\Aprenentatge Computacional (1er Semestre)\\AC Projecte\\2526\\twitter_training.csv\\twitter_training.csv"
OUTPUT_DIR = "analisis_accuracy_graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.fase_2 import limpieza

# spaCy
nlp = spacy.load("en_core_web_sm")

# ---------------- FEATURES ----------------
def compute_features(df, model=None):
    # ---- LONGITUD ----
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text_clean'].apply(len)
    df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)

    # ---- COMPLEJIDAD LÉXICA ----
    df['unique_words'] = df['text_clean'].apply(lambda x: len(set(x.split())))
    df['unique_ratio'] = df['unique_words'] / df['word_count'].replace(0, 1)

    def lexical_entropy(text):
        words = text.split()
        if not words:
            return 0
        freqs = np.array(list(pd.Series(words).value_counts(normalize=True)))
        return -(freqs * np.log2(freqs)).sum()

    df['lexical_entropy'] = df['text_clean'].apply(lexical_entropy)

    # ---- SEMÁNTICA ----
    df['sentiment'] = df['text_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)

    def pos_counts(text):
        doc = nlp(text)
        return pd.Series({
            'num_nouns': sum(t.pos_ == 'NOUN' for t in doc),
            'num_verbs': sum(t.pos_ == 'VERB' for t in doc),
            'num_adjs': sum(t.pos_ == 'ADJ' for t in doc),
        })

    df = pd.concat([df, df['text_clean'].apply(pos_counts)], axis=1)

    # ---- INTERACCIONES ----
    df['words_per_noun'] = df['word_count'] / df['num_nouns'].replace(0, 1)
    df['entropy_per_word'] = df['lexical_entropy'] / df['word_count'].replace(0, 1)
    df['sentiment_times_length'] = df['sentiment'] * df['word_count']

    # ---- PREDICCIÓN ----
    if model is not None:
        df['pred'] = model.predict(df['text_clean'])
        df['correct'] = df['pred'] == df['label']

        # ---- INDICADOR DE DIFICULTAD ----
        df['difficult'] = (
            (~df['correct']) &
            (
                (df['lexical_entropy'] > df['lexical_entropy'].quantile(0.75)) |
                (df['unique_ratio'] > df['unique_ratio'].quantile(0.75)) |
                (df['word_count'] > df['word_count'].quantile(0.75))
            )
        )

    return df

# ---------------- PLOTEO ----------------
def plot_accuracy_by_bins(df, model, feature, bins=10):
    df = df.copy()
    df['bin'] = pd.cut(df[feature], bins=bins)

    stats = []
    for b in df['bin'].dropna().unique():
        subset = df[df['bin'] == b]
        if len(subset) < 5:
            continue

        stats.append({
            'bin_left': b.left,
            'bin_label': f"{int(b.left)}–{int(b.right)}",
            'accuracy': accuracy_score(
                subset['label'],
                model.predict(subset['text_clean'])
            )
        })

    if not stats:
        return

    stats_df = pd.DataFrame(stats).sort_values('bin_left')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=stats_df, x='bin_label', y='accuracy', marker='o')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.xlabel(f"{feature} (rangos)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy según {feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"accuracy_por_{feature}.png"))
    plt.close()


def plot_distribution(df, feature, bins=10):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[feature].dropna(), bins=bins)
    plt.xlabel(feature)
    plt.ylabel("Número de tweets")
    plt.title(f"Distribución de tweets según {feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"distribucion_{feature}.png"))
    plt.close()


def plot_sentiment_distribution(df):
    df = df.copy()
    df['sentiment_cat'] = pd.cut(
        df['sentiment'],
        bins=[-1, -0.05, 0.05, 1],
        labels=['Negativo', 'Neutral', 'Positivo']
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x='sentiment',
        hue='sentiment_cat',
        multiple='stack',
        palette={'Negativo': 'red', 'Neutral': 'gray', 'Positivo': 'green'}
    )

    plt.xlabel("Sentimiento")
    plt.ylabel("Número de tweets")
    plt.title("Distribución del sentimiento")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribucion_sentimiento.png"))
    plt.close()


def plot_heatmap_accuracy(df, model, word_bins=10, char_bins=10):
    df = df.copy()
    df['word_bin'] = pd.cut(df['word_count'], bins=word_bins)
    df['char_bin'] = pd.cut(df['char_count'], bins=char_bins)

    heat = pd.DataFrame(
        index=sorted(df['word_bin'].dropna().unique()),
        columns=sorted(df['char_bin'].dropna().unique())
    )

    for w in heat.index:
        for c in heat.columns:
            subset = df[(df['word_bin'] == w) & (df['char_bin'] == c)]
            if len(subset) < 5:
                heat.loc[w, c] = np.nan
            else:
                heat.loc[w, c] = accuracy_score(
                    subset['label'],
                    model.predict(subset['text_clean'])
                )

    plt.figure(figsize=(14, 8))
    sns.heatmap(heat.astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
    plt.xlabel("char_count bins")
    plt.ylabel("word_count bins")
    plt.title("Heatmap de Accuracy: word_count vs char_count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_accuracy_word_vs_char.png"))
    plt.close()


def plot_scatter_accuracy(df, model):
    df = df.copy()
    df['pred'] = model.predict(df['text_clean'])
    df['correct'] = df['pred'] == df['label']

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df,
        x='word_count',
        y='char_count',
        hue='correct',
        palette={True: 'green', False: 'red'},
        alpha=0.6
    )

    plt.xlabel("word_count")
    plt.ylabel("char_count")
    plt.title("Verde = correcto | Rojo = incorrecto")
    plt.legend(title="Predicción correcta")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_correct_incorrect.png"))
    plt.close()


def plot_scatter_difficult(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df,
        x='word_count',
        y='lexical_entropy',
        hue='difficult',
        palette={True: 'red', False: 'gray'},
        alpha=0.6
    )
    plt.xlabel("word_count")
    plt.ylabel("lexical_entropy")
    plt.title("Tweets difíciles (rojo)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_tweets_dificiles.png"))
    plt.close()

# ---------------- ANÁLISIS DE ERRORES ----------------
def show_difficult_tweets(df, n=20):
    cols = [
        'text_clean', 'label', 'pred',
        'word_count', 'unique_ratio',
        'lexical_entropy', 'sentiment'
    ]

    return (
        df[df['difficult']]
        .sort_values(['lexical_entropy', 'unique_ratio'], ascending=False)
        [cols]
        .head(n)
    )


def export_difficult_tweets(df):
    cols = [
        'text_clean', 'label', 'pred',
        'word_count', 'unique_ratio',
        'lexical_entropy', 'sentiment'
    ]

    path = os.path.join(OUTPUT_DIR, "tweets_dificiles_accuracy_baja.csv")
    df[df['difficult']][cols].to_csv(path, index=False, encoding='utf-8')
    print(f"📄 Tweets difíciles exportados en: {path}")

# ---------------- MAIN ----------------
def main():
    print("--- CARGANDO DATOS ---")
    df = limpieza.load_and_clean_data(DATA_PATH, balance=False)

    print("--- CARGANDO MODELO ---")
    model = joblib.load(MODEL_PATH)

    print("--- COMPUTANDO FEATURES ---")
    df = compute_features(df, model)

    print("--- GENERANDO GRÁFICAS ---")
    features = [
        'word_count', 'char_count', 'unique_ratio',
        'lexical_entropy', 'sentiment',
        'num_nouns', 'num_verbs', 'num_adjs'
    ]

    for feat in features:
        plot_accuracy_by_bins(df, model, feat)
        plot_distribution(df, feat)

    plot_sentiment_distribution(df)
    plot_heatmap_accuracy(df, model)
    plot_scatter_accuracy(df, model)
    plot_scatter_difficult(df)

    print("\n--- TWEETS MÁS PROBLEMÁTICOS ---")
    print(show_difficult_tweets(df, n=15))

    export_difficult_tweets(df)

    print(f"\n✔ Todo guardado en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()




