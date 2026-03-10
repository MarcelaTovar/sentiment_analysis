import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def cargar_datos(file_path):
    df = pd.read_csv(file_path)
    return df


def EDA(df):
    print(df.head())
    print(df["sentiment"].value_counts())
    print(df["review"].iloc[0])

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúñü ]", "", texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return " ".join(palabras)


def main():
    df = cargar_datos('./IMDB Dataset.csv')
    EDA(df)
    df["review_limpia"] = df["review"].apply(limpiar_texto)
    print(df[["review", "review_limpia"]].head())


if __name__ == "__main__":
    main()