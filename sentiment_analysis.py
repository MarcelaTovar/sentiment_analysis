import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

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

def preprocesar(df):
    df["review_limpia"] = df["review"].apply(limpiar_texto)
    onehot = OneHotEncoder(sparse_output=False, drop="first")
    dummies = onehot.fit_transform(df[["sentiment"]])
    dummies_df = pd.DataFrame(dummies, columns=onehot.get_feature_names_out(["sentiment"]), index=df.index)
    df = pd.concat([df, dummies_df], axis=1)
    print(df.columns)
    return df

def vectorizar_texto(df):
    vectorizador = CountVectorizer(max_features=3000)
    X = vectorizador.fit_transform(df["review_limpia"])
    y = df["sentiment"]
    return X, y

def division_de_datos(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train):
    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)    
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print(classification_report(y_test, y_pred))


def main():
    df = cargar_datos('./IMDB Dataset.csv')
    EDA(df)
    df = preprocesar(df)
    X, y = vectorizar_texto(df)
    X_train, X_test, y_train, y_test = division_de_datos(X, y)
    modelo = entrenar_modelo(X_train, y_train)
    evaluar_modelo(modelo, X_test, y_test)



if __name__ == "__main__":
    main()