import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import itertools
import os
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

# Carga de stopwords extendidas
nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))
english_stopwords = {'of', 'the', 'in', 'and', 'for', 'with', 'to', 'that', 'it', 'is', 'on', 'as', 'by'}
custom_words_to_remove = {
    'cid', 'etc', 'ser', 'haber', 'ver', 'él', 'caso', 'forma', 'parte', 'producir', 'momento',
    'realizar', 'referir', 'presentar', 'asimismo', 'respecto', 'principalmente', 'finalmente',
    'embargo', 'siguiente', 'través', 'doi', 'https', 'org', 'op', 'cit', 'pp', 'pp.', '100'
}
stopwords_personalizadas = stopwords_es.union(english_stopwords).union(custom_words_to_remove)

# Función para gráficos
def plot_top_n(data, top_n=20, title="Gráfico", xlabel="Descriptores", ylabel="Frecuencia", output_path="plot.png"):
    plt.figure(figsize=(12, 8))
    data.head(top_n).sort_values(ascending=True).plot(kind='barh')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Función para extraer ngramas filtrados
def get_top_ngrams(corpus, ngram_range, n=10):
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=list(stopwords_personalizadas),
        token_pattern=r'\b[a-zA-ZáéíóúñÁÉÍÓÚüÜ]+\b',
        min_df=5  # Aparece en al menos 5 documentos
    ).fit(corpus)

    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    def es_valido(ngram):
        return not any(char.isdigit() for char in ngram) and len(ngram.split()) == ngram_range[0]

    bigramas_excluir = {
        'elaboración propia',
        'si bien',
        'fuente elaboración',
        'cada vez',
        'tesis doctoral',
        'http www',
        'punto vista',
        'llevar cabo',
        'hombre mujer',
        'x x',
        'mismo tiempo',
        'primer lugar',
        'trabajo campo',
        'hombres mujeres'
    }

    return [x for x in words_freq if es_valido(x[0]) and x[0] not in bigramas_excluir][:n]


# Análisis Exploratorio
def realizar_eda(json_path):
    base_nombre = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = os.path.join("analisis_exploratorio", base_nombre)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_json(json_path)

    # Tipos y valores nulos
    print("Tipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # Métricas básicas
    df['n_descriptores'] = df['labels'].apply(len)
    df['longitud_texto'] = df['text'].apply(len)
    print("\nEstadísticas descriptivas:")
    print(df[['n_descriptores', 'longitud_texto']].describe().round(0).astype(int))

    # Distribución exacta de descriptores
    print("\nDistribución exacta del número de descriptores por documento:")
    print(df['n_descriptores'].value_counts().sort_index())

    # Outliers en número de descriptores
    print(f"\nMáximo número de descriptores en un documento: {df['n_descriptores'].max()}")
    print(f"Longitud mínima del texto: {df['longitud_texto'].min()}")
    print(f"Longitud máxima del texto: {df['longitud_texto'].max()}")

    # Distribución del número de descriptores
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["n_descriptores"], hue=df["n_descriptores"], palette="viridis", legend=False)
    plt.title("Distribución de número de descriptores por documento")
    plt.xlabel("Número de descriptores")
    plt.ylabel("Número de documentos")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribucion_num_descriptores.png"))
    plt.close()

    # Top descriptores
    todas_las_descriptores = list(itertools.chain.from_iterable(df['labels']))
    descriptores_contadas = Counter(todas_las_descriptores)
    descriptores_frecuentes = pd.DataFrame(descriptores_contadas.items(), columns=["Etiqueta", "Frecuencia"]).sort_values(by="Frecuencia", ascending=False)
    plot_top_n(descriptores_frecuentes.set_index("Etiqueta")["Frecuencia"], top_n=30,
               title="Top descriptores más frecuentes", output_path=os.path.join(output_dir, "top_descriptores.png"))

    # Exportar top descriptores a CSV
    descriptores_frecuentes.head(30).to_csv(os.path.join(output_dir, "top_descriptores.csv"), index=False)

    # Heatmap de coocurrencia
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(df["labels"])
    co_matrix = pd.DataFrame(y_bin.T.dot(y_bin), index=mlb.classes_, columns=mlb.classes_)
    np.fill_diagonal(co_matrix.values, 0)
    top_labels = descriptores_frecuentes["Etiqueta"].head(25).tolist()
    plt.figure(figsize=(14, 12))
    sns.heatmap(co_matrix.loc[top_labels, top_labels], annot=True, fmt='d', cmap='coolwarm')
    plt.title("Heatmap de co-ocurrencia entre descriptores más frecuentes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_coocurrencia.png"))
    plt.close()

    # Bigrama más frecuentes
    bigramas = get_top_ngrams(df['text'], (2, 2))
    print("\nTop 10 bigramas:")
    print(pd.DataFrame(bigramas, columns=["Bigrama", "Frecuencia"]))

    # Outliers (longitud del texto)
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df["longitud_texto"])
    plt.title("Boxplot de longitud del texto")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_longitud_texto.png"))
    plt.close()

    # Histograma de longitud del texto
    plt.figure(figsize=(10, 5))
    sns.histplot(df['longitud_texto'], bins=30, kde=True)
    plt.title("Histograma de longitud del texto")
    plt.xlabel("Número de caracteres")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograma_longitud_texto.png"))
    plt.close()

    # Correlación entre variables numéricas
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[["n_descriptores", "longitud_texto"]].corr(), annot=True, cmap="Blues")
    plt.title("Correlación entre número de descriptores y longitud del texto")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlacion_numericas.png"))
    plt.close()

    print(f"\nAnálisis completado. Resultados guardados en: {output_dir}")

# Ejecución
if __name__ == "__main__":
    archivos = ["dataset_migracion_teseo.json", "dataset_migracion_unesco.json"]
    for archivo in archivos:
        realizar_eda(archivo)
