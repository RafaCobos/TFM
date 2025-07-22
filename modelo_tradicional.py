import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random # Importar random
import torch # Importar torch para SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
import nltk
from nltk.corpus import stopwords
from joblib import dump
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# Función para fijar la semilla

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Llamar a esta función al principio del script
set_seed(42)

# Estilo
plt.style.use('default')

# Configuración
resultados = []
reportes_por_modelo = {}
carpeta_salida = "resultados_modelos"
os.makedirs(carpeta_salida, exist_ok=True)

# Espacios de búsqueda para RandomizedSearchCV
param_spaces = {
    "LogReg": {"estimator__C": np.logspace(-3, 2, 10)},
    "SVC": {"estimator__C": np.logspace(-3, 2, 10)},
    "RF": {
        "estimator__n_estimators": [50, 100, 200],
        "estimator__max_depth": [None, 10, 20, 30]
    },
    "XGB": {
        "estimator__n_estimators": [50, 100, 200],
        "estimator__learning_rate": [0.01, 0.1, 0.2],
        "estimator__max_depth": [3, 6, 10]
    }
}

datasets_archivos = {
    "TESEO": "dataset_migracion_teseo.json",
    "UNESCO": "dataset_migracion_unesco.json"
}


nltk.download('stopwords')

stopwords_es = stopwords.words('spanish')

vectorizadores = {
    "TFIDF": TfidfVectorizer(max_features=3000, stop_words=stopwords_es),
    "MiniLM": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
}

modelos = {
    "LogReg": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "SVC": LinearSVC(class_weight='balanced', max_iter=2000, dual=False),
    "RF": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    "XGB": XGBClassifier(eval_metric='logloss', verbosity=0, tree_method='hist', n_jobs=-1, random_state=42)
}

for dataset_name, archivo_json in datasets_archivos.items():
    print(f"\n--- Procesando dataset: {dataset_name} ---")

    with open(archivo_json, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    conteo_etiquetas = df['labels'].explode().value_counts()
    etiquetas_validas = conteo_etiquetas[conteo_etiquetas >= 5].index.tolist()
    df['labels'] = df['labels'].apply(lambda etiquetas: [et for et in etiquetas if et in etiquetas_validas])
    df = df[df['labels'].map(len) > 0] 
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"])
    X = df["text"].tolist()

    sss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train_texts = [X[i] for i in train_idx]
        X_test_texts = [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    for vec_name, vectorizador in vectorizadores.items():
        if vec_name == "MiniLM":
            encoder = vectorizador
            X_train = encoder.encode(X_train_texts, show_progress_bar=True)
            X_test = encoder.encode(X_test_texts, show_progress_bar=True)
        else:
            vectorizador.fit(X_train_texts)
            X_train = vectorizador.transform(X_train_texts)
            X_test = vectorizador.transform(X_test_texts)

        for mod_name, modelo in modelos.items():
            id_modelo = f"{vec_name}_{mod_name}_{dataset_name}"
            print(f"Entrenando modelo: {id_modelo}")

            base_clf = OneVsRestClassifier(modelo)
            param_dist = param_spaces.get(mod_name, {})

            if param_dist:
                cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                search = RandomizedSearchCV(
                    estimator=base_clf, param_distributions=param_dist,
                    n_iter=5, scoring='f1_macro', cv=cv_strategy,
                    verbose=1, random_state=42, n_jobs=-1
                )
            else:
                search = base_clf

            start_time = time.time()
            search.fit(X_train, y_train)
            end_time = time.time()

            clf = search.best_estimator_ if hasattr(search, "best_estimator_") else search
            y_pred = clf.predict(X_test)

            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            tiempo = round(end_time - start_time, 2)

            resultados.append({
                "id_completo": id_modelo, "F1-score": f1, "Precisión": precision,
                "Exhaustividad": recall, "Tiempo (s)": tiempo
            })

            reporte = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)
            df_clases = pd.DataFrame(reporte).T.drop(["accuracy", "macro avg", "weighted avg"], errors='ignore')
            df_clases.to_csv(f"{carpeta_salida}/{id_modelo}_reporte_clases.csv", encoding="utf-8-sig")
            reportes_por_modelo[id_modelo] = df_clases

# Guardar resumen
df_final = pd.DataFrame(resultados)
df_final.to_csv(f"{carpeta_salida}/resumen_modelos.csv", index=False, encoding="utf-8-sig")

for dataset in datasets_archivos.keys():
    df_dataset = df_final[df_final['id_completo'].str.endswith(dataset)]
    if df_dataset.empty:
        continue
    
    mejor_fila = df_dataset.loc[df_dataset['F1-score'].idxmax()]
    mejor_id = mejor_fila['id_completo']

    print(f"\nGuardando mejor modelo de {dataset}: {mejor_id}")
    vec_name, mod_name, _ = mejor_id.split('_')

    # Cargar y procesar de nuevo el dataset completo para re-entrenar el mejor modelo
    archivo_json = datasets_archivos[dataset]
    with open(archivo_json, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    conteo_etiquetas = df['labels'].explode().value_counts()
    etiquetas_validas = conteo_etiquetas[conteo_etiquetas >= 5].index.tolist()
    df['labels'] = df['labels'].apply(lambda etiquetas: [et for et in etiquetas if et in etiquetas_validas])
    df = df[df['labels'].map(len) > 0]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"])
    X = df["text"].tolist()

    vectorizador = vectorizadores[vec_name]
    modelo = modelos[mod_name]
    clf_final = OneVsRestClassifier(modelo)

    if vec_name == "MiniLM":
        X_vec = vectorizador.encode(X, show_progress_bar=True)
    else:
        vectorizador.fit(X)
        X_vec = vectorizador.transform(X)

    # Re-entrenar con todos los datos
    clf_final.fit(X_vec, y)

    # Guardado
    dataset_lower = dataset.lower()
    print(f"Guardando ficheros para el modelo {dataset_lower}...")
    
    # Guarda el clasificador
    dump(clf_final, f"{carpeta_salida}/clf_{dataset_lower}.pkl")
    
    # Guarda el vectorizador (solo si es TF-IDF)
    if vec_name == "TFIDF":
        dump(vectorizador, f"{carpeta_salida}/vectorizer_{dataset_lower}.pkl")
        
    # Guarda el binarizer de etiquetas
    dump(mlb, f"{carpeta_salida}/labels_{dataset_lower}.pkl")
    
    print(f"¡Ficheros para {dataset} guardados con éxito!")

# Gráficas generales
for dataset in df_final['id_completo'].apply(lambda x: x.split('_')[-1]).unique():
    for vectorizador in vectorizadores.keys():
        df_sub = df_final[df_final['id_completo'].str.startswith(vectorizador)]
        df_sub = df_sub[df_sub['id_completo'].str.endswith(dataset)]
        if df_sub.empty:
            continue

        df_sub = df_sub.sort_values(by="F1-score", ascending=False).reset_index(drop=True)
        x = range(len(df_sub))
        bar_width = 0.25

        # Métricas
        fig, ax = plt.subplots(figsize=(18, 10))
        bar1 = ax.bar([i - bar_width for i in x], df_sub['F1-score'], bar_width, label='F1-score', color='tomato')
        bar2 = ax.bar(x, df_sub['Precisión'], bar_width, label='Precisión', color='steelblue')
        bar3 = ax.bar([i + bar_width for i in x], df_sub['Exhaustividad'], bar_width, label='Exhaustividad', color='mediumpurple')

        ax.set_title(f"Comparativa global ({vectorizador} + {dataset})")
        ax.set_ylabel("Puntuación")
        ax.set_xlabel("Modelos")
        ax.set_xticks(x)
        ax.set_xticklabels(df_sub['id_completo'], rotation=45, ha='right')
        ax.set_ylim(0, 1)

        for bars in [bar1, bar2, bar3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', fontsize=9)

        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/grafico_comparativa_f1_{vectorizador}_{dataset}.png")
        plt.close()

        # Tiempos
        df_sub_tiempo = df_sub.sort_values(by="Tiempo (s)", ascending=True)
        plt.figure(figsize=(18, 10))
        cmap = plt.get_cmap('tab20', len(df_sub_tiempo))
        colores = [cmap(i) for i in range(len(df_sub_tiempo))]
        bar_time = plt.bar(df_sub_tiempo['id_completo'], df_sub_tiempo['Tiempo (s)'], color=colores)

        plt.title(f"Tiempo de evaluación ({vectorizador} + {dataset})")
        plt.ylabel("Tiempo (segundos)")
        plt.xlabel("Modelos")
        plt.xticks(rotation=45, ha='right')
        for bar in bar_time:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f} s', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/grafico_tiempo_modelos_{vectorizador}_{dataset}.png")
        plt.close()

# Gráficas por clase
for modelo_id, df_clases in reportes_por_modelo.items():
    promedios = df_clases.loc[df_clases.index.str.contains("avg")]
    clases_reales = df_clases.loc[~df_clases.index.str.contains("avg")]
    clases_reales = clases_reales.sort_values(by="f1-score", ascending=False)
    df_clases = pd.concat([clases_reales, promedios])

    clases = df_clases.index.tolist()
    f1_scor = df_clases["f1-score"].astype(float)
    precision = df_clases["precision"].astype(float)
    exhaustividad = df_clases["recall"].astype(float)

    x = np.arange(len(clases))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(20, 10))
    bar1 = ax.bar(x - bar_width, f1_scor, width=bar_width, label='F1-score', color='tomato')
    bar2 = ax.bar(x, precision, width=bar_width, label='Precisión', color='steelblue')
    bar3 = ax.bar(x + bar_width, exhaustividad, width=bar_width, label='Exhaustividad', color='mediumpurple')

    ax.set_title(f'Métricas por clase - {modelo_id}')
    ax.set_ylabel('Puntuación')
    ax.set_xlabel('Clases')
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=90)
    ax.set_ylim(0, 1)

    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', fontsize=8)

    ax.legend()
    plt.tight_layout()
    nombre_dataset = modelo_id.split('_')[-1]
    plt.savefig(f"{carpeta_salida}/grafico_clases_{nombre_dataset}_{modelo_id}.png")
    plt.close()