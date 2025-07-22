import os
import json
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import random
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    logging, EarlyStoppingCallback
)
from datasets import Dataset
import shutil
from joblib import dump
from sklearn.model_selection import train_test_split

logging.set_verbosity_error()

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

set_seed(42)

model_names = [
    "distilbert-base-multilingual-cased",
    "dccuchile/bert-base-spanish-wwm-cased",
    "PlanTL-GOB-ES/roberta-base-bne",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "mrm8488/electricidad-base-discriminator",
    "microsoft/deberta-v3-base",
    "bertin-project/bertin-roberta-base-spanish"
]

ruta_json = "dataset_migracion_teseo.json" # Para generar los modelos de UNESCO ("dataset_migracion_unesco.json")
print(f"--- Cargando datos desde '{ruta_json}' ---")
with open(ruta_json, encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
conteo_etiquetas = Counter(itertools.chain.from_iterable(df['labels']))
etiquetas_raras = {et for et, c in conteo_etiquetas.items() if c < 5}
df['labels_filtradas'] = df['labels'].apply(lambda l: [e for e in l if e not in etiquetas_raras])
df = df[df['labels_filtradas'].apply(len) > 0].copy()
print(f"Documentos conservados: {len(df)}")

textos = df['text'].tolist()
etiquetas = df['labels_filtradas'].tolist()

mlb = MultiLabelBinarizer()
labels_bin = mlb.fit_transform(etiquetas)
num_labels = labels_bin.shape[1]
id2label = {i: l for i, l in enumerate(mlb.classes_)}
label2id = {l: i for i, l in enumerate(mlb.classes_)}

# Métricas
def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.sigmoid(torch.from_numpy(pred.predictions)).numpy()
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_global_th = 0.5

    for th in thresholds:
        y_pred = (probs > th).astype(int)
        f1 = f1_score(labels, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_global_th = th

    class_thresholds = []
    for i in range(probs.shape[1]):
        best_f1_i = 0
        best_th_i = 0.5
        for th in thresholds:
            pred_i = (probs[:, i] > th).astype(int)
            f1_i = f1_score(labels[:, i], pred_i, zero_division=0)
            if f1_i > best_f1_i:
                best_f1_i = f1_i
                best_th_i = th
        class_thresholds.append(best_th_i)

    return {
        'f1_macro': best_f1,
        'precision_macro': precision_score(labels, (probs > best_global_th), average='macro', zero_division=0),
        'recall_macro': recall_score(labels, (probs > best_global_th), average='macro', zero_division=0),
        'best_threshold': best_global_th,
        'best_class_thresholds': class_thresholds
    }

# Gráfica tiempo
def plot_tiempos_entrenamiento(df, nombre_archivo):
    df_sorted = df.sort_values(by="Tiempo (s)", ascending=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(df_sorted))
    cmap = plt.get_cmap('tab20')
    colores = [cmap(i % 20) for i in range(len(df_sorted))]
    ax.bar(x, df_sorted['Tiempo (s)'], color=colores)
    ax.set_title("Tiempo de entrenamiento por modelo (transformers - TESEO)", fontsize=14) # UNESCO
    ax.set_ylabel("Tiempo (s)")
    ax.set_xlabel("Modelo")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['modelo'], rotation=45, ha='right')
    for i, val in enumerate(df_sorted['Tiempo (s)']):
        ax.text(i, val + 1, f"{val:.0f}s", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(nombre_archivo)
    plt.close()
# Gráfica comparativa
def plot_comparativa_conjunta(df, nombre_archivo):
    df_sorted = df.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(18, 10))
    bar1 = ax.bar(x - bar_width, df_sorted['f1_macro'], width=bar_width, label='F1-score', color='tomato')
    bar2 = ax.bar(x, df_sorted['precision_macro'], width=bar_width, label='Precisión', color='steelblue')
    bar3 = ax.bar(x + bar_width, df_sorted['recall_macro'], width=bar_width, label='Exhaustividad', color='mediumpurple')
    ax.set_title("Comparativa de métricas por modelo (transformers - TESEO)") # UNESCO
    ax.set_ylabel("Puntuación")
    ax.set_xlabel("Modelos")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['modelo'], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(nombre_archivo, bbox_inches='tight')
    plt.close()
# Gráfica clase
def plot_por_clase(df_clases, modelo_id):
    promedios = df_clases.loc[df_clases.index.str.contains("avg")]
    clases_reales = df_clases.loc[~df_clases.index.str.contains("avg")]
    clases_reales = clases_reales.sort_values(by="f1-score", ascending=False)
    df_clases = pd.concat([clases_reales, promedios])
    clases = df_clases.index.tolist()
    f1 = df_clases["f1-score"].astype(float)
    prec = df_clases["precision"].astype(float)
    rec = df_clases["recall"].astype(float)
    x = np.arange(len(clases))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(20, 10))
    bar1 = ax.bar(x - bar_width, f1, width=bar_width, label='F1-score', color='tomato')
    bar2 = ax.bar(x, prec, width=bar_width, label='Precisión', color='steelblue')
    bar3 = ax.bar(x + bar_width, rec, width=bar_width, label='Exhaustividad', color='mediumpurple')
    ax.set_title(f"Métricas por clase - {modelo_id}")
    ax.set_ylabel("Puntuación")
    ax.set_xlabel("Clases")
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"resultados_transformers/grafico_clases_{modelo_id}.png", bbox_inches='tight')
    plt.close()

# Función de entrenamiento con la lógica final y correcta
def entrenar_y_evaluar(model_name, textos, labels_bin):
    print(f"\n{'='*25}\n=== Entrenando {model_name} ===\n{'='*25}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    X_train, X_test, y_train, y_test = train_test_split(textos, labels_bin, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_dict({"text": X_train, "labels": y_train.astype(np.float32).tolist()})
    test_dataset = Dataset.from_dict({"text": X_test, "labels": y_test.astype(np.float32).tolist()})

    train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=256), batched=True).remove_columns(["text"])
    test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=256), batched=True).remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification", id2label=id2label, label2id=label2id)

    output_dir = f"./resultados_transformers/{model_name.replace('/', '-')}"
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        eval_strategy="epoch", logging_strategy="epoch",
        num_train_epochs=10, learning_rate=2e-5, weight_decay=0.01,
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True,
        fp16=torch.cuda.is_available(), report_to="none",
        save_total_limit=1,
        seed=42,  
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # Métricas del mejor modelo
    metrics = trainer.evaluate()
    metrics['elapsed_time'] = round(end_time - start_time, 2)
    
    # Obtener las predicciones y los umbrales por separado
    predictions = trainer.predict(test_dataset)
    threshold_details = compute_metrics(predictions)
    
    # Añadir los umbrales al diccionario de métricas
    metrics['best_threshold'] = threshold_details['best_threshold']
    metrics['best_class_thresholds'] = threshold_details['best_class_thresholds']
    
    # Guardado
    modelo_id = model_name.replace("/", "-")
    trainer.save_model(f"resultados_transformers/mejor_modelo_{modelo_id}")

    # Generar el informe de clasificación usando el umbral
    probs = torch.sigmoid(torch.from_numpy(predictions.predictions)).numpy()
    y_pred_bin = (probs > metrics['best_threshold']).astype(int)
    report = classification_report(y_test, y_pred_bin, target_names=mlb.classes_, output_dict=True, zero_division=0)
    df_clases = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    df_clases.to_csv(f"resultados_transformers/{modelo_id}_reporte_clases.csv", encoding="utf-8-sig")
    plot_por_clase(df_clases, modelo_id)

    print(f"Resultados para {model_name}:\n{metrics}")
    return model_name, metrics

# Ejecución
if __name__ == "__main__":
    os.makedirs("resultados_transformers", exist_ok=True)
    resultados_finales = []
    metricas_por_modelo = {}

    for modelo in model_names:
        try:
            nombre, metricas = entrenar_y_evaluar(modelo, textos, labels_bin)
            resultados_finales.append({
                "modelo": nombre,
                "f1_macro": metricas.get("eval_f1_macro"),
                "precision_macro": metricas.get("eval_precision_macro"),
                "recall_macro": metricas.get("eval_recall_macro"),
                "best_threshold": metricas.get("best_threshold"),
                "eval_loss": metricas.get("eval_loss"),
                "Tiempo (s)": metricas.get("elapsed_time")
            })
            metricas_por_modelo[nombre] = metricas
        except Exception as e:
            print(f"\nError en {modelo}: {e}")
            import traceback
            traceback.print_exc()

    if resultados_finales:
        df_resultados = pd.DataFrame(resultados_finales).sort_values(by="f1_macro", ascending=False)
        df_resultados.to_csv("resultados_transformers/comparativa_final.csv", index=False, encoding="utf-8-sig")
        plot_comparativa_conjunta(df_resultados, "resultados_transformers/grafico_comparativa_f1_transformers_TESEO.png") # UNESCO
        plot_tiempos_entrenamiento(df_resultados, "resultados_transformers/comparativa_Tiempos.png")

        mejor_fila = df_resultados.iloc[0]
        mejor_modelo = mejor_fila["modelo"]
        print(f"\nMejor modelo global: {mejor_modelo} con F1-score de {mejor_fila['f1_macro']:.4f}")

        modelo_id = mejor_modelo.replace("/", "-")
        ruta_origen = f"resultados_transformers/mejor_modelo_{modelo_id}"
        
        ruta_destino_global = f"resultados_transformers/mejor_modelo_global_TESEO" # UNESCO
        if os.path.exists(ruta_origen):
            shutil.copytree(ruta_origen, ruta_destino_global, dirs_exist_ok=True)
        
        os.makedirs("modelos", exist_ok=True)
        shutil.copytree(ruta_destino_global, "modelos/TESEO_model", dirs_exist_ok=True) # UNESCO
        
        tokenizer_mejor_modelo = AutoTokenizer.from_pretrained(mejor_modelo)
        tokenizer_mejor_modelo.save_pretrained("modelos/TESEO_tokenizer") # UNESCO
        dump(mlb, "modelos/mlb_teseo.joblib") # Guardar en la carpeta final - UNESCO
        dump(id2label, "modelos/labels_teseo.pkl") # UNESCO

        # Guardar los umbrales usando el diccionario completo
        thresholds = np.array(
            metricas_por_modelo[mejor_modelo].get("best_class_thresholds") or
            [metricas_por_modelo[mejor_modelo]["best_threshold"]] * len(id2label)
        )
        np.save("modelos/thresholds_teseo.npy", thresholds) # UNESCO
        
        print("\n¡Exportación finalizada para la aplicación web en la carpeta 'modelos'!")