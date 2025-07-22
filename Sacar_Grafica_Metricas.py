import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from rapidfuzz import fuzz
import re
from statistics import mean
import matplotlib as mpl

# Rutas
RUTA_RESULTADOS = "resultados"
RUTA_JSON = os.path.join(RUTA_RESULTADOS, "resultados_completo.json")
RUTA_EXCEL = os.path.join(RUTA_RESULTADOS, "resumen_metricas_agrupadas.xlsx")

# Funciones auxiliares

def normalizar(texto):
    return re.sub(r"[^\w\s]", "", texto.lower()).strip()

def fuzzy_match_pairs(preds, refs, threshold=90):
    tp = set()
    matched_refs = set()
    for p in preds:
        for r in refs:
            if fuzz.ratio(p, r) >= threshold and r not in matched_refs:
                tp.add(p)
                matched_refs.add(r)
                break
    return tp, matched_refs

def construir_tabla_metricas(totales_dict, conjunto):
    resumen = []
    modelos = sorted(set(m for (m, _) in totales_dict.keys()))
    for modelo in modelos:
        p_vals = totales_dict.get((modelo, "P"), [])
        r_vals = totales_dict.get((modelo, "R"), [])
        f1_vals = totales_dict.get((modelo, "F1"), [])
        if not (len(p_vals) == len(r_vals) == len(f1_vals)):
            continue
        resumen.append({
            "Modelo": modelo,
            "Conjunto": conjunto,
            "n": len(p_vals),
            "P_media": round(np.mean(p_vals), 3),
            "P_mediana": round(np.median(p_vals), 3),
            "R_media": round(np.mean(r_vals), 3),
            "R_mediana": round(np.median(r_vals), 3),
            "F1_media": round(np.mean(f1_vals), 3),
            "F1_mediana": round(np.median(f1_vals), 3)
        })
    return resumen

def graficar_comparativa_metricas(df, conjunto, salida):
    df_subset = df[df["Conjunto"] == conjunto]
    categorias = ["Explícitas", "Automáticos"]
    metricas = ["P_media", "R_media", "F1_media"]
    nombres_metricas = ["Precisión", "Exhaustividad", "F1-score"]

    valores = []
    for metrica in metricas:
        valores.append([df_subset[df_subset["Modelo"] == cat][metrica].values[0] if not df_subset[df_subset["Modelo"] == cat].empty else 0 for cat in categorias])

    x = np.arange(len(nombres_metricas))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, [v[0] for v in valores], width, label='Explícitas', color='#1f77b4')
    plt.bar(x + width/2, [v[1] for v in valores], width, label='Automáticos', color='#ff7f0e')

    for i in range(len(nombres_metricas)):
        plt.text(x[i] - width/2, valores[i][0] + 0.01, f"{valores[i][0]:.2f}", ha='center', fontsize=8)
        plt.text(x[i] + width/2, valores[i][1] + 0.01, f"{valores[i][1]:.2f}", ha='center', fontsize=8)

    plt.xticks(x, nombres_metricas)
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title(f"Métricas medias - {conjunto}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RUTA_RESULTADOS, salida)
    plt.savefig(path)
    plt.close()
    print(f"Gráfico comparativo guardado: {path}")

# Cargar datos

with open(RUTA_JSON, encoding="utf-8") as f:
    datos = json.load(f)

df = pd.DataFrame(datos)
df["labels_unesco"] = df["descriptores_unesco"].fillna("").apply(
    lambda x: [et.strip() for et in x.split(";") if et.strip()]
)
df["labels_teseo"] = df["descriptores_excel_originales_en_teseo"].fillna("").apply(
    lambda x: [et.strip() for et in x.split(";") if et.strip()]
)

metricas_unesco = defaultdict(list)
metricas_teseo = defaultdict(list)

for fila in df.to_dict(orient="records"):
    for conjunto, clave in [("UNESCO", "labels_unesco"), ("TESEO", "labels_teseo")]:
        referencias = set(fila.get(clave, []))
        if not referencias:
            continue

        # Explícitas
        kws = fila.get("explicitas_unesco" if conjunto == "UNESCO" else "explicitas_teseo", [])
        kws_norm = [normalizar(k) for k in kws if normalizar(k)]
        refs_norm = [normalizar(r) for r in referencias if normalizar(r)]
        tp_set, matched_refs = fuzzy_match_pairs(kws_norm, refs_norm, threshold=90)
        fp_set = set(kws_norm) - tp_set
        fn_set = set(refs_norm) - matched_refs
        tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        d = metricas_unesco if conjunto == "UNESCO" else metricas_teseo
        d[("Explícitas", "P")].append(p)
        d[("Explícitas", "R")].append(r)
        d[("Explícitas", "F1")].append(f1)

        # Automáticos (agrupados todos los modelos)
        kws_modelos = fila.get("modelos_unesco" if conjunto == "UNESCO" else "modelos_teseo", [])
        kws_modelos_norm = [normalizar(k) for k in kws_modelos if normalizar(k)]
        tp_set, matched_refs = fuzzy_match_pairs(kws_modelos_norm, refs_norm, threshold=90)
        fp_set = set(kws_modelos_norm) - tp_set
        fn_set = set(refs_norm) - matched_refs
        tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        d[("Automáticos", "P")].append(p)
        d[("Automáticos", "R")].append(r)
        d[("Automáticos", "F1")].append(f1)
        

# Crear resumen
tabla_unesco = construir_tabla_metricas(metricas_unesco, "UNESCO")
tabla_teseo = construir_tabla_metricas(metricas_teseo, "TESEO")
df_excel = pd.concat([pd.DataFrame(tabla_unesco), pd.DataFrame(tabla_teseo)], ignore_index=True)
df_excel_path = os.path.join(RUTA_RESULTADOS, "resumen_explicitas_vs_modelos.xlsx")
df_excel.to_excel(df_excel_path, index=False)

# Gráficas comparativas finales
graficar_comparativa_metricas(df_excel, "UNESCO", "comparativa_explicitas_vs_modelos_unesco.png")
graficar_comparativa_metricas(df_excel, "TESEO", "comparativa_explicitas_vs_modelos_teseo.png")

tiempos_por_modelo = defaultdict(list)

for fila in df.to_dict(orient="records"):
    tiempos = fila.get("tiempos_modelos", {})
    for modelo, tiempo in tiempos.items():
        tiempos_por_modelo[modelo].append(tiempo)

# Calcular el promedio por modelo
tiempos_promedio = {modelo: mean(tiempos) for modelo, tiempos in tiempos_por_modelo.items()}

# Ordenar de menor a mayor (más rápido al más lento)
tiempos_ordenados = dict(sorted(tiempos_promedio.items(), key=lambda x: x[1]))

# Colores para cada modelo
colormap = mpl.colormaps['tab10'].resampled(len(tiempos_ordenados))
colores = [colormap(i) for i in range(len(tiempos_ordenados))]

# Graficar
plt.figure(figsize=(10, 5))
bars = plt.bar(tiempos_ordenados.keys(), tiempos_ordenados.values(), color=colores)

# Añadir etiquetas de tiempo sobre cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f"{height:.2f}s", ha='center', va='bottom', fontsize=8)

plt.ylabel("Tiempo promedio (s)")
plt.title("Tiempo promedio por modelo de extracción")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Guardar imagen
output_path = os.path.join(RUTA_RESULTADOS, "tiempos_promedio_modelos.png")
plt.savefig(output_path)
plt.close()
print(f"Gráfico de tiempos promedio guardado: {output_path}")



