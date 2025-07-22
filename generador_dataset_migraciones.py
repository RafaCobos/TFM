#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import pdfplumber
import json
from tqdm import tqdm

# Función de extracción robusta
def extraer_secciones_por_posicion(texto_completo):
    secciones_extraidas = {}
    texto_lower = texto_completo.lower()
    patrones = {
        'resumen': r'^\s*(resumen|abstract)\s*$',
        'introduccion': r'^\s*(?:[ivx\d\.\s-]*)introducci(ó|o)n\s*$',
        'conclusiones': r'^\s*(?:[ivx\d\.\s-]*)conclusi(ó|o)n(es)?\s*$',
        'bibliografia': r'^\s*(bibliograf(í|i)a|referencias)\s*$'
    }
    mapa_posiciones = {}
    for nombre_seccion, patron_regex in patrones.items():
        for match in re.finditer(patron_regex, texto_lower, re.MULTILINE | re.IGNORECASE):
            if nombre_seccion not in mapa_posiciones:
                mapa_posiciones[nombre_seccion] = match.start()
    if len(mapa_posiciones) < 2: return None
    secciones_ordenadas = sorted(mapa_posiciones.items(), key=lambda item: item[1])
    texto_final = ""
    secciones_deseadas = ['resumen', 'introduccion', 'conclusiones']
    for i, (nombre_seccion, pos_inicio) in enumerate(secciones_ordenadas):
        if nombre_seccion in secciones_deseadas:
            if i + 1 < len(secciones_ordenadas):
                pos_fin = secciones_ordenadas[i+1][1]
            else:
                pos_fin = pos_inicio + 40000
            texto_seccion = texto_completo[pos_inicio:pos_fin].strip()
            texto_seccion = re.sub(patrones[nombre_seccion], '', texto_seccion, flags=re.IGNORECASE | re.MULTILINE).strip()
            texto_final += f"[{nombre_seccion.upper()}] {texto_seccion} "
    return texto_final.strip()


# Función principal para crear el dataset enriquecido y enfocado
def crear_dataset_enriquecido_migracion(ruta_pdfs, ruta_json_fuente):
    print("--- Iniciando construcción del Dataset enriquecido de migraciones ---")

    # Conjunto de etiquetas que queremos buscar
    etiquetas_objetivo = { #cambiar etiquetas objetivo si se quiere utilizar UNESCO a (Migración, Inmigración, Inmigrante, Emigración, Política migratoria y Migrante)
        "MIGRACIONES",
        "MOVILIDAD Y MIGRACIONES INTERNACIONALES",
        "MOVILIDAD Y MIGRACIONES INTERIORES"
    }

    # Cargar el JSON
    print(f"Cargando archivo JSON fuente: {ruta_json_fuente}")
    with open(ruta_json_fuente, 'r', encoding='utf-8') as f:
        datos_fuente = json.load(f)

    # Filtrar los documentos que nos interesan
    documentos_relevantes = []
    for item in datos_fuente:
        # Juntamos todas las etiquetas de TESEO en una sola
        labels_excel = item.get("labels_teseo", []) # unesco
        labels_autor = item.get("explicitas_teseo", []) # unesco
        labels_modelo = item.get("modelos_teseo", []) # unesco
        toda_la_evidencia = set(labels_excel + labels_autor + labels_modelo)
        
        # Comprobamos si alguna de nuestras etiquetas objetivo está en esta
        if any(label in etiquetas_objetivo for label in toda_la_evidencia):
            documentos_relevantes.append(item)
    
    print(f"Se han encontrado {len(documentos_relevantes)} documentos relevantes para el tema de migración en todas las fuentes.")

    # Procesar solo los PDFs relevantes
    dataset_final = []
    for item in tqdm(documentos_relevantes, desc="Procesando Tesis de Migración"):
        nombre_pdf = item.get("id")
        ruta_pdf_completa = os.path.join(ruta_pdfs, nombre_pdf)

        if not os.path.exists(ruta_pdf_completa): continue
        
        texto_completo = ""
        try:
            with pdfplumber.open(ruta_pdf_completa) as pdf:
                for page in pdf.pages:
                    texto_pagina = page.extract_text(x_tolerance=2)
                    if texto_pagina: texto_completo += "\n" + texto_pagina
        except Exception:
            continue
        
        texto_dorado = extraer_secciones_por_posicion(texto_completo)
        
        if texto_dorado and len(texto_dorado) > 500:
            # Creamos la etiqueta final uniendo todas las fuentes y filtrando
            labels_excel = item.get("labels_teseo", []) # unesco
            labels_autor = item.get("explicitas_teseo", []) # unesco
            labels_modelo = item.get("modelos_teseo", []) # unesco
            toda_la_evidencia = set(labels_excel + labels_autor + labels_modelo)
            
            labels_finales = [label for label in toda_la_evidencia if label in etiquetas_objetivo]
            
            if labels_finales:
                dataset_final.append({
                    "id": nombre_pdf,
                    "text": texto_dorado,
                    "labels": sorted(labels_finales) # Ordenamos para consistencia
                })
    
    print(f"\n\n--- Proceso finalizado ---")
    print(f"Se han generado {len(dataset_final)} registros de alta calidad.")
    return dataset_final

if __name__ == "__main__":
    RUTA_PDFS = r"C:\Users\rafa_\Documents\TFM\Tesis"
    RUTA_JSON_FUENTE = "dataset_entrenamiento_completo.json"
    
    dataset_migracion = crear_dataset_enriquecido_migracion(RUTA_PDFS, RUTA_JSON_FUENTE)
    
    ruta_salida = "dataset_migracion_teseo.json" # "dataset_migracion_unesco.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(dataset_migracion, f, ensure_ascii=False, indent=2)
        
    print(f"\nDataset enriquecido de migraciones guardado en: '{ruta_salida}'")
    print("Dataset creado. ¡Listo para entrenar!")


# In[ ]:




