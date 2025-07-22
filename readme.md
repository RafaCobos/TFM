# Generador de palabras clave automáticas para Tesis Doctorales. Propuesta de uso para TESEO

**Autor:** Cobos Pineda, Rafael 
**Máster:** Máster Universitario en Análisis y Visualización de Datos Masivos/ Visual Analytics and Big Data 
**Universidad:** UNIR 
**Fecha:** Julio 2025

---

## Descripción del Proyecto

Este proyecto, desarrollado para el Trabajo de Fin de Máster, es un generador de palabras clave automáticas para Tesis Doctorales (Teseo y UNESCO). Se utiliza técnicas de Procesamiento del Lenguaje Natural (NLP) para extraer, clasificar y asignar palabras clave y descriptores.

Se utilizan dos modelos principales:
1.  **Modelos Tradicionales:** Se utilizan algoritmos clásicos como Regresión Logística, SVC, RF y XGB combinados con TF-IDF y MiniLM.
2.  **Modelos Transformers:** Se evalúa el rendimiento de arquitecturas de modernas como RoBERTa, BERT y DeBERTa, específicamente para la tarea de clasificación multietiqueta.

Además, se utiliza los vocabularios controlados de **TESEO** y **UNESCO** en formato RDF para validar y contextualizar las palabras clave extraídas. Finalmente, se incluye una **aplicación web interactiva** desarrollada con Streamlit para generar los descriptores para TESEO y las palabras clave para UNESCO.

---

## Estructura 
.
├── Tesis/                     # Carpeta con todos los PDFs de las tesis
├── web/                       # Carpeta de la aplicación Streamlit
│   ├── data/
│   │   ├── teseo.rdf          # Vocabularios (.rdf) para la app
│   ├── modelos/               # Modelos entrenados para la app
│   │   ├── TESEO_model/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── training_args.bin
│   │   ├── TESEO_tokenizer/
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── clf_unesco.pkl
│   │   ├── labels_teseo.pkl
│   │   ├── labels_unesco.pkl
│   │   ├── thresholds_teseo.npy
│   │   └── vectorizer_unesco.pkl
│   ├── config_dominios.py     # Archivo de configuración para la app
│   └── web.py                 # Script principal de la aplicación
│
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Archivo de documentación
│
# Scripts (.py)
├── extraccion_palabras_clave.py
├── generador_dataset_migraciones.py
├── modelo_tradicional.py
├── modelo_transformadores.py
├── Sacar_Grafica_Metricas.py
├── normalizar_nombres.py
├── crear_rdf_teseo.py
├── EDA.py
│
# Notebooks (.ipynb)
├── comprobartokens.ipynb
├── comparar_pdf_excel.ipynb
├── verificacion_titulos.ipynb
│
# Archivos de Datos, Vocabularios y Resultados
├── 00Tesis.xlsx
├── CDEM.xlsx
├── teseo.rdf
├── unesco-thesaurus.rdf
├── documentRDF.pdf
├── dataset_entrenamiento_completo.json
├── dataset_migracion_teseo.json
├── dataset_migracion_unesco.json
└── resultados_completo.json

## Configuración de entorno
Se recomienda usar **Python 3.12.11**

## Instalar las dependencias
pip install -r requirements.txt

## Usar la aplicación web
Para usar la aplicación web de este proyecto, debes ejecutar el siguiente comando en tu terminal, asegurándote de estar en la carpeta web del proyecto:
streamlit run web.py

## Para la descarga de los modelos
https://drive.google.com/drive/folders/158mKAsuDuXlkyXD54XEuunfGs0jHK0lu?usp=drive_link

## Archivos
Ten en cuenta que algunos archivos en el repositorio ya contienen las transformaciones finales, y las versiones previas no están incluidas (para generar los .xlsx de normalizar_nombres.py)
