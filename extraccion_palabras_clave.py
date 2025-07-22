import traceback
import unicodedata
import json
import time
from collections import defaultdict
import rdflib
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import yake
import torch
import pdfplumber
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from rdflib.namespace import SKOS
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from summa import keywords as summa_keywords
from rapidfuzz import fuzz
import re
import gc

# Configuración
logging.basicConfig(level=logging.INFO)
nlp = spacy.load("es_core_news_lg")
nlp.max_length = 2500000 
DetectorFactory.seed = 0  # Para resultados consistentes en langdetect

EXTRA_STOPWORDS = {
    "iii", "capitulo", "resumen", "introduccion", "conclusiones", "trabajo",
    "tesis", "indice", "tabla", "figura", "iv", "v", "vi", "vii", "viii", "ix", "x"
}
spanish_stopwords = set(nlp.Defaults.stop_words) | EXTRA_STOPWORDS

# Rutas

RUTA_RDF_UNESCO = "unesco-thesaurus.rdf"
RUTA_RDF_TESEO = "teseo.rdf"
RUTA_PDFS = "Tesis"
RUTA_XLSX_TESEO = "00Tesis.xlsx"
RUTA_XLSX_UNESCO = "CDEM.xlsx"
RUTA_RESULTADOS = "resultados"
os.makedirs(RUTA_RESULTADOS, exist_ok=True)
json_output_path = os.path.join(RUTA_RESULTADOS, "resultados_completo.json")
excel_resumen_metricas = os.path.join(RUTA_RESULTADOS, "resumen_metricas.xlsx")

# Funciones auxiliares de exportación y visualización
# Exporta resultados en JSON
def exportar_json_completo(resultados_docs, json_path="resultados_completo.json"):
    """Exporta los resultados completos a un archivo JSON."""
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resultados_docs, f, indent=2, ensure_ascii=False)
        logging.info(f"JSON completo exportado en: {json_path}")
    except Exception as e:
        logging.error(f"Error exportando JSON completo: {e}")
        
def exportar_json_entrenamiento(df_resultado, terminos_unesco_cache, terminos_teseo_cache, ruta_json="dataset_entrenamiento_completo.json"):
    """Genera un archivo JSON para dataset de entrenamiento basado en los resultados."""
    registros = []

    for _, row in df_resultado.iterrows():
        archivo = row.get("archivo")
        texto = row.get("texto_procesado_lemas", "")

        # UNESCO
        labels_unesco_valid = []
        for et in row.get("labels_unesco", []):
            et_norm = lematizar_kw(normalizar(et))
            if et_norm in terminos_unesco_cache:
                labels_unesco_valid.append(terminos_unesco_cache[et_norm])

        # TESEO
        labels_teseo_valid = []
        for et in row.get("labels_teseo", []):
            et_norm = lematizar_kw(normalizar(et))
            if et_norm in terminos_teseo_cache:
                labels_teseo_valid.append(terminos_teseo_cache[et_norm])

        explicitas_unesco = row.get("explicitas_unesco", [])
        explicitas_teseo = row.get("explicitas_teseo", [])
        modelos_unesco = row.get("modelos_unesco", [])
        modelos_teseo = row.get("modelos_teseo", [])

        registros.append({
            "id": archivo,
            "text": texto,
            "labels_unesco": labels_unesco_valid,
            "labels_teseo": labels_teseo_valid,
            "explicitas_unesco": explicitas_unesco,
            "explicitas_teseo": explicitas_teseo,
            "modelos_unesco": modelos_unesco,
            "modelos_teseo": modelos_teseo
        })

    try:
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(registros, f, indent=2, ensure_ascii=False)
        logging.info(f"JSON de entrenamiento exportado a: {ruta_json}")
    except Exception as e:
        logging.error(f"Error al exportar JSON de entrenamiento: {e}")

# Métricas media y mediana por modelo
def graficar_metricas_total(tabla, tipo, conjunto):

    df = pd.DataFrame(tabla)
    modelos = df['Modelo']
    metrica_suffix = '_media' if tipo == 'media' else '_mediana'
    p_vals = df['P' + metrica_suffix]
    r_vals = df['R' + metrica_suffix]
    f1_vals = df['F1' + metrica_suffix]
    
    x = np.arange(len(modelos))
    width = 0.25

    plt.figure(figsize=(10, 5))
    barras_p = plt.bar(x - width, p_vals, width=width, label='Precisión')
    barras_r = plt.bar(x, r_vals, width=width, label='Exhaustividad')
    barras_f1 = plt.bar(x + width, f1_vals, width=width, label='F1-score')

    # Añadir valores encima de cada barra
    for bars in [barras_p, barras_r, barras_f1]:
        for bar in bars:
            altura = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, altura + 0.015, f'{altura:.2f}',
                     ha='center', va='bottom', fontsize=8)

    plt.xticks(x, modelos, rotation=45)
    plt.ylim(0, 1.05)
    plt.title(f'Métricas ({tipo.capitalize()}) - {conjunto}')
    plt.ylabel('Valor')
    plt.xlabel('Modelo')
    plt.legend()
    plt.tight_layout()

    path = f'{RUTA_RESULTADOS}/metricas_{conjunto.lower()}_{tipo}.png'
    plt.savefig(path)
    plt.show()
    print(f'Guardado gráfico: {path}')

# Normalización y filtros
# Normaliza texto eliminando acentos, puntuación y stopwords básicas
def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(
        c for c in texto
        if unicodedata.category(c) != 'Mn' or c in 'ñÑ'
    )
    texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)
    texto = re.sub(r'\b(de|la|el|los|las|y|en|del|por|con|un|una|para|que|a|al)\b', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto if texto else None
    
# Lematiza una lista de keywords, eliminando stopwords y puntuación
def lematizar_lista_kws(lista_kws):
    docs = nlp.pipe(lista_kws, batch_size=64, n_process=1)
    return [
        " ".join(
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ) for doc in docs
    ]

# Filtra keywords que no cumplen condiciones (stopwords, longitud, patrones)
def filtrar_keywords_basicas(kws, idioma='es'):
    return [
        kw for kw in kws
        if es_keyword_valida(kw) and es_keyword_valida(kw, idioma)
    ]
    
# Lematiza una sola keyword (strings individuales)
def lematizar_kw(kw):
    return " ".join(
        token.lemma_ for token in nlp(kw)
        if not token.is_stop and not token.is_punct and not token.is_space
    )
    
# Reglas básicas para considerar una keyword como válida
# Se filtran números, palabras muy cortas o largas, repetitivas o con demasiados espacios
def es_keyword_valida(kw: str, idioma: str = 'es') -> bool:
    palabras_prohibidas = EXTRA_STOPWORDS | {"b", "c", "d", "e", "f", "g"}
    kw_clean = kw.strip().lower()
    kw_clean = unicodedata.normalize("NFD", kw_clean)
    kw_clean = "".join([c for c in kw_clean if unicodedata.category(c) != "Mn"])
    if any(stop in kw_clean.split() for stop in palabras_prohibidas):
        return False
    if len(kw_clean) < 4 or len(kw_clean) > 40: return False
    if re.search(r"(.)\1{4,}", kw_clean): return False
    if re.fullmatch(r"[\W_]+", kw_clean): return False
    if kw_clean.count(" ") > 4: return False
    if re.search(r"\d", kw_clean): return False
    try:
        if detect(kw_clean) != idioma:
            return False
    except Exception:
        return False
    doc = nlp(kw)
    if all(token.is_oov for token in doc): return False
    if all(token.pos_ not in {'NOUN', 'PROPN', 'VERB'} for token in doc): return False
    if re.fullmatch(r"de(l[oa]s?)?", kw_clean): return False
    if not re.search(r'[aeiouáéíóúüñ]', kw_clean): return False
    return True

# Agrupa keywords similares usando su forma lematizada como clave
def agrupa_similares(keywords):
    agrupadas = defaultdict(list)
    for kw in keywords:
        lema = lematizar_kw(normalizar(kw))
        agrupadas[lema].append(kw)
    return {lema: list(set(forms)) for lema, forms in agrupadas.items()}

# Validar el PDF
def es_texto_valido(texto, min_palabras=200):
    """Devuelve True si el texto extraído es suficientemente largo como para considerarse válido."""
    if not texto:
        return False
    palabras = texto.split()
    return len(palabras) >= min_palabras
# Funciones de extracción y procesado
# Carga RDF y extrae etiquetas (prefLabels) de conceptos
def load_rdf_labels(rdf_path, rdf_name=None): 
    label_cache = {}
    graph = rdflib.Graph()
    try:
        graph.parse(rdf_path, format="xml")
        for subj in graph.subjects(rdflib.RDF.type, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#Concept")):
            labels = [
                o for s, p, o in graph.triples((subj, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"), None))
                if isinstance(o, rdflib.term.Literal) and o.language == 'es'
            ]
            for label in labels:
                original = str(label).strip()
                norm = normalizar(original)
                lema = lematizar_kw(norm)
                if lema:
                    label_cache[lema] = original
        logging.info(f"Cargados {len(label_cache)} términos desde {rdf_name or rdf_path}")
    except Exception as e:
        logging.error(f"Error cargando RDF {rdf_path}: {str(e)}")
    return label_cache

def load_teseo_terms(rdf_path): 
    terminos_teseo = {}
    g = rdflib.Graph()
    try:
        g.parse(rdf_path, format="xml")
        for s, p, o in g:
            if p == SKOS.prefLabel:
                label_str = ""
                if isinstance(o, str):
                    label_str = o
                elif hasattr(o, 'language') and o.language == 'es':
                    label_str = str(o)
                if label_str:
                    norm = normalizar(label_str)
                    lema = lematizar_kw(norm)
                    if lema:
                        terminos_teseo[lema] = label_str
        logging.info(f"Cargados {len(terminos_teseo)} términos desde TESEO")
    except Exception as e:
        logging.error(f"Error cargando TESEO RDF {rdf_path}: {str(e)}")
    return terminos_teseo

# Carga de Excels con descriptores para TESEO y UNESCO
def load_excel_descriptors_teseo(path_excel_teseo: str) -> dict: 
    df = pd.read_excel(path_excel_teseo)
    mapa_pdf_teseo = {}

    for _, row in df.iterrows():
        fichero = str(row.get('Fichero', '')).strip().lower()
        if not fichero or fichero == 'nan':
            continue

        descriptores_raw = str(row.get('Descriptores', '')).strip()
        if not descriptores_raw:
            continue

        descriptores = [desc.strip().lower() for desc in descriptores_raw.split(';') if desc.strip()]
        mapa_pdf_teseo[fichero.replace('.pdf', '')] = descriptores

    return mapa_pdf_teseo

def load_excel_descriptors_unesco(path_excel_unesco: str) -> dict: 

    df = pd.read_excel(path_excel_unesco)
    mapa_pdf_unesco = {}

    for _, row in df.iterrows():
        fichero = str(row.get('Fichero', '')).strip().lower()
        if not fichero or fichero == 'nan':
            continue

        descriptores_raw = str(row.get('Descriptores', '')).strip()
        if not descriptores_raw:
            continue

        # Usar expresión regular para separar por ',' o ';'
        descriptores = [desc.strip().lower() for desc in re.split(r'[;,]', descriptores_raw) if desc.strip()]
        mapa_pdf_unesco[fichero.replace('.pdf', '')] = descriptores

    return mapa_pdf_unesco
    
# Extrae palabras clave explícitas del texto si aparecen bajo etiquetas como Palabras clave
def extract_explicit_keywords(text):
    explicit_kws = {}
    try:
        # Solo si empieza con palabras clave(s)
        match = re.search(r'(?i)(.{0,50})(palabras clave|palabras claves)[\s:;\-–—]+(.{10,700})', text, re.DOTALL)
        if match:
            contexto_previo = match.group(1).strip().lower()
            bloque_encontrado = match.group(2)
            kws_segment = match.group(3).strip()

            # Descartar si lo precede una preposición (como de palabras clave)
            if re.search(r'\b(de|del|a|la|las|en|sobre|por|para|con|una|un)\s*$', contexto_previo):
                logging.debug(f"Bloque descartado por preposición antes de '{bloque_encontrado}': ...{contexto_previo}")
                return {}

            # Descartar si el bloque contiene frases comunes
            if re.search(r'(?i)términos.*(utiliza|usa|emplea)|más frecuentes', kws_segment[:100]):
                logging.debug("Bloque descartado por contener texto narrativo (no es un encabezado de keywords).")
                return {}

            corte_match = re.search(r'\b(resumen|resum|abstract|introducción|summary|introduction)\b', kws_segment, flags=re.IGNORECASE)
            if corte_match:
                kws_segment = kws_segment[:corte_match.start()].strip()

            raw_keywords = re.split(r'[\n,;|•/·\-–—]+|\s{2,}|\s+y\s+', kws_segment)
            logging.debug(f"Palabras clave extraídas en bruto: {raw_keywords}")
            clean_keywords = []
            for kw in raw_keywords:
                kw = re.sub(r'\b(?:[ivxlcdm]{1,4}|[a-zA-Z]|[0-9]{1,3})\b$', '', kw).strip()
                if not kw or len(kw) < 4:
                    continue
                if any(char.isdigit() for char in kw):
                    continue
                if re.search(r'\b(tabla|figura|p[aá]g|estudio)\b', kw):
                    continue
                if len(kw.split()) > 8:
                    continue
                kw = normalizar(kw)
                if kw:
                    clean_keywords.append(kw)
            logging.debug(f"Keywords detectadas antes de filtro: {clean_keywords}")
            if len(clean_keywords) >= 1:
                for norm_kw in clean_keywords:
                    if re.search(r'[a-záéíóúñ]', norm_kw):
                        explicit_kws[norm_kw] = 'Explícita'
            else:
                logging.debug("Bloque con pocas palabras clave válidas. Descartado.")
    except Exception as e:
        logging.error(f"Error en extracción explícita: {str(e)}")
    return explicit_kws

# Extracción con todos los modelos
def extract_keywords_with_models(text, kb_instance=None, top_n=15, ngram_range=(1, 2)):
    resultados_keywords = {}
    tiempos_modelos = {}
    modelos_config = {
        #"distiluse": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        #"miniLM12": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        #"miniLM6": "sentence-transformers/all-MiniLM-L6-v2",
        #"xlm-r": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        #"stsb-xlm-r": "sentence-transformers/stsb-xlm-r-multilingual",
        #"LaBSE": "sentence-transformers/LaBSE",
        "MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2"
        #"e5-base": "intfloat/multilingual-e5-base",
        #"e5-large": "intfloat/e5-large-v2"
    }
    for nombre_modelo in modelos_config.keys():
        try:
            start = time.time()
            kws = kb_instance.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words=list(spanish_stopwords),
                top_n=top_n,
                diversity=0.6,
            )
            end = time.time()
            duracion = round(end - start, 2)
            tiempos_modelos[nombre_modelo] = duracion
    
            kws_norm_dict = {
                norm_kw: 'KeyBERT'
                for kw_tuple in kws
                if kw_tuple and kw_tuple[0]
                for norm_kw in [normalizar(kw_tuple[0])]
                if norm_kw and norm_kw.strip()
            }
            resultados_keywords[nombre_modelo] = kws_norm_dict
        except Exception as e:
            logging.error(f"Error en extracción con KeyBERT ({nombre_modelo}): {str(e)}")
            resultados_keywords[nombre_modelo] = {}


    # YAKE
    try:
        start = time.time()
        yake_extractor = yake.KeywordExtractor(lan="es", n=ngram_range[1], top=top_n, dedupLim=0.85, dedupFunc="seqm")
        yake_kws = yake_extractor.extract_keywords(text)
        end = time.time()
        duracion = round(end - start, 2)
        tiempos_modelos["YAKE"] = duracion
    
        keywords_yake_dict = {
            norm_kw: 'YAKE'
            for kw_tuple in yake_kws
            for norm_kw in [normalizar(kw_tuple[0])]
            if norm_kw and norm_kw.strip()
        }
    
        resultados_keywords["YAKE"] = keywords_yake_dict
    
    except Exception as e:
        logging.error(f"Error extrayendo keywords YAKE: {e}")

    # TF-IDF
    try:
        start = time.time()
        tfidf = TfidfVectorizer(
            max_features=top_n,
            ngram_range=ngram_range,
            stop_words=list(spanish_stopwords)
        )
        tfidf.fit_transform([text])
        end = time.time()
        duracion = round(end - start, 2)
        tiempos_modelos["TF-IDF"] = duracion

        keywords_tfidf = tfidf.get_feature_names_out()
        keywords_tfidf_dict = {
            norm_kw: 'TF-IDF'
            for k in keywords_tfidf
            for norm_kw in [normalizar(k)]
            if norm_kw and norm_kw.strip()
        }
        resultados_keywords["TF-IDF"] = keywords_tfidf_dict
    except Exception as e:
        logging.error(f"Error en TF-IDF: {e}")

    # TextRank-summa
    try:
        start = time.time()
        kw = summa_keywords.keywords(text, words=top_n, language="spanish", split=True)
        end = time.time()
        duracion = round(end - start, 2)
        tiempos_modelos["TextRank-summa"] = duracion

        # Lematizar y filtrar
        filtered = filtrar_keywords_basicas(lematizar_lista_kws(kw))
        filtered = [k for k in filtered if k and k.strip()]
        filtered = [k for k in filtered if text.lower().count(k) >= 2]
        filtered = list(set(filtered))
    
        resultados_keywords["TextRank-summa"] = {k: 'TextRank-summa' for k in filtered}
    except Exception as e:
        logging.error(f"Error en TextRank-summa: {e}")

    # Agrupación por lema
    todas_keywords = []
    for kws in resultados_keywords.values():
        for kw in kws:
            if kw and text.lower().count(kw) >= 2:
                todas_keywords.extend(list(kws.keys()))

    agrupadas = agrupa_similares(todas_keywords)
    for modelo in resultados_keywords:
        resultados_keywords[modelo] = dict(sorted(resultados_keywords[modelo].items()))
    return resultados_keywords, tiempos_modelos

# Filtro de idioma
def filtrar_por_idioma(keywords_dict, idioma='es'):
    filtradas = {}
    for kw, fuente in keywords_dict.items():
        try:
            if kw.strip() and len(kw.strip()) > 2 and detect(kw) == idioma:
                filtradas[kw] = fuente
        except Exception:
            if len(kw.split()) == 1 and len(kw) <=5 :
                 pass
            else:
                logging.warning(f"No se pudo detectar idioma para: '{kw}', descartando.")
            continue
    return filtradas
    
# Compara embeddings de keywords con vocabularios controlados
def get_semantic_matches(extracted_kws_norm, emb_vocab, vocab_terms_norm, vocab_cache_norm_to_orig, embedding_model, umbral=0.7):
    matched_keywords = {}
    extracted_kws_norm = [k for k in extracted_kws_norm if k and k.strip()]
    if not extracted_kws_norm:
        return matched_keywords

    emb_extracted_kws = embedding_model.encode(extracted_kws_norm, convert_to_tensor=True, show_progress_bar=False)
    cos_sim_matrix = util.cos_sim(emb_extracted_kws, emb_vocab)

    for i in range(len(extracted_kws_norm)):
        if cos_sim_matrix.shape[1] == 0:
            continue
        kw_norm = extracted_kws_norm[i]
        max_sim_val, idx_max = torch.max(cos_sim_matrix[i], dim=0)

        if max_sim_val.item() >= umbral:
            matched_vocab_norm = vocab_terms_norm[idx_max.item()]
            original_vocab_term = vocab_cache_norm_to_orig.get(matched_vocab_norm, matched_vocab_norm)
            if kw_norm not in matched_keywords or max_sim_val.item() > matched_keywords[kw_norm][1]:
                matched_keywords[kw_norm] = (original_vocab_term, max_sim_val.item())

    return matched_keywords

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

def comparar_modelos_con_referencias(modelos_keywords_dict, referencias_set, threshold=90):
    referencias_norm = set([normalizar(r) for r in referencias_set])
    metricas_por_modelo = {}

    print(f"\nComparando contra referencias normalizadas: {referencias_norm}\n")

    for modelo, kw_dict in modelos_keywords_dict.items():
        kws_modelo = set([
            normalizar(kw) for kw in kw_dict.keys()
            if es_keyword_valida(kw)
        ])
        tp_set, matched_refs = fuzzy_match_pairs(kws_modelo, referencias_norm, threshold)
        fp_set = kws_modelo - tp_set
        fn_set = referencias_norm - matched_refs

        print(f" - Modelo: {modelo}")
        print(f" - Keywords normalizadas: {sorted(kws_modelo)}")
        print(f" - Verdaderos positivos (TP): {sorted(tp_set)}")
        print(f" - Falsos positivos (FP): {sorted(fp_set)}")
        print(f" - Falsos negativos (FN): {sorted(fn_set)}")

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        metricas_por_modelo[modelo] = {
            "P": precision,
            "R": recall,
            "F1": f1
        }

    return metricas_por_modelo

def extract_text_from_pdf(pdf_path):
    try:
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        return " ".join(text_content) if text_content else ""
    except Exception as e:
        logging.error(f"Error extrayendo texto del PDF {pdf_path}: {e}")
        return ""

def limpiar_texto(texto):
    texto = re.sub(r'[_\-]{2,}', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)
    return texto.strip()

# Aplica extracción, evaluación y similitud a un solo PDF
def process_single_pdf(pdf_file_name, full_pdf_path, pdf_text_content, 
                       descriptores_excel_map, 
                       terminos_unesco_cache, terminos_teseo_cache,
                       vocab_unesco_terms_norm, vocab_teseo_terms_norm,
                       embedding_model,
                       emb_vocab_unesco, emb_vocab_teseo, kb_instance,
                       descriptores_unesco_map,
                       sim_umbral=0.85, ngram_range=(1,2), top_n_extraction=15):
    resultado_pdf = {"archivo": pdf_file_name}
    try:
        detected_lang = detect(pdf_text_content)
        resultado_pdf["idioma_detectado"] = detected_lang
        print(f"Idioma detectado en {pdf_file_name}: {detected_lang}")
    except:
        resultado_pdf["idioma_detectado"] = "und"
        print(f"No se pudo detectar idioma en {pdf_file_name}")
    if detected_lang != "es":
        logging.warning(f"Idioma detectado no es español ({detected_lang}) en {pdf_file_name}. Saltando.")
        return {}, set(), set(), {}, {}, {}, {}
    file_name_sin_ext = pdf_file_name.lower().replace(".pdf", "")
    file_name_no_ext = file_name_sin_ext
    ref_unesco = descriptores_unesco_map.get(file_name_sin_ext, [])
    ref_teseo = descriptores_excel_map.get(file_name_sin_ext, [])
    logging.debug(f"Referencias UNESCO para {file_name_sin_ext}: {ref_unesco}")
    logging.debug(f"Referencias TESEO para {file_name_sin_ext}: {ref_teseo}")

    explicit_kws_dict = extract_explicit_keywords(pdf_text_content)
    explicit_kws_dict = {
    k: v for k, v in explicit_kws_dict.items()
    if k and k.strip()
}

    logging.debug(f"Keywords explícitas extraídas de {pdf_file_name}: {explicit_kws_dict}")
    modelos_keywords_dict, tiempos_modelos_dict = extract_keywords_with_models(
        pdf_text_content, kb_instance=kb_instance, top_n=top_n_extraction, ngram_range=ngram_range
    )
    kws_by_source = {}
    if explicit_kws_dict:
        kws_by_source["Explícitas"] = filtrar_por_idioma(explicit_kws_dict, 'es')
    for model_name, kws in modelos_keywords_dict.items():
        kws_by_source[model_name] = filtrar_por_idioma(kws, 'es')
    print(f"\n Tiempo total del modelo en '{pdf_file_name}':")
    for modelo, tiempo in tiempos_modelos_dict.items():
        print(f"  {modelo: <20} → {tiempo:.2f} segundos")
    print(f"\n--- Palabras clave del modelo para '{pdf_file_name}' ---")
    for modelo, keywords in modelos_keywords_dict.items():
        print(f"\nModelo: {modelo}")
        print("Palabras clave extraídas (RAW):")
        print(list(keywords.keys()))
        print("Palabras clave (limpias):")
        for kw in keywords.keys():
            if es_keyword_valida(kw):
                print(f"  - {kw}")

    all_extracted_kws_by_source = {}
    if explicit_kws_dict:
        all_extracted_kws_by_source["Explícitas"] = filtrar_por_idioma(explicit_kws_dict, 'es')
    for model_name, kws in modelos_keywords_dict.items():
        all_extracted_kws_by_source[model_name] = filtrar_por_idioma(kws, 'es')

    agrupadas_por_lema = agrupa_similares([
        kw for kws_dict in all_extracted_kws_by_source.values() for kw in kws_dict.keys()
    ])
    lemas_unicos = [
    lema for lema in agrupadas_por_lema.keys()
    if es_keyword_valida(lema)
]
    if lemas_unicos:
        lemas_unicos = [lema for lema in lemas_unicos if es_keyword_valida(lema)]
    print(f"\n--- {pdf_file_name}: Similitud semántica con vocabulario UNESCO (Umbral: {sim_umbral}) ---")
    
    matches_unesco_dict = get_semantic_matches(
        lemas_unicos,
        emb_vocab_unesco,
        vocab_unesco_terms_norm,
        terminos_unesco_cache,
        embedding_model,
        umbral=sim_umbral
    )
    for k, (vocab_term, score) in matches_unesco_dict.items():
        print(f"  - '{k}' -> '{vocab_term}' ({score:.2f})")
    print(f"\n--- {pdf_file_name}: Similitud semántica con vocabulario TESEO (Umbral: {sim_umbral}) ---")
    matches_teseo_dict = get_semantic_matches(
        lemas_unicos,
        emb_vocab_teseo,
        vocab_teseo_terms_norm,
        terminos_teseo_cache,
        embedding_model,
        umbral=sim_umbral
    )
    logging.debug("\nCoincidencias semánticas UNESCO:")
    for lema, (termino_vocab, score) in matches_unesco_dict.items():
        print(f" - '{lema}' -> '{termino_vocab}' ({score:.2f})")

    for k, (vocab_term, score) in matches_unesco_dict.items():
        print(f"  - '{k}' -> '{vocab_term}' ({score:.2f})")

    print(f"\n--- {pdf_file_name}: Comparación de Modelos vs Descriptores manuales de UNESCO (Excel) ---")
    print(f"(Referencia UNESCO: {len(ref_unesco)} descriptores manuales)")
    comparar_modelos_con_referencias(all_extracted_kws_by_source, ref_unesco)
    
    print(f"\n--- {pdf_file_name}: Comparación de Modelos vs Descriptores manuales de TESEO (Excel) ---")
    print(f"(Referencia TESEO: {len(ref_teseo)} descriptores manuales)")
    comparar_modelos_con_referencias(all_extracted_kws_by_source, ref_teseo)
    
    
    unesco_original_matched_terms = sorted(
        {vocab_term for _, (vocab_term, _) in matches_unesco_dict.items()}
    )
    teseo_original_matched_terms = sorted(
        {vocab_term for _, (vocab_term, _) in matches_teseo_dict.items()}
    )

    descriptores_excel_validados_teseo_orig = []
    for desc in descriptores_excel_map.get(file_name_sin_ext, []):
        norm = normalizar(desc)
        lema = lematizar_kw(norm)
        if lema in terminos_teseo_cache:
            descriptores_excel_validados_teseo_orig.append(terminos_teseo_cache[lema])

    descriptores_unesco_validados_orig = []
    for desc in descriptores_unesco_map.get(file_name_sin_ext, []):
        norm = normalizar(desc)
        lema = lematizar_kw(norm)
        if lema in terminos_unesco_cache:
            descriptores_unesco_validados_orig.append(terminos_unesco_cache[lema])

    processed_text_lemmas = " ".join([
        token.lemma_ for token in nlp(pdf_text_content[:1000000])
        if not token.is_stop and not token.is_punct and not token.is_space
    ])
    # Comparar explícitas con RDF (vía embeddings)
    lematizadas_explicitas = set([normalizar(k) for k in lematizar_lista_kws(explicit_kws_dict.keys())])
    unesco_match = set([normalizar(k) for k in matches_unesco_dict.keys()])
    teseo_match = set([normalizar(k) for k in matches_teseo_dict.keys()])
    lemas_explicitas = [
    lematizar_kw(normalizar(k))
    for k in explicit_kws_dict.keys()
]
    lemas_explicitas = [
    lematizar_kw(normalizar(k))
    for k in explicit_kws_dict.keys()
    if k and len(k.strip()) >= 3
]

    matches_explicitas_unesco = get_semantic_matches(
        lemas_explicitas,
        emb_vocab_unesco,
        vocab_unesco_terms_norm,
        terminos_unesco_cache,
        embedding_model,
        umbral=0.90
    )
    matches_explicitas_teseo = get_semantic_matches(
        lemas_explicitas,
        emb_vocab_teseo,
        vocab_teseo_terms_norm,
        terminos_teseo_cache,
        embedding_model,
        umbral=0.90
    )
    
    # Añadir explícitas a los conjuntos de coincidencias si hacen match
    unesco_match.update(normalizar(k) for k in matches_explicitas_unesco.keys())
    teseo_match.update(normalizar(k) for k in matches_explicitas_teseo.keys())
    
    logging.debug("Coincidencias explícitas con UNESCO:")
    for lema, (term, score) in matches_explicitas_unesco.items():
        print(f"  - '{lema}' -> '{term}' ({score:.2f})")
    
    logging.debug("Coincidencias explícitas con TESEO:")
    for lema, (term, score) in matches_explicitas_teseo.items():
        print(f"  - '{lema}' -> '{term}' ({score:.2f})")
    no_cubiertas = lematizadas_explicitas - unesco_match - teseo_match
    logging.debug(f"Lemas explícitas: {lematizadas_explicitas}")
    logging.debug(f"Coinciden UNESCO: {unesco_match}")
    logging.debug(f"Coinciden TESEO: {teseo_match}")
    logging.debug(f"No cubiertas: {no_cubiertas}")
    return {
        "archivo": pdf_file_name,
        "descriptores_excel_originales_en_teseo": "; ".join(sorted(list(set(descriptores_excel_validados_teseo_orig)))),
        "descriptores_unesco": "; ".join(sorted(set(descriptores_unesco_validados_orig))),
        "keywords_explicitas_originales_norm": "; ".join(sorted(explicit_kws_dict.keys())),
        "keywords_unesco_similitud_original": "; ".join(unesco_original_matched_terms),
        "keywords_teseo_similitud_original": "; ".join(teseo_original_matched_terms),
        "texto_procesado_lemas": processed_text_lemmas,
        "tiempos_modelos": tiempos_modelos_dict,
        "num_keywords_explicitas_no_vocab": len(no_cubiertas),
        "num_keywords_modelos_cubiertas_teseo": len(ref_teseo),
        "num_keywords_modelos_cubiertas_unesco": len(ref_unesco)
    }, ref_unesco, ref_teseo, all_extracted_kws_by_source, explicit_kws_dict, matches_explicitas_unesco, matches_explicitas_teseo
    

# Aplica extracción, evaluación y similitud a un solo PDF
def main_process_pdfs(RUTA_PDFS, RUTA_XLSX_TESEO, RUTA_XLSX_UNESCO,
                        RUTA_RDF_UNESCO, RUTA_RDF_TESEO, 
                        embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        sim_umbral=0.90, ngram_extraction_range=(1,2), top_n_extraction=15):
    logging.info("Cargando recursos iniciales...")
    terminos_unesco_cache = load_rdf_labels(RUTA_RDF_UNESCO)
    terminos_teseo_cache = load_teseo_terms(RUTA_RDF_TESEO)
    descriptores_teseo_map = load_excel_descriptors_teseo(RUTA_XLSX_TESEO)
    descriptores_unesco_map = load_excel_descriptors_unesco(RUTA_XLSX_UNESCO)
    vocab_unesco_terms_norm = list(terminos_unesco_cache.keys())
    vocab_teseo_terms_norm = list(terminos_teseo_cache.keys())
    logging.info(f"Cargando modelo de embedding para similitud: {embedding_model_name}...")
    try:
        embedding_model_sim = SentenceTransformer(embedding_model_name)
        kb_instance = KeyBERT(model=embedding_model_sim)
        emb_vocab_unesco = embedding_model_sim.encode(vocab_unesco_terms_norm, convert_to_tensor=True, show_progress_bar=False)
        emb_vocab_teseo = embedding_model_sim.encode(vocab_teseo_terms_norm, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        logging.error(f"No se pudo cargar el modelo de embedding {embedding_model_name}: {e}. Saliendo.")
        return

    resultados_docs = []
    metricas_unesco_totales = defaultdict(list)
    metricas_teseo_totales = defaultdict(list)
    num_pdfs_procesados = 0
    pdfs_a_procesar = [f for f in os.listdir(RUTA_PDFS) if f.lower().endswith(".pdf")]
    total_pdfs = len(pdfs_a_procesar)
    logging.info(f"Se procesarán {total_pdfs} archivos PDF.")
    count_con_keywords_explicitas = 0
    for pdf_file in pdfs_a_procesar:
        num_pdfs_procesados += 1
        logging.info(f"--- Iniciando procesamiento PDF {num_pdfs_procesados}/{total_pdfs}: {pdf_file} ---")
        full_pdf_path = os.path.join(RUTA_PDFS, pdf_file)
        try:
            pdf_text_content = extract_text_from_pdf(full_pdf_path)
            if not pdf_text_content or not pdf_text_content.strip():
                logging.warning(f"PDF {pdf_file} sin texto extraído o vacío. Saltando.")
                continue
            pdf_text_content = limpiar_texto(pdf_text_content)
            resultado_pdf, ref_unesco, ref_teseo, kws_by_source, explicit_kws_dict, matches_explicitas_unesco, matches_explicitas_teseo = process_single_pdf(
                pdf_file_name=pdf_file,
                full_pdf_path=full_pdf_path,
                pdf_text_content=pdf_text_content,
                descriptores_excel_map=descriptores_teseo_map,
                terminos_unesco_cache=terminos_unesco_cache,
                terminos_teseo_cache=terminos_teseo_cache,
                vocab_unesco_terms_norm=vocab_unesco_terms_norm,
                vocab_teseo_terms_norm=vocab_teseo_terms_norm,
                embedding_model=embedding_model_sim,
                emb_vocab_unesco=emb_vocab_unesco,
                emb_vocab_teseo=emb_vocab_teseo,
                kb_instance=kb_instance,
                descriptores_unesco_map=descriptores_unesco_map,
                sim_umbral=sim_umbral,
                ngram_range=ngram_extraction_range,
                top_n_extraction=top_n_extraction
            )
            if explicit_kws_dict and isinstance(explicit_kws_dict, dict):
                # Lematizar y normalizar explícitas
                kws_explicitas_lemas = {
                    lematizar_kw(normalizar(k)): "Explícita" for k in explicit_kws_dict.keys()
                }
                kws_by_source["Explícitas"] = kws_explicitas_lemas
            
                logging.debug("kws_by_source justo antes de comparar UNESCO:", kws_by_source.keys())
                logging.debug("Keywords explícitas enviadas a comparación UNESCO:", kws_by_source.get("Explícitas"))
                logging.debug("Claves explícitas dict:", list(explicit_kws_dict.keys()))
            
                # Comparación contra descriptores manuales del Excel
                metricas_unesco = comparar_modelos_con_referencias(kws_by_source, ref_unesco)
                metricas_teseo = comparar_modelos_con_referencias(kws_by_source, ref_teseo)

                # Guardar métricas de comparación con Excel (no RDF)
                for modelo, vals in metricas_unesco.items():
                    for metrica, valor in vals.items():
                        metricas_unesco_totales[(modelo, metrica)].append(valor)
                for modelo, vals in metricas_teseo.items():
                    for metrica, valor in vals.items():
                        metricas_teseo_totales[(modelo, metrica)].append(valor)
            
            if resultado_pdf:
                if resultado_pdf.get("keywords_explicitas_originales_norm"):
                    count_con_keywords_explicitas += 1
            
                file_name = pdf_file[:-4].lower()
                resultado_pdf["labels_teseo"] = descriptores_teseo_map.get(file_name, [])
                resultado_pdf["labels_unesco"] = descriptores_unesco_map.get(file_name, [])
            
                # Guardar coincidencias explícitas con RDF
                resultado_pdf["explicitas_unesco"] = [str(v[0]) for _, v in matches_explicitas_unesco.items()]
                resultado_pdf["explicitas_teseo"] = [str(v[0]) for _, v in matches_explicitas_teseo.items()]
            
                # Extraer solo los matches de modelo que no fueron explícitas
                modelo_unesco = [k.strip() for k in resultado_pdf.get("keywords_unesco_similitud_original", "").split(";") if k.strip()]
                resultado_pdf["modelos_unesco"] = modelo_unesco
                modelo_teseo = [k.strip() for k in resultado_pdf.get("keywords_teseo_similitud_original", "").split(";") if k.strip()]
                resultado_pdf["modelos_teseo"] = modelo_teseo
            
                logging.debug("matches_explicitas_unesco:", matches_explicitas_unesco)
                logging.debug("matches_explicitas_teseo:", matches_explicitas_teseo)
            
                resultados_docs.append(resultado_pdf)

        except Exception as e:
            logging.error(f"Error mayor procesando {pdf_file}: {e}")
            logging.error(traceback.format_exc())
        logging.info(f"--- Finalizado procesamiento PDF {num_pdfs_procesados}/{total_pdfs}: {pdf_file} ---")
    if not resultados_docs:
        logging.info("No se procesaron documentos o no hubo resultados para guardar.")
        return
    df_result = pd.DataFrame(resultados_docs)
    logging.debug("Matching keywords vs descriptores por PDF:")
    for idx, row in df_result.iterrows():
        archivo = row["archivo"]
        kws_unesco = row.get("keywords_unesco_similitud_original", "")
        kws_teseo = row.get("keywords_teseo_similitud_original", "")
        desc_unesco = row.get("labels_unesco", [])
        desc_teseo = row.get("labels_teseo", [])
    
        print(f"\n{archivo}")
        print(f" - Keywords UNESCO extraídas: {kws_unesco}")
        print(f" - Descriptores UNESCO esperados: {desc_unesco}")
        print(f" - Keywords TESEO extraídas: {kws_teseo}")
        print(f" - Descriptores TESEO esperados: {desc_teseo}")
    logging.debug("Resumen por PDF - Keywords explícitas fuera de vocabulario:")
    df_result["num_validas_vocab"] = df_result.apply(
        lambda row: len(set(
            row.get("keywords_unesco_similitud_original", "").split(";") +
            row.get("keywords_teseo_similitud_original", "").split(";")
        )) if pd.notna(row.get("keywords_unesco_similitud_original")) or pd.notna(row.get("keywords_teseo_similitud_original")) else 0,
        axis=1
    )
    
    df_validos = df_result[df_result["num_validas_vocab"] > 0]
    cuenta_pdfs_validos = df_validos.shape[0]
    media_validas = df_validos["num_validas_vocab"].mean()
    mediana_validas = df_validos["num_validas_vocab"].median()
    print(f"PDFs con al menos una keyword válida UNESCO o TESEO: {cuenta_pdfs_validos}")
    print(f"Media de keywords válidas por PDF: {media_validas:.2f}")
    print(f"Mediana de keywords válidas por PDF: {mediana_validas}")
    exportar_json_completo(resultados_docs, json_path="resultados/resultados_completo.json")
    exportar_json_entrenamiento(
        df_resultado=df_result,
        terminos_unesco_cache=terminos_unesco_cache,
        terminos_teseo_cache=terminos_teseo_cache,
        ruta_json="resultados/dataset_entrenamiento_completo.json"
    )
    logging.info(f"Archivo JSON guardado en 'resultados/resultados_completo.json' con {len(resultados_docs)} registros.")

        # PDFs sin texto y PDFs no presentes en Excels

    pdfs_vacios = []
    pdfs_sin_excel = []

    df_excel_1 = pd.read_excel(RUTA_XLSX_TESEO, engine="openpyxl")
    df_excel_2 = pd.read_excel(RUTA_XLSX_UNESCO, engine="openpyxl")
    nombres_excel_1 = set(str(f).strip().lower().replace('.pdf','') for f in df_excel_1["Fichero"].dropna())
    nombres_excel_2 = set(str(f).strip().lower().replace('.pdf','') for f in df_excel_2["Fichero"].dropna())

    for pdf_file in os.listdir(RUTA_PDFS):
        if not pdf_file.lower().endswith(".pdf"):
            continue
        full_pdf_path = os.path.join(RUTA_PDFS, pdf_file)
        texto = extract_text_from_pdf(full_pdf_path)
        file_name_no_ext = pdf_file[:-4].lower()
        if not es_texto_valido(texto):
            pdfs_vacios.append(pdf_file)
        if file_name_no_ext not in nombres_excel_1 and file_name_no_ext not in nombres_excel_2:
            pdfs_sin_excel.append(pdf_file)
    if pdfs_vacios:
        with open("pdfs_sin_texto.txt", "w", encoding="utf-8") as f:
            for pdf in pdfs_vacios:
                f.write(f"{pdf}\n")
        print(f"PDFs sin texto guardados en pdfs_sin_texto.txt")

    if pdfs_sin_excel:
        with open("pdfs_sin_excel.txt", "w", encoding="utf-8") as f:
            for pdf in pdfs_sin_excel:
                f.write(f"{pdf}\n")
        print(f"PDFs sin metadatos en excels guardados en pdfs_sin_excel.txt")

    # Bloque para exportar _unesco y _teseo
    df_result["labels_unesco"] = df_result["descriptores_unesco"].fillna("").apply(
        lambda x: [et.strip() for et in x.split(";") if et.strip()]
    )
    df_result["labels_teseo"] = df_result["descriptores_excel_originales_en_teseo"].fillna("").apply(
        lambda x: [et.strip() for et in x.split(";") if et.strip()]
    )

    # Métricas promedio
    print("\n\n--- Métricas promedio por modelo ---")
    def construir_tabla_metricas(totales_dict, conjunto):
        resumen = []
        modelos = sorted(set(m for (m, _) in totales_dict.keys()))
        for modelo in modelos:
            p_vals = totales_dict.get((modelo, "P"), [])
            r_vals = totales_dict.get((modelo, "R"), [])
            f1_vals = totales_dict.get((modelo, "F1"), [])
    
            # Verifica longitudes iguales
            if not (len(p_vals) == len(r_vals) == len(f1_vals)):
                logging.warning(f"Diferentes cantidades de métricas para el modelo {modelo}: "
                                f"P={len(p_vals)}, R={len(r_vals)}, F1={len(f1_vals)}")
                continue
    
            p_media = np.mean(p_vals) if p_vals else 0
            r_media = np.mean(r_vals) if r_vals else 0
            f1_media = np.mean(f1_vals) if f1_vals else 0
    
            p_mediana = np.median(p_vals) if p_vals else 0
            r_mediana = np.median(r_vals) if r_vals else 0
            f1_mediana = np.median(f1_vals) if f1_vals else 0
    
            p_std = np.std(p_vals) if p_vals else 0
            r_std = np.std(r_vals) if r_vals else 0
            f1_std = np.std(f1_vals) if f1_vals else 0
    
            p_max = np.max(p_vals) if p_vals else 0
            r_max = np.max(r_vals) if r_vals else 0
            f1_max = np.max(f1_vals) if f1_vals else 0
    
            p_min = np.min(p_vals) if p_vals else 0
            r_min = np.min(r_vals) if r_vals else 0
            f1_min = np.min(f1_vals) if f1_vals else 0
    
            n_muestras = len(p_vals)
    
            resumen.append({
                "Modelo": modelo,
                "Conjunto": conjunto,
                "n": n_muestras,
    
                "P_media": round(p_media, 3),
                "P_mediana": round(p_mediana, 3),
                "P_std": round(p_std, 3),
                "P_min": round(p_min, 3),
                "P_max": round(p_max, 3),
    
                "R_media": round(r_media, 3),
                "R_mediana": round(r_mediana, 3),
                "R_std": round(r_std, 3),
                "R_min": round(r_min, 3),
                "R_max": round(r_max, 3),
    
                "F1_media": round(f1_media, 3),
                "F1_mediana": round(f1_mediana, 3),
                "F1_std": round(f1_std, 3),
                "F1_min": round(f1_min, 3),
                "F1_max": round(f1_max, 3),
            })
        return resumen
    
    def imprimir_metricas_promedio(totales_dict, conjunto):
        print(f"\n--- Promedio de métricas ({conjunto}) ---")
        modelos = sorted(set(m for (m, _) in totales_dict.keys()))
        for modelo in modelos:
            p_vals = totales_dict.get((modelo, "P"), [])
            r_vals = totales_dict.get((modelo, "R"), [])
            f1_vals = totales_dict.get((modelo, "F1"), [])
            p = sum(p_vals) / len(p_vals) if p_vals else 0
            r = sum(r_vals) / len(r_vals) if r_vals else 0
            f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0
            print(f"Modelo: {modelo: <15} - P: {p:.2f}, R: {r:.2f}, F1: {f1:.2f}")

    imprimir_metricas_promedio(metricas_unesco_totales, "UNESCO")
    imprimir_metricas_promedio(metricas_teseo_totales, "TESEO")
    tabla_unesco = construir_tabla_metricas(metricas_unesco_totales, "UNESCO")
    tabla_teseo = construir_tabla_metricas(metricas_teseo_totales, "TESEO")
    # Exportar resumen combinado con medias y medianas
    df_metricas_completo = pd.concat([pd.DataFrame(tabla_unesco), pd.DataFrame(tabla_teseo)], ignore_index=True)
    # Crear DataFrame para métricas explícitas UNESCO
    if ("Explícitas", "P") in metricas_unesco_totales:
        df_explicitas_unesco = pd.DataFrame([{
            "Modelo": "Explícitas",
            "Conjunto": "UNESCO",
            "n": len(metricas_unesco_totales[("Explícitas", "P")]),
            "P_media": np.mean(metricas_unesco_totales[("Explícitas", "P")]),
            "R_media": np.mean(metricas_unesco_totales[("Explícitas", "R")]),
            "F1_media": np.mean(metricas_unesco_totales[("Explícitas", "F1")]),
            "P_mediana": np.median(metricas_unesco_totales[("Explícitas", "P")]),
            "R_mediana": np.median(metricas_unesco_totales[("Explícitas", "R")]),
            "F1_mediana": np.median(metricas_unesco_totales[("Explícitas", "F1")])
        }])
    else:
        df_explicitas_unesco = pd.DataFrame()
    
    # Crear DataFrame para métricas explícitas TESEO
    if ("Explícitas", "P") in metricas_teseo_totales:
        df_explicitas_teseo = pd.DataFrame([{
            "Modelo": "Explícitas",
            "Conjunto": "TESEO",
            "n": len(metricas_teseo_totales[("Explícitas", "P")]),
            "P_media": np.mean(metricas_teseo_totales[("Explícitas", "P")]),
            "R_media": np.mean(metricas_teseo_totales[("Explícitas", "R")]),
            "F1_media": np.mean(metricas_teseo_totales[("Explícitas", "F1")]),
            "P_mediana": np.median(metricas_teseo_totales[("Explícitas", "P")]),
            "R_mediana": np.median(metricas_teseo_totales[("Explícitas", "R")]),
            "F1_mediana": np.median(metricas_teseo_totales[("Explícitas", "F1")])
        }])
    else:
        df_explicitas_teseo = pd.DataFrame()
    # Crear archivo si no existe
    if not os.path.exists(excel_resumen_metricas):
        # Crea un archivo Excel vacío con una hoja Inicial, evitar errores si el archivo no existe
        pd.DataFrame({"Inicialización": ["Archivo creado"]}).to_excel(
            excel_resumen_metricas, sheet_name="Inicial", index=False, engine="openpyxl"
        )
    # Guardar en hojas de Excel si hay datos
    try:
        with pd.ExcelWriter(excel_resumen_metricas, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            if not df_explicitas_unesco.empty:
                df_explicitas_unesco.to_excel(writer, sheet_name="Explícitas_UNESCO", index=False)
            if not df_explicitas_teseo.empty:
                df_explicitas_teseo.to_excel(writer, sheet_name="Explícitas_TESEO", index=False)
        print("Exportadas métricas explícitas a hojas de Excel.")
    except Exception as e:
        logging.warning(f"Error al escribir métricas explícitas en Excel: {e}")

    print(f"Resumen de métricas también exportado a '{excel_resumen_metricas}'.")
    print("\n--- Promedio de tiempos por modelo ---")
    tiempos_acumulados = defaultdict(list)

    for resultado in resultados_docs:
        tiempos_path = resultado.get("tiempos_modelos")
        if tiempos_path:
            for modelo, tiempo in tiempos_path.items():
                tiempos_acumulados[modelo].append(tiempo)

    tiempos_promedio = {modelo: sum(valores) / len(valores) for modelo, valores in tiempos_acumulados.items()}

    for modelo, tiempo in sorted(tiempos_promedio.items(), key=lambda x: x[1]):
        print(f"{modelo:<20} → {tiempo:.2f} segundos promedio")

    # Gráfico de tiempos promedio
    modelos = list(tiempos_promedio.keys())
    tiempos = list(tiempos_promedio.values())

    plt.figure(figsize=(10, 5))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    barras = plt.bar(modelos, tiempos, color=colores[:len(modelos)])
    for barra in barras:
        y = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, y + 0.1, f"{y:.2f}s",
                 ha='center', va='bottom', fontsize=8)

    plt.ylabel("Tiempo promedio en (s)")
    plt.title("Tiempo promedio por modelo de extracción")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_file = os.path.join(RUTA_RESULTADOS, "tiempo_promedio_modelos.png")
    plt.savefig(output_file)
    plt.show()
    print(f"Gráfico de tiempos exportado a {output_file}")


# Exportar tiempos por modelo por PDF
    tiempos_modelo_pdf = []
    for r in resultados_docs:
        archivo = r["archivo"]
        tiempos = r.get("tiempos_modelos", {})
        for modelo, tiempo in tiempos.items():
            tiempos_modelo_pdf.append({
                "PDF": archivo,
                "Modelo": modelo,
                "Tiempo (s)": tiempo
            })
    
    df_tiempos = pd.DataFrame(tiempos_modelo_pdf)
    if not df_tiempos.empty:
        df_tiempos.sort_values(by=["Modelo", "PDF"], inplace=True)
    pdfs_exitosos = len(resultados_docs)
    with pd.ExcelWriter("resultados/resumen_metricas.xlsx", engine="openpyxl", mode="w") as writer:
        if not df_metricas_completo.empty:
            df_metricas_completo.to_excel(writer, sheet_name="Medias_y_Medianas", index=False)
        if not df_tiempos.empty:
            df_tiempos.to_excel(writer, sheet_name="TiemposModelos", index=False)
    
    
    print(f"\nTotal de PDFs con palabras clave explícitas detectadas: {count_con_keywords_explicitas} de {pdfs_exitosos}")
    cobertura = (count_con_keywords_explicitas / pdfs_exitosos) * 100 if pdfs_exitosos > 0 else 0
    print(f"Cobertura de PDFs con keywords explícitas: {cobertura:.2f}%")
    print(f"\nTotal de PDFs procesados con éxito: {pdfs_exitosos} de {total_pdfs}")
    return tabla_unesco, tabla_teseo

# Sistema de batches y checkpoints
CHECKPOINT_JSON = os.path.join(RUTA_RESULTADOS, "resultados_batchs.json")
BATCH_SIZE = 20

def cargar_checkpoint(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []

def main_process_pdfs_batch(
        RUTA_PDFS, RUTA_XLSX_TESEO, RUTA_XLSX_UNESCO,
        RUTA_RDF_UNESCO, RUTA_RDF_TESEO, 
        embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        sim_umbral=0.90, ngram_extraction_range=(1,2), top_n_extraction=15, batch_size=10):

    logging.info("Cargando recursos iniciales...")
    terminos_unesco_cache = load_rdf_labels(RUTA_RDF_UNESCO)
    terminos_teseo_cache = load_teseo_terms(RUTA_RDF_TESEO)
    descriptores_teseo_map = load_excel_descriptors_teseo(RUTA_XLSX_TESEO)
    descriptores_unesco_map = load_excel_descriptors_unesco(RUTA_XLSX_UNESCO)
    vocab_unesco_terms_norm = list(terminos_unesco_cache.keys())
    vocab_teseo_terms_norm = list(terminos_teseo_cache.keys())
    logging.info(f"Cargando modelo de embedding para similitud: {embedding_model_name}...")
    try:
        embedding_model_sim = SentenceTransformer(embedding_model_name)
        kb_instance = KeyBERT(model=embedding_model_sim)
        emb_vocab_unesco = embedding_model_sim.encode(vocab_unesco_terms_norm, convert_to_tensor=True, show_progress_bar=False)
        emb_vocab_teseo = embedding_model_sim.encode(vocab_teseo_terms_norm, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        logging.error(f"No se pudo cargar el modelo de embedding {embedding_model_name}: {e}. Saliendo.")
        return

    # Cargar progreso anterior si existe
    resultados_docs = cargar_checkpoint(CHECKPOINT_JSON)
    ya_procesados = set([r['archivo'] for r in resultados_docs])
    pdfs_a_procesar = [f for f in os.listdir(RUTA_PDFS) if f.lower().endswith(".pdf") and f not in ya_procesados]
    total_pdfs = len(pdfs_a_procesar) + len(ya_procesados)
    print(f"{len(ya_procesados)} PDFs ya procesados, quedan {len(pdfs_a_procesar)} de {total_pdfs}")

    metricas_unesco_totales = defaultdict(list)
    metricas_teseo_totales = defaultdict(list)

    for i in range(0, len(pdfs_a_procesar), batch_size):
        batch = pdfs_a_procesar[i:i+batch_size]
        for pdf_file in batch:
            try:
                print(f"\nProcesando PDF: {pdf_file}")
                full_pdf_path = os.path.join(RUTA_PDFS, pdf_file)
                pdf_text_content = extract_text_from_pdf(full_pdf_path)
                if not pdf_text_content or not pdf_text_content.strip():
                    logging.warning(f"PDF {pdf_file} sin texto extraído o vacío. Saltando.")
                    continue
                pdf_text_content = limpiar_texto(pdf_text_content)
                resultado_pdf, ref_unesco, ref_teseo, kws_by_source, explicit_kws_dict, matches_explicitas_unesco, matches_explicitas_teseo = process_single_pdf(
                    pdf_file_name=pdf_file,
                    full_pdf_path=full_pdf_path,
                    pdf_text_content=pdf_text_content,
                    descriptores_excel_map=descriptores_teseo_map,
                    terminos_unesco_cache=terminos_unesco_cache,
                    terminos_teseo_cache=terminos_teseo_cache,
                    vocab_unesco_terms_norm=vocab_unesco_terms_norm,
                    vocab_teseo_terms_norm=vocab_teseo_terms_norm,
                    embedding_model=embedding_model_sim,
                    emb_vocab_unesco=emb_vocab_unesco,
                    emb_vocab_teseo=emb_vocab_teseo,
                    kb_instance=kb_instance,
                    descriptores_unesco_map=descriptores_unesco_map,
                    sim_umbral=sim_umbral,
                    ngram_range=ngram_extraction_range,
                    top_n_extraction=top_n_extraction
                )
                if resultado_pdf:
                    if explicit_kws_dict and isinstance(explicit_kws_dict, dict):
                        kws_explicitas_lemas = {
                            lematizar_kw(normalizar(k)): "Explícita" for k in explicit_kws_dict.keys()
                        }
                        kws_by_source["Explícitas"] = kws_explicitas_lemas

                        metricas_unesco = comparar_modelos_con_referencias(kws_by_source, ref_unesco)
                        metricas_teseo = comparar_modelos_con_referencias(kws_by_source, ref_teseo)

                        for modelo, vals in metricas_unesco.items():
                            for metrica, valor in vals.items():
                                metricas_unesco_totales[(modelo, metrica)].append(valor)
                        for modelo, vals in metricas_teseo.items():
                            for metrica, valor in vals.items():
                                metricas_teseo_totales[(modelo, metrica)].append(valor)
                    file_name = resultado_pdf["archivo"][:-4].lower()
                    resultado_pdf["labels_teseo"] = descriptores_teseo_map.get(file_name, [])
                    resultado_pdf["labels_unesco"] = descriptores_unesco_map.get(file_name, [])
                    resultado_pdf["explicitas_unesco"] = [str(v[0]) for _, v in matches_explicitas_unesco.items()]
                    resultado_pdf["explicitas_teseo"] = [str(v[0]) for _, v in matches_explicitas_teseo.items()]
                    modelo_unesco = [k.strip() for k in resultado_pdf.get("keywords_unesco_similitud_original", "").split(";") if k.strip()]
                    resultado_pdf["modelos_unesco"] = modelo_unesco
                    modelo_teseo = [k.strip() for k in resultado_pdf.get("keywords_teseo_similitud_original", "").split(";") if k.strip()]
                    resultado_pdf["modelos_teseo"] = modelo_teseo
                    resultados_docs.append(resultado_pdf)
            except Exception as e:
                logging.error(f"Error procesando {pdf_file}: {e}\n{traceback.format_exc()}")
                continue

        # Guardado del progreso batch
        with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
            json.dump(resultados_docs, f, indent=2, ensure_ascii=False)
        print(f"\nCheckpoint guardado tras {len(resultados_docs)} PDFs.")

        # Limpieza de memoria tras cada batch
        torch.cuda.empty_cache()  # Limpia memoria de la GPU
        gc.collect()              # Limpia objetos no referenciados en RAM
        
    # Métricas y guardado final
    if not resultados_docs:
        print("No se han procesado PDFs. Revisa los logs.")
        logging.error("No se han procesado PDFs.")
        return

    print("Calculando métricas y guardando resultados finales...")
    df_result = pd.DataFrame(resultados_docs)
    exportar_json_completo(resultados_docs, json_output_path)
    exportar_json_entrenamiento(df_result, terminos_unesco_cache, terminos_teseo_cache)
    print("¡Resultados y dataset de entrenamiento exportados!")
    print("Proceso completo finalizado.")


# Main
if __name__ == "__main__":
    main_process_pdfs_batch(
        RUTA_PDFS, RUTA_XLSX_TESEO, RUTA_XLSX_UNESCO,
        RUTA_RDF_UNESCO, RUTA_RDF_TESEO,
        embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        sim_umbral=0.90,
        ngram_extraction_range=(1,2),
        top_n_extraction=15,
        batch_size=BATCH_SIZE
    )



