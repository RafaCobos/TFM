import streamlit as st
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from docx import Document
import fitz
from rdflib import Graph
from rdflib.namespace import SKOS
import re
import warnings
from flashtext import KeywordProcessor
from langdetect import detect, LangDetectException, DetectorFactory
from urllib.parse import unquote

# Importar las configuraciones desde el archivo config_dominios.py
from config_dominios import DOMINIOS_CIENCIAS_SOCIALES_ZS, DOMINIOS_TECNICOS_CANDIDATOS_ZS, ETIQUETAS_MIGRACION

# Resultados de langdetect sean consistentes
DetectorFactory.seed = 0

# Ignorar advertencias futuras
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración y estilos
st.set_page_config(page_title="Generador de palabras clave automáticas para Tesis Doctorales", layout="wide")
st.title("Clasificación automática de Tesis Doctorales")

# Definición de funciones
@st.cache_resource
def cargar_vocabularios_rdf():
    """Carga el vocabulario de TESEO desde un archivo RDF."""
    grafo_teseo = Graph()
    grafo_teseo.parse("data/teseo.rdf", format="xml")
    procesador_teseo = KeywordProcessor(case_sensitive=False)
    for _, _, o in grafo_teseo.triples((None, SKOS.prefLabel, None)):
        if o.language == "es" or o.language is None:
            procesador_teseo.add_keyword(str(o).strip())
    return grafo_teseo, procesador_teseo

@st.cache_resource
def crear_mapa_dominios_desde_rdf(_grafo):
    """Crea un diccionario que mapea cada etiqueta de TESEO a su dominio principal desde el RDF."""
    mapa_dominios = {}
    for s, p, o in _grafo.triples((None, SKOS.prefLabel, None)):
        try:
            fragmento = s.split('#')[-1]
            if fragmento:
                dominio = unquote(fragmento)
                label = str(o).strip()
                mapa_dominios[label.upper()] = dominio.upper()
        except Exception:
            continue
    return mapa_dominios

@st.cache_resource
def cargar_modelo_TESEO():
    """Carga el modelo de clasificación para TESEO (basado en Transformers)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("modelos/TESEO_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("modelos/TESEO_model")
    model.to(device)
    id_to_label_map = joblib.load("modelos/labels_teseo.pkl")
    thresholds = np.load("modelos/thresholds_teseo.npy")
    return tokenizer, model, id_to_label_map, thresholds, device

@st.cache_resource
def cargar_modelo_unesco():
    """Carga el modelo de clasificación para UNESCO (basado en TF-IDF)."""
    vectorizer = joblib.load("modelos/vectorizer_unesco.pkl")
    classifier = joblib.load("modelos/clf_unesco.pkl")
    mlb = joblib.load("modelos/labels_unesco.pkl")
    return vectorizer, classifier, mlb

@st.cache_resource
def cargar_clasificador_dominio():
    """Carga el modelo de zero-shot para la clasificación de dominio general."""
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    return classifier

def detectar_idioma(texto: str) -> str:
    """Detecta el idioma de un fragmento de texto de forma más robusta."""
    try:
        texto_muestra = texto[1000:3000]
        if not texto_muestra.strip():
            texto_muestra = texto[:1000]
        return detect(texto_muestra)
    except (LangDetectException, Exception):
        try:
            return detect(texto)
        except:
            return "desconocido"

def extraer_texto(file):
    """Extrae texto de archivos PDF o DOCX."""
    texto_completo = ""
    if file.type == "application/pdf":
        try:
            with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                texto_completo = "".join(page.get_text() for page in doc)
        except Exception as e:
            st.error(f"Error al leer el PDF con PyMuPDF: {e}")
            return None
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        texto_completo = "\n".join([p.text for p in doc.paragraphs])
    elif file.type.startswith("text"):
        texto_completo = str(file.read(), 'utf-8', errors='ignore')
    return texto_completo

def extraer_fragmento_representativo(texto: str, inicio=3000, longitud=3000):
    """Extrae un fragmento del cuerpo del texto para evitar introducciones y resúmenes."""
    fragmento = texto[inicio:inicio + longitud]
    if not fragmento.strip():
        fragmento = texto[:longitud]
    return fragmento

def dividir_en_chunks(texto, max_palabras=400, solape=50):
    palabras = re.split(r'\s+', texto)
    chunks = []
    start = 0
    while start < len(palabras):
        end = start + max_palabras
        chunks.append(" ".join(palabras[start:end]))
        start += max_palabras - solape
    return chunks

def clasificar_TESEO_con_chunks(texto, tokenizer, model, thresholds, id2label, device):
    MINI_BATCH_SIZE = 16
    chunks = dividir_en_chunks(texto)
    if not chunks: return []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(chunks), MINI_BATCH_SIZE):
            batch_chunks = chunks[i:i + MINI_BATCH_SIZE]
            inputs = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs_batch = torch.sigmoid(logits)
            all_probs.append(probs_batch.cpu())
    full_probs_tensor = torch.cat(all_probs, dim=0)
    pred_promedio = full_probs_tensor.mean(dim=0).numpy()
    
    return [(id2label[i], float(pred_promedio[i])) for i in range(len(pred_promedio)) if pred_promedio[i] > thresholds[i]]

def clasificar_UNESCO_con_tfidf(texto, vectorizer, model, mlb):
    texto_tfidf = vectorizer.transform([texto])
    etiquetas_con_confianza = []
    if hasattr(model, "predict_proba"):
        probabilidades = model.predict_proba(texto_tfidf)
        for i in range(len(mlb.classes_)):
            if isinstance(probabilidades, list):
                confianza = probabilidades[i][0, 1] if probabilidades[i].shape[1] > 1 else probabilidades[i][0, 0]
            else:
                confianza = probabilidades[0, i]

            etiqueta = mlb.classes_[i]
            etiquetas_con_confianza.append((etiqueta, float(confianza)))
            
    return etiquetas_con_confianza

def clasificar_dominio_general(texto_fragmento, clasificador, dominios_posibles, umbral=0.3):
    """Clasifica el dominio, devolviendo solo las etiquetas que superan el umbral."""
    if not texto_fragmento:
        return [], []

    resultado = clasificador(texto_fragmento, dominios_posibles, multi_label=True)

    dominios_relevantes, scores_relevantes = [], []
    for label, score in zip(resultado['labels'], resultado['scores']):
        if score >= umbral:
            dominios_relevantes.append(label)
            scores_relevantes.append(score)
    return dominios_relevantes, scores_relevantes

def detectar_conceptos_rdf_en_texto(texto: str, keyword_processor: KeywordProcessor) -> list:
    """
    Detecta conceptos del vocabulario RDF (TESEO) que aparecen
    literalmente en un texto utilizando KeywordProcessor.
    """
    found_keywords = keyword_processor.extract_keywords(texto)
    return list(set(found_keywords))

# Ejecución

try:
    grafo_teseo, procesador_teseo = cargar_vocabularios_rdf()
    mapa_dominios_teseo = crear_mapa_dominios_desde_rdf(grafo_teseo)
    tokenizer_teseo, model_teseo, id2label_teseo, thresholds_teseo, device_teseo = cargar_modelo_TESEO()
    vectorizer_unesco, clf_unesco, mlb_unesco = cargar_modelo_unesco()
    clasificador_dominio = cargar_clasificador_dominio()
    st.success("¡Modelos cargados con éxito! La aplicación está lista.")
except Exception as e:
    st.error(f"Error crítico al cargar los modelos o vocabularios. La aplicación no puede continuar. Error: {e}")
    st.stop()

uploaded_file = st.file_uploader(label="Sube la tesis (.pdf o .docx):", type=["pdf", "docx"])

if uploaded_file:
    # Bloque de inicialización de variables
    resultados_filtrados_teseo = []
    resultados_filtrados_unesco = []
    conceptos_teseo_lower = set()

    with st.status("Procesando tesis...", expanded=True) as status:
        st.write("1/5 - Extrayendo texto del documento...")
        texto_completo = extraer_texto(uploaded_file)
        if not texto_completo or len(texto_completo) < 200:
            st.warning("El texto extraído está vacío o es demasiado corto para ser analizado.")
            status.update(label="Error en la extracción de texto.", state="error", expanded=True)
            st.stop()

        fragmento_texto = extraer_fragmento_representativo(texto_completo)
        st.write("2/5 - Detectando idioma...")
        idioma = detectar_idioma(texto_completo)
        st.write(f"Idioma detectado: **{idioma.upper()}**")

        st.write("3/5 - Analizando dominio temático general (multilingüe)...")
        dominios_relevantes_sociales_preds, confianzas_sociales_preds = clasificar_dominio_general(
            fragmento_texto, clasificador_dominio, DOMINIOS_CIENCIAS_SOCIALES_ZS, umbral=0.3
        )
        dominios_relevantes_tecnicos_preds, confianzas_tecnicas_preds = clasificar_dominio_general(
            fragmento_texto, clasificador_dominio, DOMINIOS_TECNICOS_CANDIDATOS_ZS, umbral=0.3
        )

        es_tesis_tecnica = False
        max_conf_social = max(confianzas_sociales_preds) if confianzas_sociales_preds else 0.0
        max_conf_tecnica = max(confianzas_tecnicas_preds) if confianzas_tecnicas_preds else 0.0

        if max_conf_tecnica > 0.35 and (max_conf_tecnica > max_conf_social * 1.1 or max_conf_social == 0.0):
            es_tesis_tecnica = True

        st.write("4/5 - Ejecutando modelos de clasificación específicos (español)...")
        etiquetas_teseo_base_preds = []
        etiquetas_unesco_base_preds = []

        if idioma == 'es':
            etiquetas_teseo_base_preds = clasificar_TESEO_con_chunks(texto_completo, tokenizer_teseo, model_teseo, thresholds_teseo, id2label_teseo, device_teseo)
            etiquetas_unesco_base_preds = clasificar_UNESCO_con_tfidf(texto_completo, vectorizer_unesco, clf_unesco, mlb_unesco)

            st.write("5/5 - Aplicando lógica de filtrado adaptativa...")

            base_teseo_threshold_display = 0.5
            base_unesco_threshold_display = 0.4
            ETIQUETAS_MIGRACION_LOWER = {e.lower() for e in ETIQUETAS_MIGRACION}

            hay_dominio_social_migratorio_fuerte = False
            for dominio_social, conf_social in zip(dominios_relevantes_sociales_preds, confianzas_sociales_preds):
                if conf_social >= 0.80 and dominio_social in DOMINIOS_CIENCIAS_SOCIALES_ZS:
                    hay_dominio_social_migratorio_fuerte = True
                    break

            teseo_migracion_confidence = 0.0
            unesco_migracion_confidence = 0.0

            for etiqueta, prob in etiquetas_teseo_base_preds:
                if etiqueta.lower() in ETIQUETAS_MIGRACION_LOWER:
                    teseo_migracion_confidence = max(teseo_migracion_confidence, prob)

            for etiqueta, prob in etiquetas_unesco_base_preds:
                if etiqueta.lower() in ETIQUETAS_MIGRACION_LOWER:
                    unesco_migracion_confidence = max(unesco_migracion_confidence, prob)
            
            UMBRAL_RESCATE_TESEO_MIGRACION_INDIVIDUAL = 0.99
            UMBRAL_RESCATE_UNESCO_MIGRACION_INDIVIDUAL = 0.63

            teseo_predice_migracion_fuerte_especifica = teseo_migracion_confidence >= UMBRAL_RESCATE_TESEO_MIGRACION_INDIVIDUAL
            unesco_predice_migracion_fuerte_especifica = unesco_migracion_confidence >= UMBRAL_RESCATE_UNESCO_MIGRACION_INDIVIDUAL

            fuerte_senal_migracion_combinada = False
            if teseo_migracion_confidence >= 0.80 and unesco_migracion_confidence >= 0.60:
                fuerte_senal_migracion_combinada = True

            if es_tesis_tecnica and not (teseo_predice_migracion_fuerte_especifica or unesco_predice_migracion_fuerte_especifica):
                st.info("Tesis identificada con **dominio técnico**. Se aplicará un filtro contextual estricto.")
                MAPA_CLASIFICADOR_A_TESEO_DOMINIOS_TECNICOS = {
                    "Investigaciones en ingeniería civil, geotecnia, geología estructural o geociencias": ["INGENIERIA", "GEOLOGIA", "GEOGRAFIA FISICA"],
                    "Temas de salud pública, medicina clínica, biotecnología o biología molecular": ["BIOLOGIA", "MEDICINA", "FARMACIA", "VETERINARIA", "BIOTECNOLOGIA"],
                    "Investigación en física teórica, química inorgánica, matemáticas aplicadas o ciencias exactas": ["FISICA", "QUIMICA", "MATEMATICAS", "CIENCIAS EXACTAS"],
                    "Informática y sistemas de información": ["INFORMATICA", "TECNOLOGIAS DE LA INFORMACION"],
                    "Ingeniería eléctrica y electrónica": ["INGENIERIA ELECTRICA", "ELECTRONICA"],
                    "Ciencia de materiales": ["CIENCIA DE MATERIALES"],
                    "Inteligencia artificial y robótica": ["INTELIGENCIA ARTIFICIAL", "ROBOTICA"]
                }
                dominios_teseo_permitidos_por_contexto = set()
                for dominio_zs in dominios_relevantes_tecnicos_preds:
                    dominios_teseo_permitidos_por_contexto.update(MAPA_CLASIFICADOR_A_TESEO_DOMINIOS_TECNICOS.get(dominio_zs, []))

                for etiqueta, confianza in etiquetas_teseo_base_preds:
                    dominio_de_la_etiqueta_teseo = mapa_dominios_teseo.get(etiqueta.upper())
                    if (etiqueta.lower() not in ETIQUETAS_MIGRACION_LOWER and
                        confianza >= base_teseo_threshold_display and
                        dominio_de_la_etiqueta_teseo and
                        dominio_de_la_etiqueta_teseo in dominios_teseo_permitidos_por_contexto):
                        resultados_filtrados_teseo.append((etiqueta, confianza))
                    elif (etiqueta.lower() in ETIQUETAS_MIGRACION_LOWER and confianza >= UMBRAL_RESCATE_TESEO_MIGRACION_INDIVIDUAL):
                        resultados_filtrados_teseo.append((etiqueta, confianza))

                for etiqueta, prob in etiquetas_unesco_base_preds:
                    if (etiqueta.lower() not in ETIQUETAS_MIGRACION_LOWER and prob >= base_unesco_threshold_display):
                        resultados_filtrados_unesco.append((etiqueta, prob))
                    elif (etiqueta.lower() in ETIQUETAS_MIGRACION_LOWER and prob >= UMBRAL_RESCATE_UNESCO_MIGRACION_INDIVIDUAL):
                        resultados_filtrados_unesco.append((etiqueta, prob))

            elif (not es_tesis_tecnica) and (hay_dominio_social_migratorio_fuerte or teseo_predice_migracion_fuerte_especifica or unesco_predice_migracion_fuerte_especifica or fuerte_senal_migracion_combinada):
                st.info("Tesis identificada con **dominio de Ciencias Sociales/Humanidades** o un fuerte componente migratorio. Se muestran los descriptores relevantes.")
                for etiqueta, prob in etiquetas_teseo_base_preds:
                    if etiqueta.lower() in ETIQUETAS_MIGRACION_LOWER:
                        if prob >= 0.15:
                            resultados_filtrados_teseo.append((etiqueta, prob))
                    elif prob >= base_teseo_threshold_display:
                        resultados_filtrados_teseo.append((etiqueta, prob))

                for etiqueta, prob in etiquetas_unesco_base_preds:
                    if etiqueta.lower() in ETIQUETAS_MIGRACION_LOWER:
                        if prob >= 0.40:
                            resultados_filtrados_unesco.append((etiqueta, prob))
                    elif prob >= base_unesco_threshold_display:
                        resultados_filtrados_unesco.append((etiqueta, prob))
            else:
                st.info("No se ha detectado un dominio temático claro. Se aplicarán umbrales estrictos y se excluirán los términos de migración.")
                UMBRAL_TESEO_SIN_CONTEXTO_AMBIGUO = 0.98
                UMBRAL_UNESCO_SIN_CONTEXTO_AMBIGUO = 0.95
                for etiqueta, prob in etiquetas_teseo_base_preds:
                    if prob >= UMBRAL_TESEO_SIN_CONTEXTO_AMBIGUO and etiqueta.lower() not in ETIQUETAS_MIGRACION_LOWER:
                        resultados_filtrados_teseo.append((etiqueta, prob))
                for etiqueta, prob in etiquetas_unesco_base_preds:
                    if prob >= UMBRAL_UNESCO_SIN_CONTEXTO_AMBIGUO and etiqueta.lower() not in ETIQUETAS_MIGRACION_LOWER:
                        resultados_filtrados_unesco.append((etiqueta, prob))

        status.update(label="¡Análisis completado!", state="complete", expanded=False)

    st.success(f"Texto extraído correctamente ({len(texto_completo)} caracteres).")

    if idioma != 'es':
        st.warning(f"Idioma detectado: **{idioma.upper()}**. Los modelos de clasificación están optimizados solo para español y no se aplicarán.")

    if idioma == 'es':
        conceptos_teseo_detectados = detectar_conceptos_rdf_en_texto(texto_completo, procesador_teseo)
        conceptos_teseo_lower = {c.lower() for c in conceptos_teseo_detectados}
        if conceptos_teseo_lower:
            with st.expander("Ver conceptos del vocabulario TESEO encontrados literalmente en el texto"):
                st.write(list(conceptos_teseo_detectados))

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Clasificación de descriptores (TESEO)")
        if resultados_filtrados_teseo:
            st.write("**Descriptores para TESEO:**")
            resultados_filtrados_teseo.sort(key=lambda x: x[1], reverse=True)
            for etiqueta, prob in resultados_filtrados_teseo:
                color_etiqueta = "green" if prob > 0.7 else "orange" if prob > 0.4 else "red"
                marca = "(Encontrada literalmente en el texto)" if etiqueta.lower() in conceptos_teseo_lower else ""
                st.markdown(f"• <span style='color:{color_etiqueta}; font-weight:bold'>{etiqueta}</span> – confianza: `{prob:.2f}` {marca}", unsafe_allow_html=True)
        else:
            st.warning("Ninguna etiqueta de TESEO superó los umbrales de filtrado contextual.")

    with col2:
        st.header("Clasificación de palabras clave (UNESCO)")
        if resultados_filtrados_unesco:
            st.write("**Palabras clave de UNESCO:**")
            resultados_filtrados_unesco.sort(key=lambda x: x[1], reverse=True)
            for etiqueta, prob in resultados_filtrados_unesco:
                color_etiqueta = "green" if prob > 0.6 else "orange" if prob > 0.4 else "red"
                st.markdown(f"• <span style='color:{color_etiqueta}; font-weight:bold'>{etiqueta}</span> – confianza: `{prob:.2f}`", unsafe_allow_html=True)
        else:
            st.warning("Ninguna etiqueta de UNESCO superó los umbrales de filtrado contextual.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()