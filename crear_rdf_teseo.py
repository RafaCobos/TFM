from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
import fitz
import re
from collections import defaultdict
from urllib.parse import quote
import os

# Extraer datos del PDF
def extraer_datos(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    lines = text.splitlines()

    categorias = defaultdict(lambda: {"labels": [], "notations": []})
    current_category = ""
    codigo_pattern = re.compile(r"^\d{6}$")
    categoria_pattern = re.compile(r"^\d{2}\.\s")

    for i in range(len(lines)):
        line = lines[i].strip()

        if categoria_pattern.match(line):
            current_category = line.split('. ', 1)[-1].replace(" ", "_")
        elif codigo_pattern.match(line) and i + 1 < len(lines):
            codigo = line
            label = lines[i + 1].strip()
            if label and not codigo_pattern.match(label):
                categorias[current_category]["labels"].append(label)
                categorias[current_category]["notations"].append(codigo)

    return categorias

# Crear RDF agrupado por categoría
def crear_rdf_agrupado(categorias, salida_rdf):
    g = Graph()
    NS1 = Namespace("http://www.w3.org/2004/02/skos/core#")
    g.bind("ns1", NS1)

    for categoria, datos in categorias.items():
        uri_categoria = URIRef(f"file:///mnt/data/Teseo.rdf#{quote(categoria)}")
        for label in datos["labels"]:
            g.add((uri_categoria, NS1.prefLabel, Literal(label, lang="es")))
        for notation in datos["notations"]:
            g.add((uri_categoria, NS1.notation, Literal(notation)))

    g.serialize(destination=salida_rdf, format="pretty-xml")

# Ejecutar proceso
pdf_path = "documentRDF.pdf"
rdf_output_path = "teseo.rdf"
datos = extraer_datos(pdf_path)
crear_rdf_agrupado(datos, rdf_output_path)

rdf_output_path
