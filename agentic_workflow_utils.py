from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from collections import defaultdict

def collect_statistics(context):
    """
    Collects statistics about the most cited documents and their associated pages.
    
    Args:
    - context: A list of `Document` objects (LangChain) containing metadata with `doc_name` and `page_no` as strings.
    
    Returns:
    - str: A readable string summary of the most cited documents and their associated pages.
    """
    
    # Dictionnaires pour compter les citations des documents et les pages associées à chaque document
    doc_stats = defaultdict(int)  # Compteur pour les documents
    page_stats = defaultdict(list)  # Liste des pages par document

    # Parcours de chaque document dans le contexte
    for doc in context:
        doc_name = doc.metadata.get("doc_name", "Unknown")
        page_str = doc.metadata.get("page_no", "")
        
        # Convertir la chaîne de pages en une liste d'entiers
        pages = [int(p.strip()) for p in page_str.split(",") if p.strip().isdigit()]
        if page_str == "" :
            pages = [doc.metadata.get("sheet_name","")]
        # Incrémenter le compteur pour le document
        doc_stats[doc_name] += 1

        # Ajouter les pages au document
        page_stats[doc_name].extend(pages)

    # Trier les documents par nombre d'occurrences
    sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1], reverse=True)

    # Construire une chaîne lisible en anglais
    result_str = "Retrieved Documents: \n"
    
    for doc_name, count in sorted_docs:
        pages_str = ", ".join(map(str, sorted(page_stats[doc_name])))  # Convertir les pages en une chaîne lisible
        if pages_str == "" :
            result_str += f" - {doc_name}: {count} citation(s)\n"
        else :
            result_str += f" - {doc_name}: {count} citation(s), Pages: {pages_str}\n"
    
    return result_str

from pathlib import Path 

def collect_statistics_from_retrieved_pages(retrieved_pages):
    stats = {}
    for pdf_path, page_no in retrieved_pages :
        if Path(pdf_path).stem in stats :
            stats[Path(pdf_path).stem].append(page_no)
        else :
            stats[Path(pdf_path).stem] = [page_no]
        
    result_str = "Retrieved Documents: \n"
    for doc_name in stats :
        pages_str = ", ".join(map(str, sorted(stats[doc_name])))
        result_str += f"{doc_name}, pages : {pages_str}\n"
    return result_str

import json 
import os
from datetime import datetime
# Affichage des résultats

# Fonction utilitaire pour loguer chaque étape dans un fichier local
def log_step(step_name, input_data, output_data, retrieval_statistics = None, log_file = "logs/trace_log.json", log_file_text = "logs/trace_log.txt"):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step_name,
        "input": input_data,
        "output": output_data
    }
    
    log_text = f"[{log_entry['timestamp']}] Step: {log_entry['step']}\n"
    if input_data :
        log_text += f"Input: {json.dumps(input_data, indent=2)}\n"
    

    if retrieval_statistics :
        log_entry['retrieval_statistics'] = retrieval_statistics
        log_text += retrieval_statistics+'\n'
    if output_data :
        log_text += f"Output: {json.dumps(output_data, indent=2)}\n"
    
    log_text += "\n" + "-"*50 + "\n"
   
    # Créer un dossier pour stocker les logs s'il n'existe pas
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Sauvegarder les logs dans un fichier JSON pour chaque étape
    
    # Ajouter le log au fichier JSON
    with open(log_file, "a") as file:
        json.dump(log_entry, file)
        file.write("\n")

    # Optionnel : Loguer aussi dans un fichier texte pour facilité de lecture
    
    with open(log_file_text, "a") as file:
        file.write(log_text)
    
    return(log_text)

def serialize_messages(messages):
    """Sérialise les messages LangChain en format compatible avec JSON."""
    serialized = []
    
    for message in messages:
        # Vérifier le type de chaque message et extraire les informations
        if isinstance(message, SystemMessage):
            serialized_message = {
                "role": "system",
                "content": message.content
            }
        elif isinstance(message, HumanMessage):
            serialized_message = {
                "role": "user",
                "content": message.content
            }
        elif isinstance(message, AIMessage):
            serialized_message = {
                "role": "assistant",
                "content": message.content
            }
        elif isinstance(message, ToolMessage):
            serialized_message = {
                "role": "tool",
                "content": message.content
            }
        else:
            # Si le message est d'un type inconnu, on l'ignore ou on le gère différemment
            serialized_message = {
                "role": "unknown",
                "content": str(message)  # Convertir en chaîne
            }

        # Sérialiser les tool_calls et response_metadata si présents
        if hasattr(message, 'tool_calls') and message.tool_calls:
            serialized_message['tool_calls'] = str(message.tool_calls)  # Convertir en chaîne
        if hasattr(message, 'response_metadata') and message.response_metadata:
            serialized_message['response_metadata'] = str(message.response_metadata)  # Convertir en chaîne
        
        serialized.append(serialized_message)
    
    return serialized

from my_paths import * 
import fitz
from PIL import Image

import numpy as np

from rapidfuzz import process, fuzz
from pathlib import Path

# Exemple avec cache global
_file_index = None

def build_file_index(data_dirs):
    # Construit un dictionnaire {stem: Path} sans doc/docx
    index = {}
    for data_dir in data_dirs:
        if data_dir and os.path.exists(data_dir) : 
            for f in Path(data_dir).rglob("*"):
                if f.is_file() and f.suffix.lower() not in ['.doc', '.docx']:
                    stem = f.stem.strip().lower()
                    index[stem] = f
    return index

def find_document_path(doc_name: str, data_dirs=[RAW_DATA_DIR, RAW_DATA_DIR_B], fuzzy_threshold=60) -> str:
    global _file_index
    if _file_index is None:
        _file_index = build_file_index(data_dirs)
    doc_name = doc_name.strip().lower()
    # match exact
    if doc_name in _file_index:
        return _file_index[doc_name]
    # utiliser rapidfuzz pour trouver le meilleur match
    matches = process.extractOne(doc_name, _file_index.keys(), scorer=fuzz.ratio)
    if matches:
        best_match, score, _ = matches
        if score >= fuzzy_threshold :
            return _file_index[best_match]
    return None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

from bge_m3_embeddings import MyAPIEmbeddings
api = MyAPIEmbeddings(chunk_size=20)

def match_doc_names(doc_name: str, data_dir: str, api: MyAPIEmbeddings = api, top_k=10, batch_size=1500) -> list:
    # Récupération des noms de fichiers valides
    all_names = [f.name.replace('.pdf','') for f in Path(data_dir).rglob("*") if f.suffix.lower() in {".pdf", ".xls", ".xlsx"}]

    vec_query = api.embed_documents([doc_name])[0]

    embeddings = []
    for i in range(0, len(all_names), batch_size):
        batch = all_names[i:i+batch_size]
        batch_embeddings = api.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    
    matchs = []
    for name, emb in zip(all_names, embeddings[1:]):
        vec_name = np.array(emb)
        sim = cosine_similarity(vec_query, vec_name)
        matchs.append((name, sim))
    
    matchs_sorted = sorted(matchs, key=lambda x: x[1], reverse=True)
    
    return [name for name,_ in matchs_sorted[:top_k]]

def match_html_doc_names(doc_name: str, data_dir: str, api: MyAPIEmbeddings = api, top_k=10, batch_size=1500) -> list:
    # Récupération des noms de fichiers valides
    all_names = [str(f.name) for f in Path(data_dir).rglob("*") if f.suffix.lower() == ".html"]

    vec_query = api.embed_documents([doc_name])[0]

    embeddings = []
    for i in range(0, len(all_names), batch_size):
        batch = all_names[i:i+batch_size]
        batch_embeddings = api.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    
    matchs = []
    for name, emb in zip(all_names, embeddings):
        vec_name = np.array(emb)
        sim = cosine_similarity(vec_query, vec_name)
        matchs.append((name, sim))
    
    matchs_sorted = sorted(matchs, key=lambda x: x[1], reverse=True)
    
    return [name for name,_ in matchs_sorted[:top_k]]

def get_page_image(pdf_path, page_num) :
    doc = fitz.open(pdf_path)
    # Attention : PyMuPDF utilise un index 0-based pour les pages
    page_index = page_num - 1
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=96)  # Rend un rendu de la page en image
    #pix.save(f"pages_images/{page_num}.png")
    # Convertir pixmap en image PIL pour pouvoir l'encodage sans sauvegarder sur disque
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

import re
import urllib.parse

def remplacer_fichier_dans_texte(texte, fichier):
    """
    Supprime uniquement les backticks autour du fichier puis remplace le fichier
    par un lien cliquable, sans toucher aux éventuelles étoiles.
    """
    # Regex qui capture optionnellement des backticks autour du fichier
    pattern = re.compile(rf'(`+)?\b({re.escape(fichier)})\b(`+)?')

    def remplacer(match):
        nom_fichier = match.group(2)
        url = find_document_path(nom_fichier)
        if url:
            url = str(url).replace('\\', '/')
            url = urllib.parse.quote(url, safe="/:")
            # On retourne un lien markdown sans backticks, mais on conserve les étoiles éventuelles
            # donc on remet en forme le lien avec le nom du fichier
            return f'[{nom_fichier}]({url})'
        else:
            # Si pas d'URL trouvée, on retourne original avec backticks conservés
            return match.group(0)

    texte_modifie = pattern.sub(remplacer, texte)
    return texte_modifie

def rendre_citation_cliquable(texte, liste_fichiers):
    for fichier in liste_fichiers:
        texte = remplacer_fichier_dans_texte(texte, fichier)
    return texte
