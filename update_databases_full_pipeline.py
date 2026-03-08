import sys
import os
import json
import logging

logging.getLogger("urllib3").setLevel(logging.ERROR)

child_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'process_pdf_documents'))
if child_dir not in sys.path:
    sys.path.insert(0, child_dir)

child_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'process_excel_documents'))
if child_dir not in sys.path:
    sys.path.insert(0, child_dir)

child_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'process_html_documents'))
if child_dir not in sys.path:
    sys.path.insert(0, child_dir)

import shutil
# ---------------------
# FONCTIONS UTILITAIRES
# ---------------------

def supprimer_fichiers_cache(dossier):
    for root, dirs, files in os.walk(dossier):
        for file in files:
            if file.strip().startswith('~$'):
                chemin = os.path.join(root, file)
                try:
                    os.remove(chemin)
                    print(f"Supprimé : {chemin}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {chemin} : {e}")

def remove_folders(root_dir, relative_paths):
    """
    Supprime récursivement les dossiers listés (paths relatifs à root_dir).

    :param root_dir: chemin absolu ou relatif du dossier racine
    :param relative_paths: liste de chemins relatifs à root_dir à supprimer
    """
    for rel_path in relative_paths:
        folder_path = os.path.join(root_dir, rel_path)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Suppression du dossier : {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"Attention, le dossier n'existe pas ou n'est pas un dossier : {folder_path}")

def get_extra_processed_db_folders(raw_database_dir, processed_data_dir):
    """Renvoie les dossiers en trop dans processed_data_dir par rapport à raw_database_dir"""
    
    # 1. Récupérer les fichiers pdf dans raw_database_dir, chemins relatifs sans l'extension
    raw_pdfs = []
    for dirpath, _, filenames in os.walk(raw_database_dir):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, raw_database_dir)
                raw_pdfs.append(rel_path)

    raw_base_paths = set(os.path.splitext(p)[0].strip() for p in raw_pdfs)

    # 2. Récupérer les dossiers feuilles dans processed_data_dir en excluant certains sous-dossiers
    excluded_dirs = {"pictures", "tables", "pictures_deleted", "tables_serialized", "formulas", "html_chunks"}
    leaf_dirs = set()

    for dirpath, dirnames, filenames in os.walk(processed_data_dir):
        # Exclure certains sous-dossiers de l'exploration
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        # Condition pour être un dossier feuille : pas de sous-dossiers dans dirnames après filtre ; et différent de processed_data_dir
        if (not dirnames) and (dirpath.strip() != str(processed_data_dir).strip()):
            # rel_path du dossier feuille
            rel_dir = os.path.relpath(dirpath, processed_data_dir)
            leaf_dirs.add(rel_dir)

    # 3. Trouver les dossiers feuilles qui ne correspondent pas à un fichier pdf dans raw_database_dir
    # Compare en enlevant l'extension .pdf des fichiers dans raw_database_dir avec les dossiers feuilles dans processed_data_dir
    extras = leaf_dirs - raw_base_paths
    return sorted(extras)

def remove_extra_excel_chunks(raw_data_dir, output_json_path) :

    file_names_raw = [f.name for f in Path(raw_data_dir).rglob("*") if f.suffix=='.xls' or f.suffix=='.xlsx']
    kept_chunks = []
    
    with open(output_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    for chunk in chunks :
        if chunk['metadata']['doc_name'] in file_names_raw :
            kept_chunks.append(chunk)
        else :
            print(f"Suppression des chunks associés au fichier {chunk['metadata']['doc_name']}")

    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(kept_chunks, f_out, ensure_ascii=False, indent=2)

import os
from langchain_chroma import Chroma
from bge_m3_embeddings import MyAPIEmbeddings
from pathlib import Path

def get_doc_names_in_chromadb(db) :
    all_docs = db.get()  # récupère tous documents avec metadata

    # extraire les doc_names existants
    doc_names = set()
    for meta in all_docs['metadatas']:
        if meta and 'doc_name' in meta:
            doc_names.add(meta['doc_name'])
    return doc_names

def cleanup_chromadb_entries(raw_database_dir, chroma_dir):
    """
    Supprime de db ChromaDB toutes les entrées dont doc_name ne correspond pas à
    un fichier PDF dans raw_database_dir.
    """
    
    # Récupérer la liste des noms des pdf dans raw_database_dir (sans extension)
    file_names_raw = [f.name.replace('.pdf','').strip() for f in Path(raw_database_dir).rglob("*")]

    embeddings = MyAPIEmbeddings()
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

    doc_names = get_doc_names_in_chromadb(db)    

    # Trouver les doc_names à supprimer (ceux qui ne sont pas dans pdf_names)
    to_remove = []
    for doc_name in doc_names :
        if not(doc_name.strip() in file_names_raw) :
            to_remove.append(doc_name) 

    if not to_remove:
        print("Aucune entrée à supprimer de chromadb, tout est à jour.")
        return

    print(f"Suppression de {len(to_remove)} documents de la database '{chroma_dir}'")

    for doc_name in to_remove:
        # filtrer et supprimer par doc_name dans metadata
        db.delete(where={"doc_name": doc_name})
        print(f"Supprimé de '{chroma_dir}' : {doc_name}")

    print("Nettoyage terminé.")

def delete_docs_by_doc_name(ix, doc_name):
    # Ouvrir un writer
    writer = ix.writer()
    try:
        # Suppression par terme exact sur la valeur 'doc_name'
        writer.delete_by_term('doc_name', doc_name)
        writer.commit()
        print(f"Documents avec doc_name={doc_name} supprimés.")
    except Exception as e:
        writer.cancel()
        print(f"Erreur lors de la suppression: {e}")

def get_doc_names_in_whoosh(ix) :
    doc_names = set()
    with ix.searcher() as searcher:
        for doc in searcher.all_stored_fields():
            doc_names.add(doc["doc_name"])
    return(doc_names)

def cleanup_page_embeddings(raw_database_dir, page_embeddings_file) :
    from visual_rag_on_sdd import get_every_SDD_document_path, parse_path_and_page
    import numpy as np
    file_names_raw = [Path(f).stem.strip() for f in get_every_SDD_document_path(raw_database_dir)]
    embeddings_data = {}
    if os.path.exists(page_embeddings_file):
        embeddings_data = np.load(page_embeddings_file, allow_pickle=True).item()

    keys_to_remove = []
    pdf_to_remove = []
    for key in embeddings_data :
        rel_path,_ = parse_path_and_page(key)
        if not(Path(rel_path).stem.strip() in file_names_raw) :
            if not(rel_path in pdf_to_remove) :
                print(f'Fichier {rel_path} non présent dans {raw_database_dir}. Suppression des embeddings visuels associés.')
            else :
                pdf_to_remove.append(rel_path)
            keys_to_remove.append(key)
    
    if keys_to_remove :
        for key in keys_to_remove :
            del embeddings_data[key]
        np.save(page_embeddings_file, embeddings_data)
    else :
        print('Aucun embedding en trop dans les embeddings visuels.')

def cleanup_whoosh_entries(raw_database_dir, whoosh_dir) :
    """
    Supprime de l'index whoosh toutes les entrées dont doc_name ne correspond pas à
    un fichier PDF dans raw_database_dir.
    """
    # Récupérer la liste des noms des pdf dans raw_database_dir (sans extension)
    file_names_raw = [f.name.replace('.pdf','').strip() for f in Path(raw_database_dir).rglob("*")]

    from hybrid_search import init_whoosh
    ix = init_whoosh(whoosh_dir)

    doc_names = get_doc_names_in_whoosh(ix)

    # Trouver les doc_names à supprimer (ceux qui ne sont pas dans pdf_names)
    to_remove = []
    for doc_name in doc_names :
        if not(doc_name.strip() in file_names_raw) :
            to_remove.append(doc_name) 

    if not to_remove:
        print("Aucune entrée à supprimer de whoosh, tout est à jour.")
        return

    print(f"Suppression de {len(to_remove)} documents de la database '{whoosh_dir}'")

    for doc_name in to_remove:
        # filtrer et supprimer par doc_name dans metadata
        delete_docs_by_doc_name(ix, doc_name)
        print(f"Supprimé de '{whoosh_dir}' : {doc_name}")

    print("Nettoyage terminé.")

def remove_extra_html_chunks_file(raw_data_dir, html_chunks_dir) :
    
    # Récupérer la liste des noms des html dans raw_database_dir (sans extension)
    file_names_raw = [f.stem.strip() for f in Path(raw_data_dir).rglob("*")]
    # Récupérer la liste des noms des fichiers dans html_chunks_dir (sans extension)
    file_names_chunked = [f.stem.strip() for f in Path(html_chunks_dir).rglob("*")]
    
    for file_name in file_names_chunked :
        if not file_name in file_names_raw :
            print(f"Fichier {file_name}.html non présent dans la base de données brutes. Suppression.")
            file_path = Path(html_chunks_dir) / (file_name+'.json')
            if os.path.exists(file_path):
                os.remove(file_path)    
    print('Nettoyage des htmls terminés.')

def clear_excels_from_chroma_and_whoosh(raw_database_dir, whoosh_dir, chroma_dir) :
    excel_file_names_raw = [f.name.strip() for f in Path(raw_database_dir).rglob("*") if f.suffix=='.xls' or f.suffix=='.xlsx']

    embeddings = MyAPIEmbeddings()
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    doc_names_chroma = get_doc_names_in_chromadb(db)    

    from hybrid_search import init_whoosh
    ix = init_whoosh(whoosh_dir)
    doc_names_whoosh = get_doc_names_in_whoosh(ix)
    for excel_name in excel_file_names_raw :
        if excel_name in doc_names_whoosh :
            delete_docs_by_doc_name(ix, excel_name)
            print(f"Supprimé de '{whoosh_dir}' : {excel_name}")
        if excel_name in doc_names_chroma : 
            db.delete(where={"doc_name": excel_name})
            print(f"Supprimé de '{chroma_dir}' : {excel_name}")
    print('Tous les chunks liés à des excels ont été supprimés')

# ---------------------
# FONCTIONS PRINCIPALES
# ---------------------
def update_databases(raw_data_dir, parsed_data_dir, chroma_dir, whoosh_dir, pages_embeddings_file=None, glossary_path=None, delete_non_essential_files=False) :
    """Full pipeline de mise à jour des bases de données. 
    Garantit que le contenu des bases de données parsée (parsed_data_dir), chroma (chroma_dir) et whoosh (whoosh_dir)
    correspond strictement à celui de la base de donnée brute (raw_data_dir)"""
    from process_database_pdfs import process_raw_database_pdfs
    from mistral_figure_annotations import annotate_database_figures

    # ÉTAPE 0 : Suppression des fichiers caches, conversion des words de raw_data_dir en PDF
    supprimer_fichiers_cache(RAW_DATA_DIR)
    parsed_data_dir.mkdir(parents=True, exist_ok=True)
    from word_to_pdf import convert_folder
    convert_folder(raw_data_dir)

    # ÉTAPE 1 : ON NETTOIE LES BASES DE DONNEES
    
    # a) On supprime les fichiers existants dans la base de données parsée mais n'existant plus dans la base de données brute
    if os.path.exists(parsed_data_dir) :
        extras_folders = get_extra_processed_db_folders(raw_data_dir, parsed_data_dir)
        str_extras = " ,".join(extras_folders)
    else : 
        extras_folders = []
        str_extras = ''

    if extras_folders :
        print(f"Dossiers en trop dans {parsed_data_dir} : {str_extras}")
        remove_folders(root_dir=parsed_data_dir, relative_paths=extras_folders)

    chunks_excels_path = parsed_data_dir / 'chunks_excels.json'
    if os.path.exists(chunks_excels_path) :
        remove_extra_excel_chunks(raw_data_dir, output_json_path=chunks_excels_path)
    
    html_chunks_dir = parsed_data_dir / 'html_chunks'
    if os.path.exists(html_chunks_dir) : 
        remove_extra_html_chunks_file(raw_data_dir, html_chunks_dir)
    
    # b) On supprime les entrées de chroma et whoosh correspondants à des fichiers n'existant plus dans la base de données brute

    cleanup_chromadb_entries(raw_data_dir, chroma_dir)
    cleanup_whoosh_entries(raw_data_dir, whoosh_dir)

    # c) On supprime les entrées du fichier .npy correspondants à des fichiers n'existant plus dans la base de données brute
    if pages_embeddings_file and os.path.exists(pages_embeddings_file) :
        cleanup_page_embeddings(raw_data_dir, pages_embeddings_file)
    
    # ÉTAPE 2 : PARSING ET CHUNKING DES DOCUMENTS
    
    # a) On parse les documents excels de raw_data_dir 

    from chunk_every_sheet import export_sheets_chunked
    export_sheets_chunked(chunks_excels_path, parsed_data_dir / 'headers.json', raw_data_dir)

    # b) i. On parse les documents pdfs de raw_data_dir avec notre pipeline docling
    unprocessed_files = process_raw_database_pdfs(database_path=raw_data_dir, output_path=parsed_data_dir)
    if unprocessed_files :
        print(f"Échec du traitement des fichiers suivants : {unprocessed_files}")

    # b) ii. On annote les figures de nos chunks de parsed_data_dir avec mistral

    annotate_database_figures(data_dir=parsed_data_dir, pdf_dir=raw_data_dir)

    # b) iii. On supprime ou non les fichiers non essentiels (qui servent normalement à évaluer la qualité du parsing)
    if delete_non_essential_files :
        from delete_non_essential_files import delete_non_essential_files
        delete_non_essential_files(PARSED_DATA_DIR)

    # c) On parse les documents html de raw_data_dir
    from chunk_html import chunk_every_html
    chunk_every_html(raw_data_dir, html_chunks_dir)

    # ÉTAPE 3 : ON VECTORISE LES CHUNKS DANS CHROMADB ; ON INDEXE LES CHUNKS AVEC WHOOSH
    
    from hybrid_search import fill_databases_with_pdfs, fill_databases_with_chunks_from_json
    # a) On vectorise / indexe les pdfs
    fill_databases_with_pdfs(parsed_data_dir, chroma_dir, whoosh_dir)

    # b) On vectorise / indexe les excels
    if os.path.exists(chunks_excels_path) :
        fill_databases_with_chunks_from_json(chroma_dir, whoosh_dir, json_files = [chunks_excels_path])

    # c) On vectorise / indexe les fichiers html
    files_to_process = list(html_chunks_dir.glob("*"))
    print(f"Fichier htmls à traiter : {files_to_process}")
    if files_to_process :
        fill_databases_with_chunks_from_json(chroma_dir, whoosh_dir, json_files=files_to_process)

    # d) On vectorise les images des pages des SDD dans un fichier .npy
    if pages_embeddings_file :
        from visual_rag_on_sdd import process_sdd_visual_embeddings
        process_sdd_visual_embeddings(raw_data_dir, pages_embeddings_file)

    # ÉTAPE 4 : ON CRÉE LE GLOSSAIRE DU PROJET
    if glossary_path : 
        from manage_glossary import create_glossary
        create_glossary(whoosh_dir, glossary_path)

    print("Les bases de données ont été mises à jour.")

if __name__=="__main__" :
    from my_paths import *
    update_databases(RAW_DATA_DIR, PARSED_DATA_DIR, CHROMA_DIR, WHOOSH_DIR, pages_embeddings_file=PAGES_EMBEDDINGS_PATH, glossary_path=GLOSSARY_PATH)
    
    if DATA_PATH_B and DATA_PATH_B.is_dir() :
        update_databases(RAW_DATA_DIR_B, PARSED_DATA_DIR_B, CHROMA_DIR_B, WHOOSH_DIR_B)
    

    