from pathlib import Path

# ---------------------
# PATHS A PERSONNALISER
# ---------------------

# Base path : le path absolu du répertoire
BASE_PATH = Path("")

# Data path : le path absolu du dossier contenant la base de données des dernières versions de vos documents (doit être un objet Path)
DATA_PATH = BASE_PATH / "data"

# Pour le visual RAG sur les SDD (mettre None si non utilisé)
PAGES_EMBEDDINGS_PATH = DATA_PATH / "page_embeddings.npy"

# Data path B : le path absolu du dossier contenant la base de données des diverses versions de vos documents (mettre None si non utilisé) (doit être un objet Path)
DATA_PATH_B = None

# -----------------------------------------------
# BASE DE DONNEES A : AVEC LES DERNIERES VERSIONS
# -----------------------------------------------

# Données brutes
RAW_DATA_DIR = DATA_PATH / "Raw_database"
GLOSSARY_PATH = DATA_PATH / "glossary.json"

# Données parsées
PARSED_DATA_DIR = DATA_PATH / "Processed_database"

# Données pour la récupération

# RAG hybride
CHROMA_DIR = DATA_PATH / "chroma_store"
WHOOSH_DIR = DATA_PATH / "whoosh_index"

# -----------------------------------------------
# BASE DE DONNEES B : AVEC DIVERSES VERSIONS
# -----------------------------------------------

if DATA_PATH_B :
    # Données brutes
    RAW_DATA_DIR_B = DATA_PATH_B / "Raw_database"

    # Données parsées
    PARSED_DATA_DIR_B = DATA_PATH_B / "Processed_database"

    # Données pour la récupération

    # RAG hybride
    CHROMA_DIR_B = DATA_PATH_B / "chroma_store"
    WHOOSH_DIR_B = DATA_PATH_B / "whoosh_index"
else : 
   RAW_DATA_DIR_B = None

# -----------------------------------------------
# AUTRES PATH
# -----------------------------------------------

# Tokenizer mistral pour gérer la taille des requêtes
MISTRAL_TOKENIZER_PATH = BASE_PATH / "models/tekken.json"

# Reranker optionel pour le RAG hybride
RERANKER_PATH = ""

# Modèle d'embedding visuel
COLSMOL_PATH = BASE_PATH / "models/colSmol-256M"

# Path pour le processing des documents 
BGE_M3_TOKENIZER_PATH = BASE_PATH / "models/bge-m3-tokenizer"
DOCLING_ARTIFACTS_PATH = BASE_PATH / "models/docling_artifacts"