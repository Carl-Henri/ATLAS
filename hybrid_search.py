import os
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from langchain_core.documents import Document

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, Or
from whoosh.analysis import KeywordAnalyzer


from langchain_chroma import Chroma
from bge_m3_embeddings import MyAPIEmbeddings

# ==============================
# UTILITAIRES
# ==============================
def make_doc_id(doc: Document) -> str:
    """Construit un hash unique et stable pour chaque chunk."""
    key = doc.page_content + str(sorted(doc.metadata.items()))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def load_chunks_as_documents(json_path: Path):
    """Charge un fichier JSON en liste de Document LangChain."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    documents = []
    for chunk in chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        if "page_no" in metadata :
            metadata["page_no"] = ", ".join([str(p) for p in metadata.get("page_no", [])])
        doc = Document(page_content=content, metadata=metadata)
        doc.metadata["id"] = make_doc_id(doc)
        documents.append(doc)
    return documents


# ==============================
# CHROMADB
# ==============================
def build_chroma(docs, persist_dir):
    embeddings = MyAPIEmbeddings()
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    print(f"Chroma créé dans {persist_dir}")
    return db

def add_chroma(docs, persist_dir, max_batch_size = 5400):
    embeddings = MyAPIEmbeddings()
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print(f'Ajout de {len(docs)} chunks à {persist_dir}')
    if len(docs) <= max_batch_size :
        db.add_documents(docs)
    else : 
        it = len(docs) // max_batch_size 
        for i in range(it) :
            db.add_documents(docs[i*max_batch_size:(i+1)*max_batch_size])
        db.add_documents(docs[it*max_batch_size:])
    return db

def get_existing_chroma_ids(chroma_dir):
    """Récupère tous les IDs déjà présents dans Chroma (via métadatas)."""
    if not os.path.exists(chroma_dir):
        return set()
    embeddings = MyAPIEmbeddings()
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    
    # Récupérer tous les documents et leurs métadatas
    all_docs = db.get()
    metadatas = all_docs.get("metadatas", [])
    
    # Extraire tous les IDs présents
    existing_ids = set()
    for meta_list in metadatas:  # chaque doc peut avoir plusieurs métadatas si vectorstore multi-dim
        if isinstance(meta_list, dict):
            if "id" in meta_list:
                existing_ids.add(meta_list["id"])
        elif isinstance(meta_list, list):
            for m in meta_list:
                if "id" in m:
                    existing_ids.add(m["id"])
    return existing_ids

# ==============================
# WHOOSH
# ==============================
def init_whoosh(index_dir):
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        content=TEXT(stored=True),
        doc_name=ID(stored=True),
        hierarchy=TEXT(stored=True)
    )
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
        ix = index.create_in(index_dir, schema)
        print(f"Index Whoosh créé dans {index_dir}")

    else:
        ix = index.open_dir(index_dir)
        print(f"Index Whoosh ouvert depuis {index_dir}")
    return ix

def add_to_whoosh(ix, docs):
    writer = ix.writer()
    for doc in docs:
        writer.add_document(
            doc_id=doc.metadata["id"],
            content=doc.page_content,
            doc_name=doc.metadata.get("doc_name", "unknown"),
            hierarchy=doc.metadata.get("hierarchy", "unknown")
        )
    writer.commit()
    print(f"{len(docs)} docs ajoutés à Whoosh.")

def get_existing_whoosh_ids(ix):
    """Récupère tous les doc_id déjà présents dans Whoosh."""
    existing_ids = set()
    with ix.searcher() as searcher:
        for doc in searcher.all_stored_fields():
            existing_ids.add(doc["doc_id"])
    return existing_ids

# ==============================
# PIPELINES DE CONSTRUCTION
# ==============================
def fill_databases_with_pdfs(data_dir, chroma_dir, whoosh_dir):
    ix = init_whoosh(whoosh_dir)
    all_files = [f for f in Path(data_dir).rglob("*") if f.name.endswith("annotated_chunks.json")]
    print(f"{len(all_files)} fichiers annotated_chunks.json trouvés.")

    existing_whoosh_ids = get_existing_whoosh_ids(ix)
    existing_chroma_ids = get_existing_chroma_ids(chroma_dir)
    for file in tqdm(all_files):
        print(f"\nTraitement : {file}")
        docs = load_chunks_as_documents(file)

        # Filtrer uniquement les docs non traités
        docs_to_add_chroma = []
        docs_to_add_whoosh = []
        for doc in docs:
            if not doc.metadata['id'] in existing_chroma_ids:
                docs_to_add_chroma.append(doc)
            if not doc.metadata['id'] in existing_whoosh_ids:
                docs_to_add_whoosh.append(doc)

        # Ajouter à Chroma
        if docs_to_add_chroma : 
            if os.path.exists(chroma_dir):
                    add_chroma(docs_to_add_chroma, chroma_dir)
            else:
                build_chroma(docs_to_add_chroma, chroma_dir)
        else : 
            print(f"\nTous les chunks de {file} ont déjà été ajoutés à Chroma")

        # Ajouter à Whoosh
        if docs_to_add_whoosh :
            add_to_whoosh(ix, docs_to_add_whoosh)
        else : 
            print(f"\nTous les chunks de {file} ont déjà été ajoutés à Whoosh")

def fill_databases_with_langchain_docs(docs, chroma_dir, whoosh_dir):
    for doc in docs : 
        if not('id' in doc.metadata) : 
            doc.metadata["id"] = make_doc_id(doc)
    print(f"Ajout de {len(docs)} aux bases de données")
    ix = init_whoosh(whoosh_dir)
    existing_whoosh_ids = get_existing_whoosh_ids(ix)
    existing_chroma_ids = get_existing_chroma_ids(chroma_dir)

    # Filtrer uniquement les docs non traités
    docs_to_add_chroma = []
    docs_to_add_whoosh = []
    for doc in docs:
        if not doc.metadata['id'] in existing_chroma_ids:
            docs_to_add_chroma.append(doc)
        if not doc.metadata['id'] in existing_whoosh_ids:
            docs_to_add_whoosh.append(doc)

    # Ajouter à Chroma
    if docs_to_add_chroma : 
        if os.path.exists(chroma_dir):
                add_chroma(docs_to_add_chroma, chroma_dir)
        else:
            build_chroma(docs_to_add_chroma, chroma_dir)
    else : 
        print(f"\nTous les docs ont déjà été ajoutés à Chroma")

    # Ajouter à Whoosh
    if docs_to_add_whoosh :
        add_to_whoosh(ix, docs_to_add_whoosh)
    else : 
        print(f"\nTous les docs ont déjà été ajoutés à Whoosh")

def fill_databases_with_chunks_from_json(chroma_dir, whoosh_dir, json_files) :
    ix = init_whoosh(whoosh_dir)

    existing_whoosh_ids = get_existing_whoosh_ids(ix)
    existing_chroma_ids = get_existing_chroma_ids(chroma_dir)
    docs = []
    for json_file in json_files :
        docs.extend(load_chunks_as_documents(json_file))

    # Filtrer uniquement les docs non traités
    docs_to_add_chroma = []
    docs_to_add_whoosh = []
    for doc in docs:
        if not doc.metadata['id'] in existing_chroma_ids:
            docs_to_add_chroma.append(doc)
        if not doc.metadata['id'] in existing_whoosh_ids:
            docs_to_add_whoosh.append(doc)

    # Ajouter à Chroma
    if docs_to_add_chroma : 
        if os.path.exists(chroma_dir):
                add_chroma(docs_to_add_chroma, chroma_dir)
        else:
            build_chroma(docs_to_add_chroma, chroma_dir)
    else : 
        print(f"\nTous les chunks de {json_file} ont déjà été ajoutés à Chroma")

    # Ajouter à Whoosh
    if docs_to_add_whoosh :
        add_to_whoosh(ix, docs_to_add_whoosh)
    else : 
        print(f"\nTous les chunks de {json_file} ont déjà été ajoutés à Whoosh")

# --- Chroma vector search ---
import threading

# Créer un lock global
model_lock = threading.Lock()
# ==============================
# RECHERCHE HYBRIDE
# ==============================
def hybrid_search(query, chroma_dir, whoosh_dir, doc_filter=None, top_k=60, k_rrf=30, alpha=0.5):
    """
    Recherche hybride BM25 + vectorielle, fusion RRF pondérée, retourne texte des documents + score.
    
    Args:
        query (str): requête.
        chroma_dir (str): chemin vers Chroma DB.
        whoosh_dir (str): chemin vers index Whoosh.
        doc_filter (list[str], optional): liste de doc_name à filtrer.
        top_k (int): nombre de résultats à retourner.
        k_rrf (int): paramètre de lissage RRF.
        alpha (float): pondération BM25 vs vectoriel.
    
    Returns:
        List[Tuple[str, float]]: [(contenu_doc, score_fusion), ...]
    """
    print('Lancement de la recherche hybride')
    embeddings = MyAPIEmbeddings()
    # --- BM25 Whoosh ---
    print('Début de la recherche par mots-clés')
    ix = index.open_dir(whoosh_dir)
    with ix.searcher() as searcher:
        og = OrGroup.factory(0.9) # le facteur permet de faire en sorte de favoriser la présence de plusieurs termes différents par rapport à la répétition d'un seul terme
        parser = QueryParser("content", schema=ix.schema, group=og) # group=og (OrGroup) pour récupérer un document si il contient l'un des termes de la requêtes
        q = parser.parse(query.replace('?','').replace('**',''))
        filter_q = None
        if doc_filter:
            terms = [Term("doc_name", name) for name in doc_filter]
            filter_q = Or(terms)
        results_bm25 = searcher.search(q, limit=top_k, filter=filter_q, terms=True)
        if results_bm25.has_matched_terms() :
            print(f"The terms which matched in the results : {results_bm25.matched_terms()}")
            for i, hit in enumerate(results_bm25[:min(8,len(results_bm25))]) : 
                print(f"The doc {i+1} retrieved matched {hit.matched_terms()}")
        bm25_scores = {r["doc_id"]: r.score for r in results_bm25}

    print(f"{threading.current_thread().name} attente du lock")
    with model_lock :
        print(f"{threading.current_thread().name} a acquis le lock")
        # --- Chroma search ---
        print('Début de la recherche sémantique')
        db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        filter_chroma = None

        if doc_filter:
            if len(doc_filter) == 1:
                filter_chroma = {"doc_name": str(doc_filter[0])}
            elif len(doc_filter) > 1:
                filter_chroma = {"doc_name": {"$in": [str(name) for name in doc_filter]}}
        print(filter_chroma)

        results_chroma = db.similarity_search_with_score(query, k=top_k, filter=filter_chroma)
        chroma_scores = {d.metadata["id"]: score for d, score in results_chroma}
    print(f"{threading.current_thread().name} a libéré le lock")
    
    # --- Fusion par Reciprocal Rank Fusion ---
    all_ids = set(bm25_scores) | set(chroma_scores)
    fused = {}
    for doc_id in all_ids:
        rank_bm25 = list(bm25_scores.keys()).index(doc_id) + 1 if doc_id in bm25_scores else top_k + 1
        rank_chroma = list(chroma_scores.keys()).index(doc_id) + 1 if doc_id in chroma_scores else top_k + 1
        fused[doc_id] = alpha / (k_rrf + rank_bm25) + (1-alpha) / (k_rrf + rank_chroma)


    fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [id for id,_ in fused_sorted[:top_k]]
    if len(top_doc_ids) == 1 :
        top_docs_filter = {"id": top_doc_ids[0]}
    elif len(top_doc_ids) > 1 :
        top_docs_filter = {"$or": [{"id": id} for id in top_doc_ids]}
    else :
        return []
    top_docs_dict = db.get(
        where=top_docs_filter,          
        include=["documents","metadatas"]      
    )
    top_docs = [Document(page_content=content, metadata=metadata) for content, metadata in zip(top_docs_dict["documents"],top_docs_dict["metadatas"])]
    doc_map = {doc.metadata["id"]: doc for doc in top_docs}
    ordered_docs = [doc_map[doc_id] for doc_id in top_doc_ids if doc_id in doc_map]
    return ordered_docs

# fonction de rerank 
# Fonction de reranking 
from my_paths import RERANKER_PATH
def rerank_chunks(context, query, prefetch) : 
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=RERANKER_PATH,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    model.to('cuda') # or 'cpu' if no GPU is available
    model.eval()
    documents = [doc.page_content for doc in context]
    result = model.rerank(
        query,
        documents,
        max_query_length=512,
        max_length=1024,
        top_n=prefetch
    )
    context = [context[result[i]['index']] for i in range(len(result))]
    #print(f"Nouveau classement : {[result[i]['index'] for i in range(len(result))]}")
    return(context)