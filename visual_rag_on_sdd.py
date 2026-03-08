import torch
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
import os
import numpy as np
from tqdm import tqdm 
from time import perf_counter as time
import re
import io
import base64

from my_paths import COLSMOL_PATH, PAGES_EMBEDDINGS_PATH
from utils import clean_folder_name

# Charger modèle ET embeddings une fois et conserver dans global
model = None
processor = None
embeddings = None

# Fonction pour charger le modèle et le processeur
def load_model_and_processor():
    colsmol_path = COLSMOL_PATH

    model = ColIdefics3.from_pretrained(
        colsmol_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()

    processor = ColIdefics3Processor.from_pretrained(colsmol_path)
    return model, processor

def load_embeddings_from_npy(storage_file):
    if os.path.exists(storage_file):
        embeddings_data = np.load(storage_file, allow_pickle=True)
        return embeddings_data
    else:
        print(f"Aucun fichier de stockage trouvé à {storage_file}")
        return None

def load_model_and_embeddings():
    global model, processor, embeddings
    start = time()
    model, processor = load_model_and_processor()
    end = time()
    print(f"Temps total pour charger le modèle Colsmol : {end-start}")
    start = time()
    embeddings = load_embeddings_from_npy(PAGES_EMBEDDINGS_PATH).item()
    end = time()
    print(f"Temps total pour charger les embeddings visuels des SDD : {end-start}")

# Fonction pour calculer l'embedding d'une page PDF
def compute_page_embedding(pdf_path, page_num, processor, model):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # zoom x2 pour meilleure qualité
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    batch_image = processor.process_images([img]).to(model.device)

    with torch.no_grad():
        image_embedding = model(**batch_image)

    return image_embedding

# Fonction pour calculer l'embedding d'un requête
def compute_query_embedding(query, processor, model):
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    return(query_embeddings)

# Fonction pour calculer les scores avec l'algorithme MaxSim
def get_scores(query_embeddings, image_embeddings, processor) : 
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    return(scores)

# Fonction pour calculer et stocker les embeddings pour un document PDF
def process_document_and_store_embeddings(pdf_path, data_dir, processor, model, storage_file):
    doc = fitz.open(pdf_path)
    already_processed = False
    # Vérifier si le fichier de stockage existe déjà
    embeddings_data = {}
    if os.path.exists(storage_file):
        embeddings_data = np.load(storage_file, allow_pickle=True).item()

    # Parcourir les pages du PDF et calculer les embeddings
    for page_num in range(doc.page_count):
        relative_path = pdf_path.relative_to(data_dir)
        key = f"{relative_path}_{page_num + 1}" # On stocke le "vrai" numéro de page
        
        # Si l'embedding pour cette page existe déjà, on le saute
        if key in embeddings_data:
            print(f"Pdf {pdf_path} déjà traité.")
            already_processed = True
            break

        print(f"Traitement de la page {page_num + 1}...")
        
        # Calcul de l'embedding de la page
        image_embedding = compute_page_embedding(pdf_path, page_num, processor, model)
        
        # Sauvegarder l'embedding dans le dictionnaire
        embeddings_data[key] = image_embedding.cpu().to(torch.float32).numpy()

    doc.close()

    # Sauvegarder les embeddings dans un fichier .npy
    if not already_processed :
        np.save(storage_file, embeddings_data)
        print(f"Embeddings sauvegardés dans {storage_file}")
    
# Fonction pour obtenir tous les chemins de documents SDD
def get_every_SDD_document_path(data_dir):
    all_SDD_files = [f for f in Path(data_dir).rglob("*") if 'SDD' in str(f) and f.suffix==".pdf"]
    return all_SDD_files

def parse_path_and_page(input_str):
    # Regex qui capture :
    # - tout ce qui précède un fichier se terminant par .pdf
    # - puis un underscore '_'
    # - puis un ou plusieurs chiffres à la fin de la chaîne
    pattern = re.compile(r'^(.*?\.pdf)_(\d+)$')
    match = pattern.match(input_str)
    
    if not match:
        raise ValueError("Le format ne correspond pas à '<chemin>.pdf_<numéro_page>'")
    
    path = match.group(1)
    page_number = int(match.group(2))
    
    return path, page_number

def retrieve_pages(query, processor, model, embeddings, top_k=4) :
    start = time()
    query_embedding = compute_query_embedding(query, processor, model)
    scores_dict = {}
    for key, image_embedding in tqdm(list(embeddings.items())) :
        image_embedding = torch.tensor(image_embedding, dtype=torch.bfloat16)
        scores_dict[key] = get_scores(query_embedding, image_embedding, processor)
    results = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    end = time()
    top_k_results = [parse_path_and_page(key) for key,_ in results[:top_k]]

    print(f"Temps total pour calculer l'embedding de la query + les scores des documents: {end-start}")
    return(top_k_results)


def extract_pdf_page_as_base64(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    # Attention : PyMuPDF utilise un index 0-based pour les pages
    page_index = page_num - 1
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=96)  # Rend un rendu de la page en image
    #pix.save(f"pages_images/{page_num}.png")
    # Convertir pixmap en image PIL pour pouvoir l'encodage sans sauvegarder sur disque
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Encoder en base64
    page_base64 = base64.b64encode(img_bytes).decode('utf-8')
    doc.close()
    return pdf_path, page_num, page_base64

def prepare_messages(query, retrieved_pages, data_dir, save_context=True, output_dir="retrieved_context") : 
    base64_pages = [] 
    for i in range(len(retrieved_pages)) :
        relative_pdf_path, page_num = retrieved_pages[i]
        pdf_path = Path(data_dir) / relative_pdf_path
        pdf_path, page_num, page_base64 = extract_pdf_page_as_base64(pdf_path, page_num)
        base64_pages.append((pdf_path, page_num, page_base64))
        if save_context :
            image_data = base64.b64decode(page_base64)
            folder_name = clean_folder_name(query)
            context_path =  os.path.join(output_dir, folder_name)
            img_path = Path(context_path) / f"{Path(relative_pdf_path).stem}_page_{page_num}_rank_{i+1}.png"
            if not Path(context_path).exists() :
                Path(context_path).mkdir(parents=True, exist_ok=True)
            with open(img_path,"wb") as f :
                f.write(image_data)
    
    system_prompt = (
        f"You are a useful assistant that answers queries about an aerospace project based on the given context."
        f"INSTRUCTIONS :"
        f"- You MUST NOT invent any information or use any of your knowledge. ONLY answer based on the provided context. If you can't answer given the context, output that you were unable to find informations."
        " - You MUST explicitly and systematically quote the sources used to generate each paragraph of information."
        " - For every distinct paragraph or semantic unit you output, indicate clearly the exact document name and the page number from which the information was retrieved. "
        " - This citation must immediately follow the related paragraph. Do not omit or group citations; each informational paragraph must be individually sourced."
        f"- When asked about a computation, in addition to the document name and page, you must include the names of the requirements where it is referenced (e.g. the name of the LLR, low level requirement)."
        f"- You must also mention the references (names of the LLR) of everything (e.g. every inputs) that is needed for the computation."
        f"- Keep in mind the user do not have access to the context provided to you, so do not say 'the provided context' as if the user had access to it. "
    )
    user_content = [
        {
            "type": "text",
            "text": f"Answer the following query based on the given context (which is made of pdf page images) : {query}"
        }
    ]
    for pdf_path, page_num, page_base64 in base64_pages :
        user_content.append({
            "type": "text",
            "text": f"Page number {page_num} from document {Path(pdf_path).stem}"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page_base64}"
            }
        })

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    return(messages)

# Fonctions principales

def process_sdd_visual_embeddings(data_dir, storage_file):
    global model, processor 
    if (not processor) or (not model) :
        model, processor = load_model_and_processor()
    # Fichier de stockage des embeddings
    pdf_paths = get_every_SDD_document_path(data_dir)
    # Processer tous les fichiers PDF
    for pdf_path in pdf_paths:
        print(f"Traitement du document: {pdf_path}")
        process_document_and_store_embeddings(pdf_path, data_dir, processor, model, storage_file)

from mistral_langchain_wrapper import MistralChatWrapper
import os 
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')

def retrieve_SDD_pages_and_answer_query(query, data_dir, llm_model='medium') :
    global model, processor, embeddings
    if (not model) or (not processor) or (not embeddings) :
        load_model_and_embeddings()
    retrieved_pages = retrieve_pages(query, processor, model, embeddings, top_k=7)
    messages = prepare_messages(query, retrieved_pages, data_dir)
    llm = MistralChatWrapper(api_key=API_KEY, model=llm_model)
    response = llm.invoke(messages).content
    return(response, retrieved_pages)
