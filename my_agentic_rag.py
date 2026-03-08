
from hybrid_search import hybrid_search
from utils import clean_folder_name

from my_paths import PARSED_DATA_DIR, CHROMA_DIR, WHOOSH_DIR, MISTRAL_TOKENIZER_PATH

# Postprocessing des chunks récupérés 
import os
import base64
import re
import shutil
from pathlib import Path

def find_subfolder_path(root_folder, doc_name):
    """
    Cherche récursivement un sous-dossier nommé doc_name dans root_folder.
    Retourne le chemin complet du dossier s'il est trouvé, sinon None.
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if doc_name in dirnames:
            return os.path.join(dirpath, doc_name)
    return None

def postprocess_chunk_with_figures(chunks, output_dir, max_images=3, save_context=True, data_dir=PARSED_DATA_DIR):
    """
    Transforme un chunk contenant des références à des figures ou tables en une structure
    de messages conforme avec blocs texte et images en base64.
    Lorsque le nombre max d'images est atteint, on n'insère plus d'images et on renvoie le texte intact.
    
    Args:
        chunks (list of str): Liste de textes des chunks avec références comme <!-- Figure number: 20 --> ou <!-- Table number: 1 -->
        max_images (int): Nombre maximum d'images à insérer.
        save_context (bool): Si True, sauvegarde le contexte brut et le contexte traité dans des fichiers .txt et les images dans un dossier.
        output_dir (str): Dossier où les fichiers et images récupérés seront sauvegardés.

    Returns:
        dict: Messages conforme avec le format décrit.
    """
    # Expression régulière pour capturer les numéros de figures et tables
    pattern = r"<!--\s*(Figure|Table) number:\s*(\d+)\s*-->"

    parts = []
    nb_figure = 0
    max_reached = False
    all_raw_text = ""
    image_filenames = []
    # Créer / écraser le dossier de sauvegarde si nécessaire
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)    
    os.makedirs(output_dir, exist_ok=True)
    
    for chunk_content, doc_name in zip([c.page_content for c in chunks], [c.metadata['doc_name'] for c in chunks]):
        # Sauvegarder le contexte brut si `save_context` est activé
        if save_context:
            all_raw_text += chunk_content + "\n\n"

        if max_reached:
            # Plus de découpage, on ajoute le chunk complet en texte brut
            text = chunk_content.strip()
            if text:
                parts.append({"type": "text", "text": text})

        else:
            last_pos = 0
            matches = list(re.finditer(pattern, chunk_content))
            if matches:
                base_path = find_subfolder_path(data_dir, doc_name)

            for match in matches:
                start, end = match.span()
                element_type = match.group(1)  # "Figure" ou "Table"
                element_num = match.group(2)  # Le numéro de l'élément

                if nb_figure < max_images:
                    # Ajouter le texte avant l'élément (figure ou table)
                    text_chunk = chunk_content[last_pos:start].strip()
                    if text_chunk:
                        parts.append({"type": "text", "text": text_chunk})

                    # Traiter les images pour les figures
                    if element_type == "Figure":
                        img_path = os.path.join(base_path, "pictures", f"picture-{element_num}.png")
                        if os.path.isfile(img_path):
                            with open(img_path, "rb") as img_file:
                                img_bytes = img_file.read()
                                # Sauvegarde de l'image dans le dossier "retrieved_context"
                                img_filename = f"figure-{element_num}.png"
                                img_output_path = os.path.join(output_dir, img_filename)
                                with open(img_output_path, "wb") as out_img_file:
                                    out_img_file.write(img_bytes)
                                image_filenames.append(img_filename)
                                parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
                                    }
                                })
                            nb_figure += 1
                        else:
                            parts.append({
                                "type": "text",
                                "text": f"[Image picture-{element_num}.png introuvable]"
                            })

                    # Traiter les images pour les tables
                    elif element_type == "Table":
                        table_path = os.path.join(base_path, "tables", f"table-{element_num}.png")
                        if os.path.isfile(table_path):
                            with open(table_path, "rb") as table_file:
                                table_bytes = table_file.read()
                                # Sauvegarde de l'image dans le dossier "retrieved_context"
                                table_filename = f"table-{element_num}.png"
                                table_output_path = os.path.join(output_dir, table_filename)
                                with open(table_output_path, "wb") as out_table_file:
                                    out_table_file.write(table_bytes)
                                image_filenames.append(table_filename)
                                parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64.b64encode(table_bytes).decode('utf-8')}"
                                    }
                                })
                            nb_figure += 1
                        else:
                            parts.append({
                                "type": "text",
                                "text": f"[Image table-{element_num}.png introuvable]"
                            })

                    # Si l'image est ajoutée, on ne garde pas le placeholder
                    last_pos = end

                else:
                    max_reached = True
                    break

            tail_text = chunk_content[last_pos:].strip()
            if tail_text:
                parts.append({"type": "text", "text": tail_text})

    # Sauvegarder le contexte brut et traité dans des fichiers .txt
    if save_context:
        # Sauvegarder le contexte brut
        raw_context_path = os.path.join(output_dir, "raw_context.txt")
        with open(raw_context_path, "w", encoding="utf-8") as raw_file:
            raw_file.write(all_raw_text)

        # Sauvegarder le contexte traité avec les liens locaux pour les images
        processed_context_path = os.path.join(output_dir, "processed_context.txt")
        with open(processed_context_path, "w", encoding="utf-8") as processed_file:
            n_img = 0
            for part in parts:
                if part["type"] == "text":
                    processed_file.write(part["text"] + "\n")
                elif part["type"] == "image_url":
                    # Remplacer les données base64 par le chemin local dans processed_context.txt
                    file_name = image_filenames[n_img]
                    processed_file.write(f"[Image file name: {file_name}]\n")
                    n_img+=1
    
    # Retourner les messages sans changer la structure originale
    messages = {"role": "user", "content": parts}
    return messages

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
tokenizer =  MistralTokenizer.from_file(MISTRAL_TOKENIZER_PATH)
# Fonction utilitaire pour créer les messages envoyés à l'API 
def create_messages(question, context_message) : 
    system_prompt = """
You are a useful assistant that extracts information about an aerospace project based solely on the given context.
Your task is to output relevant, clear, and complete informational paragraphs strictly from the provided text, figures, markdown tables or CSV data without inventing or adding any knowledge.
You MUST use only the data contained in the context.
For each distinct paragraph or semantic unit you output:
- Refer to the page number (or sheet name if it is an Excel file) and the name of the document given at the start of each chunk.
- The chunk hierarchy included at the beginning of each context chunk is provided to give you information about the source of the information.
- You MUST provide an explicit and precise citation immediately following the paragraph, indicating the exact document names and page numbers (or Excel sheet names).
- For Excel files (CSV data, file name that ends with .xls or .xlsx), cite the document name, sheet name, and the row and column numbers of the extracted data.
- Do NOT group multiple extracted paragraphs under a single citation; each paragraph must have its own reference.
- Do NOT quote figure captions (noted by <!-- Figure caption: ... -->) or markdown/technical tags; however, if a table caption exists, you MUST quote it.
- Do NOT quote table numbers referenced in the markdown tags (like <!-- Table markdown 15 start -->). 
- For chunks from html files, the number of the chunk is referenced at the beginning of each chunk to help you reconstitute the document. 
- DO NOT quote this chunk number (e.g. chunk 11/32).
If you are unable to find information to extract relevant content, you MUST clearly state that you were unable to find any information.
Do NOT provide a direct answer or summary; instead, output a factual, sourced extraction from the given context.
You MUST extract ALL the information relevant to the user query ; without reformulating it or summarizing it. 
"""
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if context_message :
        messages.append(context_message)
        
    messages += [
        {"role": "user", "content": f"Extract informations from the provided context that are useful to answer the following query : {question}"}
    ]
    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages)) 
    return(messages, len(tokenized.tokens))

def create_messages_with_ajusted_tokens(query, context, tokens_to_send, data_dir, output_dir) :
    prefetch = len(context)
    margin = 0.1
    max_messages_tokens = int(tokens_to_send*(1+margin))
    min_messages_tokens = int(tokens_to_send*(1-margin))
    pas_chunks = 5
    nb_chunks_envoyes = prefetch // 8
    context_to_send = context[:nb_chunks_envoyes]
    context_with_images = postprocess_chunk_with_figures(context_to_send, output_dir, data_dir=data_dir)
    messages, nb_tokens = create_messages(query, context_with_images)

    # D'abord on essaie d'ajouter des chunks si il y en a trop peu
    while nb_tokens < min_messages_tokens and nb_chunks_envoyes+pas_chunks < prefetch:
        nb_chunks_envoyes += pas_chunks # On enlève les derniers chunks
        context_to_send = context[:nb_chunks_envoyes]
        context_with_images = postprocess_chunk_with_figures(context_to_send, output_dir, data_dir=data_dir)
        #print(f"Peu de tokens : {nb_tokens}. On augmente de {pas_chunks} chunks. Essai avec {nb_chunks_envoyes} chunks.")
        messages, nb_tokens = create_messages(query, context_with_images)
    
    # Ensuite, on en retire s'il y en a trop 
    while nb_tokens > max_messages_tokens :
        nb_chunks_envoyes -= pas_chunks # On enlève les derniers chunks
        context_to_send = context[:nb_chunks_envoyes]
        context_with_images = postprocess_chunk_with_figures(context_to_send, output_dir, data_dir=data_dir)
        #print(f"Trop de tokens : {nb_tokens}. On réduit de {pas_chunks} chunks. Essai avec {nb_chunks_envoyes} chunks.")
        messages, nb_tokens = create_messages(query, context_with_images)
    return(messages, nb_tokens, nb_chunks_envoyes)

def create_final_messages(query, str_responses) :
    system_prompt = (
        f"You are a useful assistant that answers queries about an aerospace project based on a given context."
        "The provided context is composed of a set of extracted informations from technical documents of the project related to the query by a LLM."
        "Your goal is to use the informations provided to create a complete and refined response to the query."
        "INSTRUCTIONS :"
        f"- You MUST NOT invent any information or use any of your knowledge. ONLY answer based on the provided context. If you can't answer given the context, output that you were unable to find informations."
        "- You MUST explicitly and systematically quote the sources used to generate each paragraph of information."
        "- For every distinct paragraph or semantic unit you output, indicate clearly the exact document name and the page number (or sheet name if it is an Excel file) from which the information was retrieved. "
        "- This citation must immediately follow the related paragraph. Do not omit or group citations; each informational paragraph must be individually sourced."
        "- You MUST avoid any phrasing that implies the user has access to the provided context (for example, do NOT say 'in the provided context')."
        "- Only output the complete and refined response to the query."

    )
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    messages.append({"role":"user", "content":f"Now answer the following query base on the following context.\nQuery : {query}.\nContext: {str_responses}"})
    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages)) 
    return(messages, len(tokenized.tokens))
from mistral_langchain_wrapper import MistralChatWrapper
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
import time
from concurrent.futures import ThreadPoolExecutor

MAX_RPM = 100
DELAY = 60 / MAX_RPM   # 0.6 sec entre appels

def process_iteration(args):
    (it, query, remaining_context, tokens_to_send, data_dir, output_subdir, model) = args

    # Créer les messages
    messages, nb_tokens, send_chunks = create_messages_with_ajusted_tokens(
        query,
        remaining_context,
        tokens_to_send,
        data_dir,
        output_subdir / f"response_{it}"
    )

    # Appel API (protégé par délai)
    time.sleep(DELAY)
    llm = MistralChatWrapper(api_key=API_KEY, model=model)
    response = llm.invoke(messages).content

    return response, nb_tokens, send_chunks


# Interroger la DB puis générer la réponse
def agentic_retrieve_and_answer_query(query, # Messages à envoyer au LLM 
                              chroma_dir = str(CHROMA_DIR), whoosh_dir = str(WHOOSH_DIR), data_dir=PARSED_DATA_DIR, # Quelles données utiliser
                              model='medium', doc_filter=None, rerank=False, prefetch=400, min_total_tokens_to_send = 100000, tokens_to_send=20000, alpha = 0.5, # Paramètres du RAG
                              output_dir="agentic_retrieved_context" # Sauvegarde du contexte récupéré
                              ) :
    
    context = hybrid_search(chroma_dir=chroma_dir, whoosh_dir=whoosh_dir, query=query, top_k=prefetch, alpha=alpha, doc_filter=doc_filter, k_rrf=30)
    if rerank : 
        from hybrid_search import rerank_chunks
        context = rerank_chunks(context, query, prefetch)
    print(f"{len(context)} documents récupérés")
    output_subdir = Path(output_dir) / clean_folder_name(query)
    # Créer / écraser le dossier de sauvegarde si nécessaire
    if os.path.exists(output_subdir) and os.path.isdir(output_subdir):
        shutil.rmtree(output_subdir)    
    os.makedirs(output_dir, exist_ok=True)

    responses = []
    tasks_args = []

    total_send_chunks = 0
    total_send_tokens = 0
    it = 0

    while total_send_tokens < min_total_tokens_to_send and total_send_chunks < len(context):

        remaining_context = context[total_send_chunks:]

        # Préparer les arguments
        tasks_args.append((
            it, query, remaining_context, tokens_to_send, data_dir, output_subdir, model
        ))

        # On fait tourner pour connaître les tailles des chunks
        messages, nb_tokens, send_chunks = create_messages_with_ajusted_tokens(
            query,
            remaining_context,
            tokens_to_send,
            data_dir,
            output_subdir / f"response_{it}"
        )

        total_send_chunks += send_chunks
        total_send_tokens += nb_tokens
        it += 1


    # Exécution parallèle contrôlée
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(process_iteration, tasks_args))

    # Extraire les réponses
    responses = [r[0] for r in results]

    str_responses = "\n\n".join(f"Response {i}\n{response}" for i, response in enumerate(responses))
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    with open(output_subdir / 'responses.txt', 'w', encoding="utf-8") as f :
        f.write(str_responses)
    final_messages, nb_final_tokens = create_final_messages(query, str_responses)
    llm = MistralChatWrapper(api_key=API_KEY, model=model)
    final_response = llm.invoke(final_messages).content
    print(f"Nombre de tokens envoyés à l'API Mistral : {nb_final_tokens}")
    print(f"Nombre de chunks envoyés à l'API Mistral : {total_send_chunks}")
     
    print(f"Nombre total de tokens envoyés à l'API Mistral : {total_send_tokens + nb_final_tokens}")
    return final_response, context[:total_send_chunks]
