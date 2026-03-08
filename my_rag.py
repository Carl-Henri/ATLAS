import os
import base64
import re

from hybrid_search import hybrid_search
from utils import clean_folder_name

from my_paths import PARSED_DATA_DIR, CHROMA_DIR, WHOOSH_DIR, MISTRAL_TOKENIZER_PATH, RERANKER_PATH

# Postprocessing des chunks récupérés 
import os
import base64
import re
import shutil

from mistral_langchain_wrapper import MistralChatWrapper
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')

def find_subfolder_path(root_folder, doc_name):
    """
    Cherche récursivement un sous-dossier nommé doc_name dans root_folder.
    Retourne le chemin complet du dossier s'il est trouvé, sinon None.
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if doc_name in dirnames:
            return os.path.join(dirpath, doc_name)
    return None

def postprocess_chunk_with_figures(chunks, query, max_images=3, save_context=True, output_dir="retrieved_context", data_dir=PARSED_DATA_DIR):
    """
    Transforme un chunk contenant des références à des figures ou tables en une structure
    de messages conforme avec blocs texte et images en base64.
    Lorsque le nombre max d'images est atteint, on n'insère plus d'images et on renvoie le texte intact.
    
    Args:
        chunks (list of str): Liste de textes des chunks avec références comme <!-- Figure number: 20 --> ou <!-- Table number: 1 -->
        max_images (int): Nombre maximum d’images à insérer.
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
    folder_name = clean_folder_name(query)
    output_dir =  os.path.join(output_dir, folder_name)
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
def create_messages(question, context_message, history) : 
    system_prompt = (
        f"You are a useful assistant that answers queries about an aerospace project based on the given context."
        f"Intructions about HOW TO ANSWER :"
        f"  - You MUST NOT invent any information or use any of your knowledge. ONLY answer based on the provided context. If you can't answer given the context, output that you were unable to find informations."
        "   - You MUST explicitly and systematically quote the sources used to generate each paragraph of information."
        "   - For every distinct paragraph or semantic unit you output, indicate clearly the exact document name and the page number (or sheet name if it is an Excel file) from which the information was retrieved. "
        "   - This citation must immediately follow the related paragraph. Do not omit or group citations; each informational paragraph must be individually sourced."
        "Instructions about HOW TO PROCESS THE PROVIDED CONTENT :"
        f"  - In the context provided, there are <!-- Figure caption: <CAPTION> --> that describes the figure above ; do not quote it."
        f"  - There are also tags like <!-- Table markdown 1 start --> MARKDOWN TABLE <!-- Table markdown end --> that indicates a markdown table. DO NOT quote these tags directy. "
        f"  - You MUST NOT quote the number of the table referenced in these markdown tags."
        f"  - Refer to the page number (or sheet name if it is an Excel file) and the name of the document given at the start of each chunk."
        f"  - The chunk hierarchy included at the beginning of each context chunk is provided to give you information about the source of the information."
        f"  - Keep in mind the user do not have access to the context provided to you, so do not say 'the provided context' as if the user had access to it. "
        f"  - For chunks from excel files, quote the document name, the sheet name, and the row and column numbers."
        "   - For chunks from html files, the number of the chunk is referenced at the beginning of each chunk to help you reconstitute the document. "
        "   - DO NOT quote this chunk number (e.g. chunk 11/32)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if context_message :
        messages.append(context_message)
    if history :
        messages += history[:-1]
        
    messages += [
        {"role": "user", "content": f"Answer the following question based on the context items given : {question}"}
    ]

    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages)) 
    return(messages, len(tokenized.tokens))

# Interroger la DB puis générer la réponse
def retrieve_and_answer_query(query, history=None, # Messages à envoyer au LLM 
                              chroma_dir = str(CHROMA_DIR), whoosh_dir = str(WHOOSH_DIR), data_dir=PARSED_DATA_DIR, # Quelles données utiliser
                              model = 'medium', rerank=False, doc_filter=None, prefetch=400, tokens_to_send = 50000, alpha = 0.5 # Paramètres du RAG
                              ) :
 
    context = hybrid_search(chroma_dir=chroma_dir, whoosh_dir=whoosh_dir, query=query, top_k=prefetch, alpha=alpha, doc_filter=doc_filter, k_rrf=30)
    if rerank : 
        from hybrid_search import rerank_chunks
        context = rerank_chunks(context, query, prefetch)
    # Mistral a une fenêtre de contexte de 128k tokens, donc on laisse 10k tokens pour la réponse (i.e. 8k mots anglais ce qui est ok, soit environ 320 lignes)
    # Préparer les messages pour l'API
    max_messages_tokens = tokens_to_send + 5000
    min_messages_tokens = tokens_to_send - 5000
    pas_chunks = 8
    nb_chunks_envoyes = int(tokens_to_send/1000) # Nombre de chunks à essayer d'envoyer initialement. A adapter en fonction de tokens_to_send. Ici j'ai mis /1000 en estiment 1 chunk = 1000 tokens en moyenne
    context_to_send = context[:nb_chunks_envoyes]
    context_with_images = postprocess_chunk_with_figures(context_to_send, query, data_dir=data_dir)
    messages, nb_tokens = create_messages(query, context_with_images, history)

    # D'abord on essaie d'ajouter des chunks si il y en a trop peu
    while nb_tokens < min_messages_tokens and nb_chunks_envoyes+pas_chunks < prefetch:
        nb_chunks_envoyes += pas_chunks # On enlève les derniers chunks
        context_to_send = context[:nb_chunks_envoyes]
        context_with_images = postprocess_chunk_with_figures(context_to_send, query, data_dir=data_dir)
        print(f"Peu de tokens : {nb_tokens}. On augmente de {pas_chunks} chunks. Essai avec {nb_chunks_envoyes} chunks.")
        messages, nb_tokens = create_messages(query, context_with_images, history)
    
    # Ensuite, on en retire s'il y en a trop 
    while nb_tokens > max_messages_tokens :
        nb_chunks_envoyes -= pas_chunks # On enlève les derniers chunks
        context_to_send = context[:nb_chunks_envoyes]
        context_with_images = postprocess_chunk_with_figures(context_to_send, query, data_dir=data_dir)
        print(f"Trop de tokens : {nb_tokens}. On réduit de {pas_chunks} chunks. Essai avec {nb_chunks_envoyes} chunks.")
        messages, nb_tokens = create_messages(query, context_with_images, history)

    print(f"Nombre de tokens envoyés à l'API Mistral : {nb_tokens}. Nombre de chunks envoyés : {nb_chunks_envoyes}")
    llm = MistralChatWrapper(api_key=API_KEY, model=model)
    response = llm.invoke(messages).content
    return response, context_to_send
