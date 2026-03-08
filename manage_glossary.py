from pathlib import Path
import json
import re
from my_paths import GLOSSARY_PATH, CHROMA_DIR, WHOOSH_DIR

from mistral_langchain_wrapper import MistralChatWrapper
import os 
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
llm = MistralChatWrapper(api_key=API_KEY, model="medium")


from whoosh import index
from whoosh.qparser import QueryParser, OrGroup

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from my_paths import MISTRAL_TOKENIZER_PATH

tokenizer =  MistralTokenizer.from_file(MISTRAL_TOKENIZER_PATH)

def key_word_search(query, whoosh_dir, limit) :
    # --- BM25 Whoosh ---
    ix = index.open_dir(whoosh_dir)
    with ix.searcher() as searcher:
        parser = QueryParser("content", schema=ix.schema, group=OrGroup)
        q = parser.parse(query)
        print(f"Query à whoosh: {q}")
        results = searcher.search(q, limit=limit)
        print(f"Nombre de documents trouvés par Whoosh : {len(results)}")
        res = [r["content"] for r in results]
    return res

import difflib
import json

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def merge_glossaries(glossary1, glossary2):
    for key, definitions in glossary2.items():
        if key in glossary1:
            # Ajouter les définitions de glossary2 à glossary1, sans doublons
            for definition in definitions:
                if definition not in glossary1[key]:
                    glossary1[key].append(definition)
        else:
            # Clé absente, on copie la liste de définitions
            glossary1[key] = definitions.copy()

def query_glossary(acronym, glossary_path=GLOSSARY_PATH):
    with open(glossary_path, 'r', encoding='utf-8') as glossary_file:
        try:
            glossary = json.load(glossary_file)
        except:
            glossary = {}
    
    with open('AAA_glossary.json', 'r', encoding='utf-8') as AAA_glossary_file: 
        try: 
            AAA_glossary = json.load(AAA_glossary_file)
        except :
            AAA_glossary = {}
  
    merge_glossaries(glossary, AAA_glossary)

    scored_results = []
    for key in glossary:
        score = similarity(acronym.lower(), key.lower())
        scored_results.append((score, key, glossary[key]))

    # Trier par similarité décroissante
    scored_results.sort(key=lambda x: x[0], reverse=True)

    if not scored_results:
        return 'Acronyme non trouvé dans le glossaire.'

    top_results = scored_results[:3]

    # Seuils et différences
    seuil_presque_1 = 0.99
    diff_max_proche = 0.1

    premier_score = top_results[0][0]

    if premier_score < 0.1:
        # Aucun résultat vraiment pertinent
        return 'Acronyme non trouvé dans le glossaire.'

    # Si premier score est quasi 1, on renvoie juste le premier
    if premier_score >= seuil_presque_1:
        score, key, definition = top_results[0]
        return [(key, definition)]

    # Sinon on regarde la différence max entre scores des 3 premiers
    scores = [score for score, _, _ in top_results]
    if max(scores) - min(scores) <= diff_max_proche:
        # Ils sont proches, on renvoie les 3
        return [(key, definition) for score, key, definition in top_results]

    # Sinon, on renvoie juste le premier
    score, key, definition = top_results[0]
    return [(key, definition)]

def complete_glossary(response, glossary_path) : 
    added = False
    try :
        with open(glossary_path,'r', encoding='utf-8') as glossary_file : 
            glossary = json.loads(glossary_file.read())
    except : 
        glossary = {}

    regex = r"```json\s*({.*?})\s*```"
    match = re.search(regex, response, re.DOTALL)
    try : 
        extracted_json = match.group(1)
        extracted_json = json.loads(extracted_json)
    except :
        print("Erreur de parsing. Aucun ajout.")
        return(False)

    for acronym, definition in extracted_json.items() :
        if acronym in glossary : 
            print(f'Acronyme déjà présent : {acronym}')
            new_def = True
            for actual_definition in glossary[acronym] :
                if actual_definition.strip().lower() == definition.strip().lower() :
                    new_def = False
            if new_def:
                print(f'Nouvelle définition : {glossary[acronym]} ; {definition}, ajout.')
                glossary[acronym].append(definition)
                added = True
        else :
            print(f"Ajout de l'acronyme : {acronym}")
            glossary[acronym] = [definition]
            added = True

    
    with open(glossary_path,'w',encoding='utf-8') as glossary_file :
        json.dump(glossary, glossary_file, ensure_ascii=False, indent=2)
    
    return added

def create_messages(str_context) : 
    system_prompt = """
    Extract all acronyms from the following text, along with their definitions.
    Provide the results in a JSON format where each acronym is a key of type str, and the corresponding definition is the value of type str.
    Only output the json like this : ```json <JSON OUTPUT> ```
    """
    messages = [
        {'role':'system', 'content': system_prompt},
        {'role':'user', 'content': f"Here is the text to extract the acronyms from : {str_context}"}
    ]
    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
    nb_tokens = len(tokenized.tokens)
    return(messages,nb_tokens)

def create_glossary(whoosh_dir, glossary_path) : 
    if Path(glossary_path).is_file() :
        print(f'Glossaire déjà existant.')
        return
    print(f'Création du glossaire {glossary_path}')
    query = "Glossary, acronym definitions, definition, acronyms"
    docs_retrieved = key_word_search(query, whoosh_dir, limit = 1000)
    print(f"Nombre de docs total récupérés : {len(docs_retrieved)}")
    # Paramètres de contrôle
    min_messages_tokens = 40000     # seuil minimal de tokens souhaité
    max_messages_tokens = 50000    # seuil maximal de tokens accepté
    pas_chunks = 5                # nombre de chunks ajoutés/retraits à chaque itération

    nb_chunks_ini = 10       # nombre initial de chunks à envoyer
    nb_useless_it = 0   # compter les itérations successives sans ajout de définitions
    processed_chunks = 0
    total_chunks = len(docs_retrieved)

    while processed_chunks < total_chunks:

        # Limite les chunks à traiter selon ce qui reste et la taille destination max
        nb_chunks_to_try = min(nb_chunks_ini, total_chunks - processed_chunks)
        docs_to_process = docs_retrieved[processed_chunks:processed_chunks+nb_chunks_to_try]

        # Concaténation du contenu des chunks
        #str_context = "\n\n".join([doc.page_content for doc in docs_to_process])
        str_context = "\n\n".join(docs_to_process)
        # Création initiale des messages et calcul du nombre de tokens
        messages, nb_tokens = create_messages(str_context)

        nb_chunks_envoyes = nb_chunks_to_try

        # Ajuste le nombre de chunks envoyés pour que le nb_tokens soit dans la plage souhaitée
        # On augmente les chunks si trop peu de tokens et qu'on peut en ajouter
        while nb_tokens < min_messages_tokens and nb_chunks_envoyes + pas_chunks < total_chunks - processed_chunks :
            nb_chunks_envoyes += pas_chunks
            docs_to_process = docs_retrieved[processed_chunks:processed_chunks+nb_chunks_envoyes]
            #str_context = "\n\n".join([doc.page_content for doc in docs_to_process])
            str_context = "\n\n".join(docs_to_process)
            messages, nb_tokens = create_messages(str_context)
            print(f"Peu de tokens ({nb_tokens}). On ajoute {pas_chunks} chunks; total chunks = {nb_chunks_envoyes}")

        # On réduit les chunks si trop de tokens
        while nb_tokens > max_messages_tokens :
            nb_chunks_envoyes -= pas_chunks
            docs_to_process = docs_retrieved[processed_chunks:processed_chunks+nb_chunks_envoyes]
            #str_context = "\n\n".join([doc.page_content for doc in docs_to_process])
            str_context = "\n\n".join(docs_to_process)
            messages, nb_tokens = create_messages(str_context)
            print(f"Trop de tokens ({nb_tokens}). On retire {pas_chunks} chunks; total chunks = {nb_chunks_envoyes}")

        print(f"Envoi de {nb_chunks_envoyes} chunks avec {nb_tokens} tokens.")

        # Génération de la réponse à partir des messages
        response = llm.invoke(messages).content

        # Traitement de la réponse pour compléter le glossaire
        added = complete_glossary(response, glossary_path)
        if not(added) : 
            nb_useless_it += 1
        else :
            nb_useless_it = 0
        
        # Si aucun ajout depuis 2 itérations on arrête
        if nb_useless_it >= 2 :
            print(f"Plus de définitions ajoutés depuis {nb_useless_it} itérations. Arrêt.")
            break
        # Avance dans les chunks traités
        processed_chunks += nb_chunks_envoyes
    print('Glossaire créé avec succès.')
    return

import fitz  # PyMuPDF
import json
import re

def create_AAA_glossary(pdf_path, json_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    lines = full_text.splitlines()

    # Regex adaptée avec slash, virgule, tiret dans le nom de l'acronyme
    abbr_pattern = re.compile(r'^([A-Z0-9/,\-]+)\s*:\s*(.*)$')

    abbr_dict = {}
    current_abbr = None
    current_definition_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = abbr_pattern.match(line)
        if m:
            # Sauvegarder le précédent, si existant
            if current_abbr is not None:
                definition = ' '.join(current_definition_lines).strip()
                if current_abbr not in abbr_dict:
                    abbr_dict[current_abbr] = []
                if definition not in abbr_dict[current_abbr]:
                    abbr_dict[current_abbr].append(definition)

            current_abbr = m.group(1)
            current_definition_lines = [m.group(2)]

        else:
            if current_abbr is not None:
                current_definition_lines.append(line)

    # Sauvegarder le dernier
    if current_abbr is not None:
        definition = ' '.join(current_definition_lines).strip()
        if current_abbr not in abbr_dict:
            abbr_dict[current_abbr] = []
        if definition not in abbr_dict[current_abbr]:
            abbr_dict[current_abbr].append(definition)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(abbr_dict, f, ensure_ascii=False, indent=4)

    print(f"Extraction terminée, résultats sauvegardés dans {json_path}")
