
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import Term, Or

from mistral_langchain_wrapper import MistralChatWrapper
import os 
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
llm = MistralChatWrapper(api_key=API_KEY, model="medium")

from my_paths import *
parsed_example = """
"""

def get_req_chunks(doc_filter, req_name, keywords = ['Title']) :
    doc_filter = doc_filter

    ix = index.open_dir(WHOOSH_DIR)

    # On essaie d'abord de chercher avec le champ hiérarchie qui contient souvent le nom des exigences
    # Il faut aussi vérifier que le contenu du chunk trouvé est bien celui d'une exigence (avec les keywords)
    with ix.searcher() as searcher:
        parser = QueryParser("hierarchy", schema=ix.schema)
        q = parser.parse(req_name)
        filter_q = None
        if doc_filter:
            terms = [Term("doc_name", name) for name in doc_filter]
            filter_q = Or(terms)
        results_bm25 = searcher.search(q, filter=filter_q, limit=None, terms=True)
        res_bm25 = [r['content'] for r in results_bm25]
        # Si les chunks trouvés n'ont pas les keywords attendus, ils ne sont pas considérés comme étant un chunk d'une exigence
        # Problème avec ça : si l'exigence s'étale sur plusieurs chunks c'est mort... après ça a peu de chance d'arriver mais bon 
        # Solution : si il y a plus d'un chunk trouvé on passe au LLM pour faire le tri
        # Si un seul est trouvé on vérifie que c'est bien celui de l'exigence avec les mots-clés et le nom de l'exigence
        if len(res_bm25) == 1 :
            keywords.append(req_name.lower())
            for keyword in keywords :
                if not(keyword in res_bm25[0].lower()) :
                    res_bm25.pop()
                    break
    
    if not(res_bm25) :
        # Sinon on cherche dans le contenu des chunks directement si le nom de l'exigence apparaît, et on fait un appel LLM pour restituer le texte de l'exigence
        with ix.searcher() as searcher:
            # On récupère les chunks contenant le nom de l'exigence, mais aussi title 
            parser = QueryParser("content", schema=ix.schema)
            query = req_name + ' ' + ' '.join(keywords)
            q = parser.parse(query)
            filter_q = None
            if doc_filter:
                terms = [Term("doc_name", name) for name in doc_filter]
                filter_q = Or(terms)
            results_bm25 = searcher.search(q, filter=filter_q, limit=None, terms=True)
            res_bm25 = [r['content'] for r in results_bm25]
        
    if len(res_bm25) > 1 : # Si plus d'un chunk on fait le tri / recolle les morceaux            
        system_prompt = """You are a helpful agent specialized in aerospace software development.
Your task is to output the full text of a requirement and its source based on the context provided.
ONLY output the HLR full text exactly as found in the context and the source, without any other commentary.
Include everything that is specified (e.g. title, DFT, Upper, description, rationale, notes).
"""             
        context = "\n\n".join(res_bm25)
        user_prompt = f"""Here is an example of a well parsed HLR full text :
{parsed_example}
        
Now please extract the full text of the requirement {req_name} based on the following context :
{context}
"""
        messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
        response = llm.invoke(messages).content
        res_bm25 = [response]
    
    return(res_bm25)

def parse_upper_req(
    text, 
    keywords=['Upper', 'Upward Requirements', 'Upward Requirement']
):
    import re

    match = None
    i = 0

    # Pattern logic:
    #   - Start at a line beginning with one of the keywords + colon
    #   - Capture everything lazily up to:
    #       * the next line that clearly starts a new "field" (e.g. "Title :", "DFT :", "Status :"...)
    #       * OR 'End Requirement'
    #       * OR end of the text
    #
    # The "new field" line is heuristically defined as:
    #   line starting with optional spaces, then a word (letters/digits/underscores, possibly with dashes),
    #   then optional spaces, then a colon.
    #
    while not match and i < len(keywords):
        pattern = (
            rf"^[\s\*]*{re.escape(keywords[i])}[\s\*]*:"   # "Upward Requirement:" line
            r"[\s\*]*"                                    # optional spaces
            r"(.*?)"                                      # capture block lazily
            r"(?="                                        # lookahead for block terminator
                r"\n[\s\*]*[A-Za-z0-9_][A-Za-z0-9_\- ]*[\s\*]*:"  # next 'Field :' line
                r"|\n*End Requirement[\s\*]*"                     # or "End Requirement"
            r")"
        )
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        i += 1

    if not match:
        return []

    block = match.group(1)

    # Recoller toute coupure sur saut de ligne, style xxx- \n zzz
    # (You can keep / adapt your existing line-merge rules)
    block = re.sub(r'(\w+-)\s*\n\s*([A-Z]+\-\d+)', r'\1\2', block)
    block = re.sub(r'(\w+\.)\s*\n\s*([A-Z][a-z]+\-\d+)', r'\1\2', block)

    # Retirer les autres retours à la ligne, normaliser espaces
    block = block.replace('\n', ' ').replace('\r', '')
    block = re.sub(r'\s+', ' ', block)

    # Split sur point-virgule (avec/sans espace)
    reqs = [r.strip() for r in re.split(r'\s*;\s*', block) if r.strip()]

    return reqs

from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import Term, Or

def get_traceability_from_matrix(req_name, matrix_doc_name, sheet_name) :
    ix = index.open_dir(WHOOSH_DIR)
    doc_filter = matrix_doc_name
    # On cherche dans le contenu des chunks directement si le nom de l'exigence apparaît
    with ix.searcher() as searcher:
        # On récupère les chunks contenant le nom de l'exigence
        parser = QueryParser("content", schema=ix.schema)
        query = req_name
        q = parser.parse(query)
        filter_q = None
        if doc_filter:
            terms = [Term("doc_name", name) for name in doc_filter]
            filter_q = Or(terms)
        results_bm25 = searcher.search(q, filter=filter_q, limit=None, terms=True)
        res_bm25 = [r['content'] for r in results_bm25]
    res = [r for r in res_bm25 if req_name.lower() in r]

    def parser_lignes_avec_nom_req(liste_de_strings, nom_req, sheet_name):
        resultats = []
        for extrait in liste_de_strings:
            if sheet_name.lower() in extrait.lower():
                lignes = extrait.split('\r\n')
                for ligne in lignes:
                    if (nom_req.lower() in ligne.lower()):
                        resultats.append(ligne)
        return resultats

    lines_with_req_name = parser_lignes_avec_nom_req(res, req_name, sheet_name)
    for line in lines_with_req_name :
        splits = line.split(',')
        if req_name.lower().strip() in splits[1] :
            linked_req_names_from_matrix = splits[2].split(' ') 
    return(linked_req_names_from_matrix)

import re
from agentic_workflow_utils import rendre_citation_cliquable

def add_links(final_answer) :
    system_prompt = f"""Your task is to output a list of the sources referenced in the following response.  
**INSTRUCTIONS:**  
- DO NOT invent any references or page numbers.  
- DO NOT add any code tags.  
- If no document is referenced, you MUST output [].  
- Respect the order in which the references appear in the response.  
- ONLY OUTPUT a Python list formatted exactly like this: [(<string: document name>, <int: page number>), ...]  
- If the referenced document is a .html or an Excel file (.xls or .xlsx), you MUST add (<string: document name>, 1) to the list.  

For example:  
[("doc1", 10), ("doc2", 5), ("doc3.xls", 1), ("doc4.html", 1), ...]  
""" 
    messages = [{"role":"system", "content": system_prompt}]
    prompt = f"""
    Now get the list of sources referenced in the following response : \n{final_answer}
"""
    messages.append({"role":"user", "content": prompt})
    answer = llm.invoke(messages).content

    s = answer
    start = s.find('[')
    end = s.rfind(']')+1
    list_str = s[start:end]

    # Regex et extraction initiale
    pattern = re.compile(r'\(([^,]+),\s*(\d+)\)')
    matches = pattern.findall(list_str)

    ref_list = []
    for text, num in matches:
        text = text.strip().strip('"').strip("'")  # nettoyage guillemets et espaces
        ref_list.append((text, int(num)))

    # Suppression doublons
    ref_list = list(set(ref_list))

    # Regroupement images par document
    doc_names = list(set([doc_name for doc_name, _ in ref_list]))
    final_answer = rendre_citation_cliquable(final_answer, doc_names)
    return(final_answer)