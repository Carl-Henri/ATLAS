from my_paths import *

from utils import get_filter
#FBL_FILTER = get_filter(['FBL'], RAW_DATA_DIR)
FBL_FILTER = [] # pas de filtre
ABL_FILTER = get_filter(['ABL'], RAW_DATA_DIR)

MATRIX_DOC_NAMES = []
SHEET_NAME = ''

from workflows.workflow_utils import get_req_chunks, parse_upper_req, get_traceability_from_matrix
from mistral_langchain_wrapper import MistralChatWrapper
import os 
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
llm = MistralChatWrapper(api_key=API_KEY, model="medium")

def analyze_declination_based_on_traceability(SR_full_text, HLR_texts) :
    system_prompt = """You are a reasoning agent specialized in aerospace software development. 
Your task is to analyze whether a System Requirement (SR) is correctly declined or not, based on the full text of the SR and its downward requirements (which are High Level Requirements, HLR).
Provide a justification for your answer, as well as the given sources of the requirements texts.
The HLR given are ALREADY linked to the SR.
In your justification, provide the role of every HLR given in the declination of the SR.
Be sure to quote the right source : do not make any mistake on the document name and page numbers.
"""

    user_prompt = f"""Now analyse if the SR is correctly declined or not based on the provided context.

The SR full text :
{SR_full_text}

The HLR full texts (note that these are already linked to the SR):
{HLR_texts}"""


    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    result = llm.invoke(messages)
    return(result.content)

def check_if_correctly_declined(response) : 
    system_prompt = """You are given an analysis of a System Requirement (SR) of an aerospace project.
Your goal is to check whether this analysis states that the SR is correctly declined into HLRs or not. 
ONLY output True if the SR is correctly declined and False if it is not correctly declined.
"""

    user_prompt = f"""Now output True if the SR is correctly declined or False if not.
The SR declination analysis : 
{response}"""
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    correctly_declined = llm.invoke(messages).content
    print(correctly_declined)
    if correctly_declined.strip() == 'True' :
        return True
    elif correctly_declined.strip() == 'False' :
        return False 
    else :
        return True
    
from langchain.agents import create_agent
from langchain_core.tools import tool
from my_rag import retrieve_and_answer_query

@tool
def answer_question_on_ABL(query:str) -> str :
    """
    Query a LLM which is expert in the aerospace project's High Level Requirement (HLR) declared in the Applicative Baseline (ABL). 
    """    
    response, _ = retrieve_and_answer_query(query, doc_filter=ABL_FILTER, model='medium')
    return response

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
def serialize_messages(messages):
    """Sérialise les messages LangChain en str."""
    serialized_messages = []
    
    for message in messages:
        # Vérifier le type de chaque message et extraire les informations
        if isinstance(message, SystemMessage):
            serialized_message = f"System :\n{message.content}"
        elif isinstance(message, HumanMessage):
            serialized_message = f"User :\n{message.content}"
        elif isinstance(message, AIMessage):
            serialized_message = f"Assistant :\n{message.content}"
        elif isinstance(message, ToolMessage):
            serialized_message = f"Tool :\n{message.content}"
        else:
            # Si le message est d'un type inconnu, on l'ignore ou on le gère différemment
            serialized_message = f"Unknown :\n{message.content}"  

        if hasattr(message, 'tool_calls') and message.tool_calls:
            serialized_message += f"Tool call(s) : \n{message.tool_calls}"
        
        serialized_messages.append(serialized_message) 
    
    return "\n\n".join(serialized_messages)


def get_possible_HLR(SR_full_text, SR_declination_analysis) :
    system_prompt = """You are a highly adaptable reasoning agent who asks experts to reflect about the user's query.
Your goal is to gather every High Level Requirements (HLR) that might decline the System Requirement (SR) at hand, and that have not the SR name in their Upper field.
To do so, you are given the full text of the SR. You are also given an analysis of the declination of the SR based on traceability.

**Instructions:**
- You must interact with a LLM that have access to relevant informations of the aerospace project.
- Do not hesitate to ask follow-up questions to make sure ou gathered every relevant SR.
- Output "I'm done" when you are finished.

Your interaction with it will then be used by another LLM to evaluate whether the SR is correctly declined or if it is a problem of traceability. 

**When interacting with the LLM:**
    1. You MUST NOT introduce any new concepts, definitions, acronyms or assumptions that were not mentioned in the original query.
    2. Stay faithful to the original query.
    3. It is best to not define the acronyms in the queries.
    4. Do not format the questions in any way.
    5. Keep in mind that the LLMs do not have access to the history of your conversation with them.
    6. Do not hesitate to make several tool calls in parallel.
    7. If the tool response is "null", you can retry the same tool call once.
"""
    user_prompt = f"""The SR full text :
{SR_full_text}
The SR declination analysis : 
{SR_declination_analysis}"""
        
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    agent = create_agent(llm, tools=[answer_question_on_ABL])
    result = agent.invoke(
        {'messages':messages}
    )
    agent_messages = result['messages']
    return agent_messages[2:-1]

def produce_final_answer(SR_full_text, HLR_texts, SR_declination_analysis, agent_messages) :    
    system_prompt = f"""You are a reasoning agent specialized in aerospace software development.
You are given a System Requirement (SR) full text.
You are also given an analysis of the declination of the SR based on traceability.
An agent asked LLM experts of the aerospace project to gather every High Level Requirement (HLR) that might decline the SR at hand. 
Based on the agent messages, your must refine the analysis to evaluate if the SR really not correctly declined or if it is only a problem of traceability.

**Instructions: **
- Ground your answer in the requirements full texts, the SR declination analysis and on the informations gathered by the agent.
- Do NOT talk about an "LLM Expert" nor about the "agent". Rather quote directly the evidences it provides.
- Clearly distinguish between the HLR that are linked to the SR (i.e. have the SR in their Upper field) and the other HLR that could complete the traceability.
- Provide the role of every HLR given (linked to the SR) in the declination of the SR.
- In your analysis, clearly indicate which HLR you would use to complete the traceability, if applicable.
- You must only use HLR (High Level Requirements) to complete the traçability.
- Note that the HLR given are ALREADY linked to the SR.
- Output ONLY the refined SR declination analysis.
"""
    
    user_prompt = f"""
The SR full text :
{SR_full_text}

The HLR full texts (already linked to the SR): 
{HLR_texts}

The SR declination analysis: 
{SR_declination_analysis}

The agent dialog with the expert LLM: 
{agent_messages}
"""
        
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    final_answer = llm.invoke(messages).content
    
    return final_answer

from workflows.workflow_utils import add_links

from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import Term, Or

def get_abl_chunks_about_the_sr(sr_name, doc_filter=ABL_FILTER, keywords = []) :
    ix = index.open_dir(WHOOSH_DIR)        
    
    # On cherche dans le contenu des chunks directement si le nom de l'exigence apparaît
    with ix.searcher() as searcher:
        # On récupère les chunks contenant le nom de l'exigence, et qui contiennent les mots-clés correspondant à la déclaration d'une exigence
        parser = QueryParser("content", schema=ix.schema)
        query = sr_name + ' ' + ' '.join(keywords)
        q = parser.parse(query)
        filter_q = None
        if doc_filter:
            terms = [Term("doc_name", name) for name in doc_filter]
            filter_q = Or(terms)
        results_bm25 = searcher.search(q, filter=filter_q, limit=None, terms=True)
        res_bm25 = [r['content'] for r in results_bm25]
    return(res_bm25)
    
parsed_example = """
"""

def postprocess_chunk(chunk) :
    system_prompt = """You are a helpful agent specialized in aerospace software development.
Your task is to output the full text of a requirement and its source based on the context provided.
ONLY output the HLR full text exactly as found in the context and the source, without any other commentary.
Include everything that is specified (e.g. title, DFT, Upper, description, rationale, notes).
"""             
    user_prompt = f"""Here is an example of a well parsed requirement full text :
{parsed_example}
    
Now please extract the full text of the requirement based on the following context :

{chunk}
"""
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    response = llm.invoke(messages).content
    return(response)
    
def filter_abl_chunks_about_the_sr_with_upper_field(sr_name, chunks) :
    # On ne garde que les chunks de HLR qui contient la SR dans son upper
    res = []
    print(f"Nombre de chunks trouvés avant filtrage : {len(chunks)}")
    for chunk in chunks :
        upper_req = [u.replace('\n','').replace(' ','').strip().lower() for u in parse_upper_req(chunk)]
        print(f"Req : {chunk[:1000]} \n\nUpper parsé : {upper_req}")
        if sr_name.lower().replace(' ','').replace('\n','') in upper_req :
            res.append(chunk)
        else : 
            chunk = postprocess_chunk(chunk)
            upper_req = [u.replace('\n','').replace(' ','').strip().lower() for u in parse_upper_req(chunk)]
            print(f"Après postprocessing : Req : {chunk[:1000]} \n\nUpper parsé : {upper_req}")
            if sr_name.lower().replace(' ','').replace('\n','') in upper_req :
                res.append(chunk)
    print(f"Nombre de chunks après filtrage : {len(res)}")
    return(res)

def get_HLR_from_matrix(sr_name, matrix_doc_name) :
    ix = index.open_dir(WHOOSH_DIR)
    doc_filter = [matrix_doc_name]
    # On cherche dans le contenu des chunks directement si le nom de l'exigence apparaît
    with ix.searcher() as searcher:
        # On récupère les chunks contenant le nom de l'exigence, et qui contiennent les mots-clés correspondant à la déclaration d'une exigence
        parser = QueryParser("content", schema=ix.schema)
        query = sr_name
        q = parser.parse(query)
        filter_q = None
        if doc_filter:
            terms = [Term("doc_name", name) for name in doc_filter]
            filter_q = Or(terms)
        results_bm25 = searcher.search(q, filter=filter_q, limit=None, terms=True)
        res_bm25 = [r['content'] for r in results_bm25]
    res = [r for r in res_bm25 if sr_name.lower() in r]
    
    def parser_lignes_avec_mot_cle(liste_de_strings, mot_cle):
        resultats = []
        for extrait in liste_de_strings:
            lignes = extrait.split('\r\n')
            for ligne in lignes:
                if mot_cle.lower() in ligne.lower():
                    resultats.append(ligne)
        return resultats
    
    lines_with_sr_name = parser_lignes_avec_mot_cle(res, sr_name)
    for line in lines_with_sr_name :
        splits = line.split(',')
        if sr_name.lower().strip() in splits[1] :
            HLR_names_from_matrix = splits[2].split(' ')
    return(HLR_names_from_matrix)

def compare_with_matrix(sr_name, all_hlr_chunks, matrix_doc_name, sheet_name) :
    def parse_latest_req_name(chunk_lines):
        """
        Prend en entrée une liste de lignes du chunk (au moins les 3 premières).
        Retourne le nom de la req apparaissant le plus tard dans ces 3 lignes, ou None si aucune.
        """
        req_pattern = r'([A-Z0-9-]{5,}REQ[A-Z0-9-]*)'  # Exemple robuste de pattern REQ
    
        req_names = []
        for line in chunk_lines[:3]:
            matches = list(re.finditer(req_pattern, line))
            if matches:
                # Ajoute le nom et l'index de dernier match dans la ligne
                req_names.extend([(m.group(), m.start()) for m in matches])
    
        if not req_names:
            return None
        # Retourne la req au plus grand index (apparait le plus tard)
        latest = max(req_names, key=lambda t: t[1])
        return latest[0]
        
    HLR_names_from_matrix = get_traceability_from_matrix(sr_name, matrix_doc_name = matrix_doc_name, sheet_name=sheet_name)
    HLR_names = []
    for chunk in all_hlr_chunks :
        resultat = parse_latest_req_name(chunk.split('\n'))
        HLR_names.append(resultat.lower())
    print(f'HLR trouvées dans les docs : {HLR_names}')
    print(f'HLR trouvées dans la matrice : {HLR_names_from_matrix}')
    diff1 = set(HLR_names) - set(HLR_names_from_matrix)
    diff2 = set(HLR_names_from_matrix) - set(HLR_names)
    HLR_names.sort()
    HLR_names_from_matrix.sort()
    return({'hlr_trouvées_dans_les_docs':HLR_names,'hlr_trouvées_dans_la_matrice':HLR_names_from_matrix,'hlr_trouvées_en_plus_dans_les_docs':diff1 ,'hlr_trouvées_en_plus_dans_la_matrice':diff2})

def workflow_declination_analysis(sr_name) :
    # Étape 1 : recherche du texte de la SR dans la FBL
    yield False, "Recherche du texte de la SR", "En cours d'exécution..."
    chunks = get_req_chunks(FBL_FILTER, sr_name, keywords = [])
    SR_full_text = chunks[0]
    yield False, "Recherche du texte de la SR", SR_full_text

    # Étape 2 : recherche dans l'ABL des HLRs contenant la SR dans leur champ Upper
    yield False, f"Recherche des textes des HLR déclinant la SR", "En cours d'exécution..."
    all_hlr_chunks = []
    for i in range(len(sr_name)) :
        modified_sr_name = sr_name[0:i]+' '+sr_name[i:] 
        hlr_chunks = get_abl_chunks_about_the_sr(modified_sr_name, keywords = [])
        all_hlr_chunks.extend(hlr_chunks)
    all_hlr_chunks = list(set(all_hlr_chunks))
    all_hlr_chunks = filter_abl_chunks_about_the_sr_with_upper_field(sr_name, all_hlr_chunks)
    HLR_texts = "\n\n".join(all_hlr_chunks)
    yield False, f"Recherche des textes des HLR déclinant la SR", HLR_texts
    init_HLR_texts = HLR_texts

    # Étape 3 : comparaison avec la matrice de traçabilité
    res = ""
    if MATRIX_DOC_NAMES :
        yield False, "Vérification de la cohérence avec la matrice de traçabilité", "En cours d'exécution..."
        comparison = compare_with_matrix(sr_name, all_hlr_chunks, MATRIX_DOC_NAMES, SHEET_NAME)
        hlr_trouvées_en_plus_dans_les_docs = comparison['hlr_trouvées_en_plus_dans_les_docs']
        hlr_trouvées_en_plus_dans_la_matrice = comparison['hlr_trouvées_en_plus_dans_la_matrice']
        HLR_names = comparison['hlr_trouvées_dans_les_docs']
        HLR_names_from_matrix = comparison['hlr_trouvées_dans_la_matrice']
        comparison_message = f'HLR trouvées dans les docs : {" ".join(HLR_names)}\nHLR trouvées dans la matrice de traçabilité : {" ".join(HLR_names_from_matrix)}'

        if len(hlr_trouvées_en_plus_dans_les_docs) == 1 : 
            comparison_message += f'\n\nHLR liée à la SR trouvée en plus dans les documents : {" ".join(hlr_trouvées_en_plus_dans_les_docs)}.\nVérifiez la matrice de traçabilité.'
        elif len(hlr_trouvées_en_plus_dans_les_docs) > 1 :
            comparison_message += f'\n\nHLR liées à la SR trouvées en plus dans les documents : {" ".join(hlr_trouvées_en_plus_dans_les_docs)}.\nVérifiez la matrice de traçabilité.'

        if hlr_trouvées_en_plus_dans_la_matrice :
            hlr_added = ''
            for hlr in hlr_trouvées_en_plus_dans_la_matrice :
                hlr_chunks = get_req_chunks(ABL_FILTER, hlr, keywords = [])
                HLR_full_text = hlr_chunks[0]
                all_hlr_chunks.append(HLR_full_text)
                hlr_added += '\n\n' + HLR_full_text
            HLR_texts = "\n\n".join(all_hlr_chunks)
            if len(hlr_trouvées_en_plus_dans_la_matrice) == 1 : 
                comparison_message += f'\n\nHLR non trouvée dans les documents mais renseignée comme déclinant la SR dans la matrice de traçabilité : {" ".join(hlr_trouvées_en_plus_dans_la_matrice)}.\nVérifiez le champ upper.'
            elif len(hlr_trouvées_en_plus_dans_la_matrice) > 1 :
                comparison_message += f'\n\nHLR non trouvées dans les documents mais renseignées comme déclinant la SR dans la matrice de traçabilité : {" ".join(hlr_trouvées_en_plus_dans_la_matrice)}.\nVérifiez le champ upper.'
        
        if not(hlr_trouvées_en_plus_dans_la_matrice) and not(hlr_trouvées_en_plus_dans_les_docs):
            comparison_message += '\n\nHLR trouvées dans les documents cohérentes avec la matrice de traçabilité.'

        res += f"""## Comparison between documents and traceability matrix
{comparison_message}\n"""
        
        if hlr_added :
            yield False, "Vérification de la cohérence avec la matrice de traçabilité", comparison_message  + f'\nTextes des HLR ajoutés :\n{hlr_added}'
        else :
            yield False, "Vérification de la cohérence avec la matrice de traçabilité", comparison_message

    # Étape 4 : première analyse avec ces éléments de traçabilité
    yield False, "Analyse de la déclinaison basée sur la traçabilité", "En cours d'exécution..."
    response = analyze_declination_based_on_traceability(SR_full_text, HLR_texts)    
    yield False, "Analyse de la déclinaison basée sur la traçabilité", response

    # Étape 5 : on regarde si la SR a été déclarée comme bien déclinée par le LLM
    yield False, "Correctement déclinée ?", "En cours d'exécution..."
    correctly_declined = check_if_correctly_declined(response)
    yield False, "Correctement déclinée ?", correctly_declined
    
    # Étape 6 : si non, on recherche les potentielles HLR décliant la SR
    if not(correctly_declined) : 
        yield False, "Recherche des potentielles HLR déclinant la SR", "En cours d'exécution..."
        agent_messages = get_possible_HLR(SR_full_text, response)
        yield False, "Recherche des potentielles HLR déclinant la SR", serialize_messages(agent_messages)

        # Étape 7 : analyse finale avec les nouveaux éléments trouvés
        yield False, "Analyse finale", "En cours d'exécution..."
        final_response = produce_final_answer(SR_full_text, HLR_texts, response, agent_messages)
        yield False, "Analyse finale", final_response
        response = add_links(response)
        final_response = add_links(final_response)
        if MATRIX_DOC_NAMES :
            str_log = f"""
## SR full text
{SR_full_text}

## HLR texts 
{init_HLR_texts}

## Comparison with traceability matrix
{comparison_message}
{hlr_added}

## SR declination analysis based on traceability

## Correctly declined ? 
{correctly_declined}

## Agent messages
{serialize_messages(agent_messages)}
"""
        else :
            str_log = f"""
## SR full text
{SR_full_text}

## HLR texts 
{init_HLR_texts}

## SR declination analysis based on traceability

## Correctly declined ? 
{correctly_declined}

## Agent messages
{serialize_messages(agent_messages)}
"""
        res += f"""## SR declination analysis based on traceability
{response}

## Refined SR declination analysis 
{final_response}
"""
        yield True, res, str_log

    else :
        response = add_links(response)
        if MATRIX_DOC_NAMES : 
            str_log = f"""
## SR full text
{SR_full_text}

## HLR texts 
{init_HLR_texts}

## Comparison with traceability matrix
{comparison_message}
{hlr_added}

## Correctly declined ? 
{correctly_declined}
"""
        else :
            str_log = f"""
## SR full text
{SR_full_text}

## HLR texts 
{init_HLR_texts}

## Correctly declined ? 
{correctly_declined}
"""
        res += f"""## SR declination analysis based on traceability
{response}
""" 
        yield True, res, str_log 