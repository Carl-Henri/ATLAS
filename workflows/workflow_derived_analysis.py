from my_paths import *

from utils import get_filter
# FBL_FILTER = get_filter(['FBL'], RAW_DATA_DIR)
FBL_FILTER = [] # Pas de filtre
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

def analyze_derivation_based_on_traceability(HLR_full_text, str_upper_req_texts) :
    system_prompt = """You are a reasoning agent specialized in aerospace software development. 
Your task is to analyze whether a High Level Requirement (HLR) is derived or not, based on the full text of the HLR and its upper requirements.
Provide a justification for your answer, as well as the given sources of the requirements texts.
In your justification, you MUST provide the aspects of the SR declined in the HLR for every SR, so that the user understands the derivation status of the HLR.
Be sure to quote the right source : do not make any mistake on the document name and page numbers.

**Definition of a Derived Requirement:**

A derived requirement is one that introduces features not specified in any of the upper-level requirements within the parent specification document.  
A requirement is considered *totally derived* if it *solely* specifies features entirely absent from the parent specification.  
It is considered *partially derived* if it is related to the parent specification but also introduces features not present in the upper-level requirements.

**Additional note about requirement declaration:**  
If the keyword "DERIVED" appears in the "Upper" field of a requirement, it indicates that the requirement is considered derived within the project.
If only "DERIVED" is present in the "Upper" field, the requirement is classified as fully derived.
If the "Upper" field contains both "DERIVED" and other requirements, the requirement is classified as partially derived.
However, do not use this declaration as your basis for the analysis.
Your objective is to verify whether this classification is accurate based on your reasoning.
"""

    user_prompt = f"""Now analyse if the HLR is partially derived, totally derived or not derived based on the provided context.

The HLR full text :
{HLR_full_text}

The upper requirements full texts:
{str_upper_req_texts}"""


    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    result = llm.invoke(messages)
    return(result.content)

def check_if_declared_derived(response) : 
    system_prompt = """You are given an analysis of a High Level Requirement (HLR) of an aerospace project.
Your goal is to check whether this analysis states that the HLR is derived or not. 
If the HLR is declared as totally derived or partially derived, you MUST output True.
If the HLR is declared are not derived, you MUST output False.

ONLY output True if the HLR is derived and False if it is not.
    """

    user_prompt = f"""Now output True if the HLR is declared derived or False if not.
The HLR derivation analysis : 
{response}"""
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    derived = llm.invoke(messages).content
    if derived.strip() == 'True' :
        return True
    elif derived.strip() == 'False' :
        return False 
    
from langchain.agents import create_agent
from langchain_core.tools import tool
from my_rag import retrieve_and_answer_query

@tool
def answer_question_on_FBL(query:str) -> str :
    """
    Query a LLM which is expert in the aerospace project's System Requirements (SR) declared in the Functional Baseline (FBL). 
    The query should be formulated in english.
    """    
    response, _ = retrieve_and_answer_query(query, doc_filter=FBL_FILTER, model='medium')
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


def get_possible_declined_SR(HLR_full_text, HLR_derived_analysis) :
    system_prompt = """You are a highly adaptable reasoning agent who asks experts to reflect about the user's query.
Your goal is to gather every System Requirement (SR) that might be declined in the High Level Requirement (HLR) at hand, and that are not traced in the HLR.
To do so, you are given the full text of the HLR. If the HLR have upper SR declared in its Upper field, you are also given an analysis of the derivation of a HLR based on traceability.

**Instructions:**
- You must interact with a LLM that have access to relevant informations of the aerospace project.
- Ask it about the parts of the HLR that are analysed as "derived" in the analysis i.e. not specified in the Upper requirements of the HLR.
- Do not hesitate to ask follow-up questions to make sure ou gathered every relevant SR.
- Output "I'm done" when you are finished.

Your interaction with it will then be used by another LLM to evaluate whether the HLR is really derived or if it is a problem of traceability. 

**Definition of a Derived Requirement:**

    A derived requirement is one that introduces features not specified in any of the upper-level requirements within the parent specification document.  
    A requirement is considered *totally derived* if it *solely* specifies features entirely absent from the parent specification.  
    It is considered *partially derived* if it is related to the parent specification but also introduces features not present in the upper-level requirements.

**When interacting with the LLM:**
    1. You MUST NOT introduce any new concepts, definitions, acronyms or assumptions that were not mentioned in the original query.
    2. Stay faithful to the original query.
    3. It is best to not define the acronyms in the queries.
    4. Do not format the questions in any way.
    5. Keep in mind that the LLMs do not have access to the history of your conversation with them.
    6. Do not hesitate to make several tool calls in parallel.
    7. If the tool response is "null", you can retry the same tool call once.
"""
    if HLR_derived_analysis :
        user_prompt = f"""The HLR full text :
{HLR_full_text}
The HLR derivation analysis : 
{HLR_derived_analysis}"""
    else :
        user_prompt = f"""The HLR full text :
{HLR_full_text}"""
        
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    agent = create_agent(llm, tools=[answer_question_on_FBL])
    result = agent.invoke(
        {'messages':messages}
    )
    agent_messages = result['messages']
    return agent_messages[2:-1]

def produce_final_answer(HLR_full_text, HLR_upper_texts, HLR_derivation_analysis, agent_messages) :    
    system_prompt = f"""You are a reasoning agent specialized in aerospace software development.
You are given a High Level Requirement (HLR) full text.
If the HLR have upper SR declared in its Upper field, you are also given an analysis of the derivation of a HLR based on traceability.
An agent asked LLM experts of the aerospace project to gather every System Requirement (SR) that might be declined in the HLR at hand. 
Based on the agent messages, your must refine the analysis to evaluate if the HLR is really derived or if it is only a problem of traceability.

**Instructions: **
- Ground your answer in the requirement full texts, the HLR derivation analysis if provided and on the informations gathered by the agent.
- Do NOT talk about an "LLM Expert" nor about the "agent". Rather quote directly the evidences it provides.
- Clearly distinguish between the SR that are in the Upper field of the HLR full text and the other SR that could complete the traceability.
- Provide the aspects of the SR declined in the HLR for every SR.
- In your analysis, clearly indicate which SR you would use to complete the traceability, if applicable.
- Only use SR (System Requirements) to complete the traçability, if applicable.
- Output ONLY the refined HLR derivation analysis.

**Definition of a Derived Requirement: **
    A derived requirement is one that introduces features not specified in any of the upper-level requirements within the parent specification document.  
    A requirement is considered *totally derived* if it solely specifies features entirely absent from the parent specification.  
    It is considered *partially derived* if it is related to the parent specification but also introduces features not present in the upper-level requirements.

**Additional note about requirement declaration:**  
    If the keyword "DERIVED" appears in the "Upper" field of a requirement, it indicates that the requirement is considered derived within the project.
    If only "DERIVED" is present in the "Upper" field, the requirement is classified as fully derived.
    If the "Upper" field contains both "DERIVED" and other requirements, the requirement is classified as partially derived.
    However, do not use this declaration as your basis for the analysis.
    Your objective is to verify whether this classification is accurate based on your reasoning.
"""
    if HLR_derivation_analysis and HLR_upper_texts :
        user_prompt = f"""
The HLR full text :
{HLR_full_text}

The HLR Upper requirements full texts : 
{HLR_upper_texts}

The HLR_derivation_analysis: 
{HLR_derivation_analysis}

The agent dialog with the expert LLM: 
{agent_messages}
"""
    else :
        user_prompt = f"""
The HLR full text :
{HLR_full_text}

The agent dialog with the expert LLM: 
{agent_messages}
"""
        
    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    final_answer = llm.invoke(messages).content
    
    return final_answer

from workflows.workflow_utils import add_links

def compare_with_matrix(HLR_name, upper_req_list, matrix_doc_name, sheet_name) :        
    linked_req_names_from_matrix = get_traceability_from_matrix(HLR_name, matrix_doc_name = matrix_doc_name, sheet_name=sheet_name)
    upper_req_list = [upper.lower() for upper in upper_req_list]
    upper_req_list.sort()
    linked_req_names_from_matrix = [upper.lower() for upper in linked_req_names_from_matrix]
    linked_req_names_from_matrix.sort()
    print(f'Upper trouvées dans les docs : {upper_req_list}')
    print(f'Upper trouvées dans la matrice : {linked_req_names_from_matrix}')
    diff1 = set(upper_req_list) - set(linked_req_names_from_matrix)
    diff2 = set(linked_req_names_from_matrix) - set(upper_req_list)
    return({'upper_trouvées_dans_les_docs':upper_req_list,'upper_trouvées_dans_la_matrice':linked_req_names_from_matrix,'upper_trouvées_en_plus_dans_les_docs':diff1 ,'upper_trouvées_en_plus_dans_la_matrice':diff2})

def workflow_derived_analysis(HLR_name) :
    yield False, "Recherche du texte de la HLR", "En cours d'exécution..."
    chunks = get_req_chunks(ABL_FILTER, HLR_name, keywords = ['DFT','Title','Upper'])
    if not(chunks) :
        chunks = get_req_chunks(ABL_FILTER, HLR_name, keywords = [])        
    HLR_full_text = chunks[0]
    yield False, "Recherche du texte de la HLR", HLR_full_text

    upper_req_list = parse_upper_req(HLR_full_text)
    res = ""
    if MATRIX_DOC_NAMES :
        yield False, "Vérification de la cohérence avec la matrice de traçabilité", "En cours d'exécution..."
        comparison = compare_with_matrix(HLR_name, upper_req_list, MATRIX_DOC_NAMES, SHEET_NAME)
        upper_trouvées_en_plus_dans_les_docs = comparison['upper_trouvées_en_plus_dans_les_docs']
        upper_trouvées_en_plus_dans_la_matrice = comparison['upper_trouvées_en_plus_dans_la_matrice']
        upper_names = comparison['upper_trouvées_dans_les_docs']
        upper_names_from_matrix = comparison['upper_trouvées_dans_la_matrice']
        comparison_message = f'Upper trouvées dans les docs : {" ".join(upper_names)}\nUpper trouvées dans la matrice de traçabilité : {" ".join(upper_names_from_matrix)}'

        if len(upper_trouvées_en_plus_dans_les_docs) == 1 : 
            comparison_message += f'\n\nUpper trouvée en plus dans les documents : {" ".join(upper_trouvées_en_plus_dans_les_docs)}.\nVérifiez la matrice de traçabilité.'
        elif len(upper_trouvées_en_plus_dans_les_docs) > 1 :
            comparison_message += f'\n\nUpper trouvées en plus dans les documents : {" ".join(upper_trouvées_en_plus_dans_les_docs)}.\nVérifiez la matrice de traçabilité.'

        if len(upper_trouvées_en_plus_dans_la_matrice) == 1 : 
            comparison_message += f'\n\nUpper non trouvée dans les documents mais renseignée dans la matrice de traçabilité : {" ".join(upper_trouvées_en_plus_dans_la_matrice)}.\nVérifiez le champ upper.'
        elif len(upper_trouvées_en_plus_dans_la_matrice) > 1 :
            comparison_message += f'\n\nUpper non trouvées dans les documents mais renseignées dans la matrice de traçabilité : {" ".join(upper_trouvées_en_plus_dans_la_matrice)}.\nVérifiez le champ upper.'
        
        if not(upper_trouvées_en_plus_dans_la_matrice) and not(upper_trouvées_en_plus_dans_les_docs):
            comparison_message += '\n\nUpper trouvées dans les documents cohérentes avec la matrice de traçabilité.'
        

        upper_req_list = list(set(upper_names + upper_names_from_matrix))
        res += f"""## Comparison between documents and traceability matrix
{comparison_message}\n"""
        yield False, "Vérification de la cohérence avec la matrice de traçabilité", comparison_message

    response = ''
    str_upper_req_texts = ''
    
    upper_req_texts = {}
    for upper in upper_req_list :
        if upper != 'DERIVED' : 
            yield False, f"Recherche du texte de {upper}", "En cours d'exécution..."
            chunks = get_req_chunks(FBL_FILTER, upper, keywords = ['Title'])
            upper_req_texts[upper] = chunks
            yield False, f"Recherche du texte de {upper}", "\n\n".join(chunks)

    if upper_req_texts : 
        str_upper_req_texts = "\n\n".join([text for _, texts in upper_req_texts.items() for text in texts])

        yield False, "Analyse de la dérivation basée sur la traçabilité", "En cours d'exécution..."
        response = analyze_derivation_based_on_traceability(HLR_full_text, str_upper_req_texts)
        yield False, "Analyse de la dérivation basée sur la traçabilité", response

        yield False, "Dérivée ?", "En cours d'exécution..."
        derived = check_if_declared_derived(response)
        yield False, "Dérivée ?", derived
    else :
        derived = True
    
    if derived : 
        yield False, "Recherche des SR possiblement déclinées dans la HLR", "En cours d'exécution..."
        agent_messages = get_possible_declined_SR(HLR_full_text, response)
        yield False, "Recherche des SR possiblement déclinées dans la HLR", serialize_messages(agent_messages)

        yield False, "Analyse finale", "En cours d'exécution..."
        final_response = produce_final_answer(HLR_full_text, str_upper_req_texts, response, agent_messages)
        yield False, "Analyse finale", final_response
        response = add_links(response)
        final_response = add_links(final_response)

        if response :
            str_log = f"""
## HLR full text
{HLR_full_text}

## Upper REQ texts 
{str_upper_req_texts}

## HLR derivation analysis based on traceability

## Derived ? 
{derived}

## Agent messages
{serialize_messages(agent_messages)}
"""
            res += f"""## HLR derivation analysis based on traceability
{response}

## Refined HLR derivation analysis 
{final_response}
"""
            yield True, res, str_log

        else : 
            str_log = f"""
## HLR full text
{HLR_full_text}

## Derived ?
{derived}

## Agent messages
{serialize_messages(agent_messages)}
"""
            res += f"""## HLR derivation analysis 
{final_response}
"""
            yield True, res, str_log
    else :
        response = add_links(response)
        str_log = f"""
## HLR full text
{HLR_full_text}

## Upper REQ texts 
{str_upper_req_texts}

## Derived ? 
{derived}
"""
        res += f"""## HLR derivation analysis based on traceability
{response}
"""
        yield True, res, str_log 