
import os
import ast
import re

from agentic_workflow_utils import log_step, serialize_messages, collect_statistics, collect_statistics_from_retrieved_pages, find_document_path, get_page_image, match_doc_names, rendre_citation_cliquable
from utils import get_filter
from manage_glossary import query_glossary
from my_paths import *

from collections import defaultdict
from langchain_core.tools import tool

from langchain.agents import create_agent
from my_rag import retrieve_and_answer_query
from my_agentic_rag import agentic_retrieve_and_answer_query

from pathlib import Path
from openai_llm import instantiate_llm
from mistral_langchain_wrapper import MistralChatWrapper
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
llm_mistral_medium = MistralChatWrapper(api_key=API_KEY, model="medium")
llm_mistral_small = MistralChatWrapper(api_key=API_KEY, model="small")
ll_mistral_large = MistralChatWrapper(api_key=API_KEY, model="large")

llm_gpt = instantiate_llm('gpt-4.1')

llm_mini = llm_mistral_small
llm = llm_mistral_medium
llm_large = ll_mistral_large

import threading

# Créer un lock global
model_lock = threading.Lock()

# Initialisation des variables globales
DOC_CHOICES = []
ANALYSE_MODE = 'fast'
TOOLS = []

@tool
def get_doc_names_with_semantic_search(query:str, top_k:int) -> list[str] : 
    """Search for the real names of the documents in the database by performing a semantic search.
    Inputs: 
        - query, a string which is used to perform the semantic search. 
        - top_k, an int which is the number of documents names you want to retrieve.
    Ouput :
        - a list of top_k document names
    """
    doc_names = match_doc_names(query, RAW_DATA_DIR_B, top_k=top_k)
    return(doc_names)

def extract_doc_names_from_query(query) :
    llm_with_tools = llm.bind_tools([get_doc_names_with_semantic_search])
    prompt_to_get_real_document_names = f"""
        Your goal is to retrieve all the real document identifiers that are referenced in the user's query by performing a semantic search.
        Instructions : 
            - Your query should be concise and contain ONLY the document name and versions sought.
            - Keep in mind that each document may have around 15 different versions.
            - You must include in your query document names as acronyms only, exclude the definition of the acronym
            - You must include in your query the document versions sought
            - You must do one tool call per document name, to maximize the chance of retrieving the correct document identifiers.
        The user's query : {query}
    """
    tool_calls = llm_with_tools.invoke(prompt_to_get_real_document_names).tool_calls
    reference_doc_list = []
    for tool_call in tool_calls :
        reference_doc_list.extend(get_doc_names_with_semantic_search.invoke(tool_call['args']))
    
    prompt_to_extract_document_names = f"""
    Your task is to analyze the user's query and to extract document names and versions from it.
    INSTRUCTIONS : 
        - Only output a Python list[str], a list of string. 
        - Each string MUST be one of those provided in the reference list of documents ids.
        - If there is no document name nor version in the query, you MUST output []. 
        - DO NOT invent any document name or any version.
    The user's query : {query}
    The reference document ids list : {reference_doc_list}
    """
    answer = llm.invoke(prompt_to_extract_document_names).content
    s = answer
    start = s.find('[')
    end = s.rfind(']')+1
    list_str = s[start:end]
    doc_list = ast.literal_eval(list_str)
    doc_list = list(set(doc_list))
    final_doc_list = [Path(find_document_path(doc_name, [RAW_DATA_DIR_B])).name.replace('.pdf','') for doc_name in doc_list]
    return(final_doc_list)

if DATA_PATH_B :
    @tool
    def answer_question_on_multiple_document_versions(query:str) -> str :
        """
        Query a LLM which has access to several versions of each document and is expert at answering questions on several document versions.
        The query MUST include the names and versions of the documents to compare. 
        """
        # 1) On récupère à partir de la requête les noms des versions des documents
        doc_list = extract_doc_names_from_query(query)
        doc_list_context = f"The user's query is about the following documents : {', '.join(doc_list)}\n"
        print(f"On filtre sur les documents : {doc_list}")
        # 2) On fait appel au LLM-RAG en filtrant les bons documents, et avec la database contenant toutes les versions des documents
        print(f"Mode d'analyse : {ANALYSE_MODE}")
        if ANALYSE_MODE == 'fast' :
            response, context = retrieve_and_answer_query(doc_list_context+query, chroma_dir=CHROMA_DIR_B, whoosh_dir=WHOOSH_DIR_B, data_dir=PARSED_DATA_DIR_B, doc_filter=doc_list)
        elif ANALYSE_MODE == 'complex' :
            response, context = agentic_retrieve_and_answer_query(doc_list_context+query, chroma_dir=CHROMA_DIR_B, whoosh_dir=WHOOSH_DIR_B, data_dir=PARSED_DATA_DIR_B, doc_filter=doc_list)
        elif ANALYSE_MODE == 'intermediate' :
            response, context = agentic_retrieve_and_answer_query(doc_list_context+query, chroma_dir=CHROMA_DIR_B, whoosh_dir=WHOOSH_DIR_B, data_dir=PARSED_DATA_DIR_B, doc_filter=doc_list, min_total_tokens_to_send=50000)
        global statistics
        statistics.append(collect_statistics(context))
        return(response)
    TOOLS.append(answer_question_on_multiple_document_versions)

if PAGES_EMBEDDINGS_PATH and PAGES_EMBEDDINGS_PATH.is_file() :
    from visual_rag_on_sdd import retrieve_SDD_pages_and_answer_query
    @tool
    def answer_question_on_calculations(query:str) -> str :
        """
        Query a LLM which has access to the SDD (Software Design Description) documents, which describe how the computations are made in the aerospace project.
        The query should be about one calculus or algorithm or concept at a time, as the LLM will base its response on a few pages only.
        The query should include the reference of the computation if known, that mean the name of the LLR (Low Level Requirement) and the document which details the calculus.
        """
        with model_lock :
            response, retrieved_pages = retrieve_SDD_pages_and_answer_query(query, data_dir=RAW_DATA_DIR)
        global statistics
        statistics.append(collect_statistics_from_retrieved_pages(retrieved_pages))
        return(response)
    TOOLS.append(answer_question_on_calculations)

@tool
def answer_question_based_on_db(query:str) -> str :
    """
    Query a LLM which is expert in the current aeronautic product development processes, industrial norms, and engineering requirements used in the aerospace project. 
    """
    print(f"Mode d'analyse : {ANALYSE_MODE}")
    from my_paths import RAW_DATA_DIR
    filter = get_filter(DOC_CHOICES, RAW_DATA_DIR)
    print(f"On filtre sur les documents : {filter}")
    if ANALYSE_MODE == 'fast' :
        response, context = retrieve_and_answer_query(query, doc_filter=filter, model='medium')
    elif ANALYSE_MODE == 'complex' :
        response, context = agentic_retrieve_and_answer_query(query, doc_filter=filter)
    elif ANALYSE_MODE == 'intermediate' :
        response, context = agentic_retrieve_and_answer_query(query, doc_filter=filter, min_total_tokens_to_send=50000)
    global statistics
    statistics.append(collect_statistics(context))
    return response
TOOLS.append(answer_question_based_on_db)

@tool
def get_possible_acronym_definitions(acronym_list: list[str]) -> dict :
    """
    Get the list of possible definitions for the acronyms in acronym_list. 
    """
    possible_meanings = {}
    for acronym in acronym_list :
        res = query_glossary(acronym)
        possible_meanings[acronym] = res
    return(possible_meanings)

def run_agent(query, system_prompt, history, llm) :
    agent = create_agent(model=llm, tools=TOOLS)
    messages = [
        {
            "role":"system", "content":system_prompt
        }
    ]
    if history : 
        messages += history
    messages.append(
        {
            "role":"user", "content":query
        }
    )
    result = agent.invoke(
        {'messages':messages}
    )

    return result['messages'][-1].content, result['messages']


# Step 1 — Reformulate query
def reformulate_query(state):
    query = state["query"]
    history = state["history"]
    log = state["log"]
    for i, entry in enumerate(history) : 
        if 'metadata' in entry and entry['metadata'] != None :
            del history[i]

    print(f"\n\nHistorique de la conversation (après traitement) transmis au workflow : {history}")
    prompt = f"""
    You are an expert at reformulating queries. Reformulate the following user query based on the previous conversation history.
    Instructions for reformulation :
        - DO NOT change the subject, key concepts, or the intent of the question.
        - DO NOT add any new information or additional details.
        - DO NOT define any acronym. 
        - If the question refers to previous context, mention it concisely without making the question more complex.
        - Keep the reformulation simple and faithful to the original.
    Conversation history: {history}
    Original query: {query}
    Reformulated query (concise, clear, faithful, context-aware):
    """
    if history :
        new_query = llm.invoke(prompt).content
    else :
        new_query = query
    log_text = log_step("ReformulateQuery", {"query": query, "history": history}, new_query)
    log.append(log_text)
    return {"query": new_query, "log":log, "history":history}

# Step 2 — Decide if expert help is needed
def decide_expert(state):
    query = state["query"]
    history = state["history"]
    log = state["log"]
    prompt = f"""
You are a classifier. Determine whether the following query requires an expert of the aerospace project or if it can be answered directly.
You MUST output expert if : 
    - the query explicitly asks for confirmation or further informations and details
    - the query is about a part of the aeronautical project not covered in the history
    - the query can't be fully and precisely answered given the history
History: {history}
Query: "{query}"
Respond with one word only: "expert" or "direct".
    """

    decision = llm.invoke(prompt).content.strip().lower()
    log_text = log_step("DecideExpert", {"query": query}, decision)
    log.append(log_text)
    return {"decision": decision, "log":log}

# Step 3 — Direct answer path
def direct_answer(state):
    history = state["history"]
    query = state["query"]
    log = state["log"]
    system_prompt = f"""
    You are a helpful assistant who provides concise and informative answer to a user's query.
    Ground you answer as much as possible on the history. Do NOT hallucinate nor invent informations.
    """
    messages = [
        {
            "role":"system",
            "content": system_prompt
        }
    ]
    messages+=history
    messages.append(
        {
            "role":"user",
            "content":query
        }
    )
    answer = llm_mini.invoke(messages).content
    log_text = log_step("DirectAnswer", {"messages": messages}, answer)
    log.append(log_text)
    return {"final_answer": answer, "log": log, "referenced_pdf_pages":[]}

# Step 4 - Add the acronym definitions
def add_acronym_definitions(state) :
    query = state["query"]
    history = state["history"]
    log = state["log"]
    prompt = f"""
    Your task is to search for the definition of every acronym used in the user query.
    INSTRUCTIONS :
        - Search only for the acronyms that are not defined.
        - As an acronym can have several definitions, choose the most probable acronym based on the query and the history of the conversation.
        - First, search for acronym definitions, then add the meaning in the user query in parenthesis right after the acronym.
        - If you can't find a proper definition, you MUST NOT invent a definition, just leave the acronym undefined. 
        - Only output the user query with the acronyms expanded.
    EDGE CASES : 
        - DO NOT define an acronym if it is a requirement name, a test case name, or a fixture name.
        - If you can't find a perfect match for the acronym, do not define it.
        - If there are several definitions for one acronym and you are unsure about which one to pick, DO NOT define the acronym in question.
    History : {history}
    User query : "{query}"
"""
    messages = [
        {
            "role":"user",
            "content":prompt
    }
    ]
    agent = create_agent(llm, tools=[get_possible_acronym_definitions])
    result = agent.invoke(
        {'messages':messages}
    )
    print(serialize_messages(result['messages']))
    new_query = result['messages'][-1].content
    log_text = log_step("AddAcronymDefinitions", {"messages": serialize_messages(result['messages'])}, new_query)
    log.append(log_text)
    return({"query":new_query, 'log':log})

# Step 5 : End-to-end ReAct agent to ask subquestions related to the query 
def ask_subquestions_with_react_agent(state) :
    query = state["query"]
    history = state["history"]
    log = state["log"]
    system_prompt = f"""You are a highly adaptable reasoning agent who asks experts to reflect about the user's query.
The queries you must process are about an aerospace project.
Your goal is to gather detailed information about the user's complex query by dynamically generating and asking sub-questions. 

Instructions :
- You MUST not answer the user's query.
- After collecting all necessary informations, you MUST only output that you are done.
- Do not ask the user for any guidance.
- You can interact with LLMs that have access to relevant documents and informations about an aerospace project.

When interacting with the LLMs :
    1. Ask them for clarification or details whenever needed.
    2. Do not hesitate to refine your new questions with the answers you get, to answer the original query deeper.
    3. You MUST NOT introduce any new concepts, definitions, acronyms or assumptions that were not mentioned in the original query.
    4. Stay faithful to the original query.
    5. It is best to not define the acronyms in the queries.
    6. Do not format the questions in any way.
    7. Keep in mind that the LLMs do not have access to the history of your conversation with them.

You may need to ask multiple rounds of questions to gather enough details to answer the query comprehensively."""
    global statistics
    statistics = []
    final_answer, messages = run_agent(query, system_prompt, history, llm)
    retrieval_statistics = "Retrieval statistics :\n"+ "\n\n".join([f"Retrieval {i+1} :\n{statistics[i]}" for i in range(len(statistics))])
    print(serialize_messages(messages))
    log_text = log_step("AskSubquestionsWithReactAgent", {"messages": serialize_messages(messages)}, "", retrieval_statistics = retrieval_statistics)
    log.append(log_text)
    return({"final_answer": final_answer, "log":log, "retrieval_statistics":retrieval_statistics, "agent_messages":serialize_messages(messages)})

# Step 6 : Produce the final answer based on the informations collected by the ReAct agent
def produce_final_answer(state) :
    query = state["query"]
    log = state["log"]
    history = state["history"]
    agent_messages = state["agent_messages"]
    agent_messages = agent_messages[2+len(history):-1]
    if len(agent_messages) == 2:
        final_answer = agent_messages[-1]['content']
        log_text = log_step("ProduceFinalAnswer", {"messages":"The tool call made by the agent."}, "Response from the tool.")
        log.append(log_text)
        return({"final_answer": final_answer, "log":log})
    system_prompt = f"""Your goal is produce a final answer to the user's query about an aerospace project. 
To help you do so, an agent asked LLM experts of the aerospace project to collect relevant informations related to the user's query. 

When generating your final response :
    1. FULLY answer the original user's query. 
    2. You MUST quote the sources given by the LLMs, including the document names, pages, and sections if given.
    3. Stay faithful to the way the LLMs are quoting their sources. Ensure the sources are explicitly and systematically quoted after each paragraph of information.
    4. ONLY answer based on the informations provided by the LLMs.
    5. Do not talk about the expert LLMs or your reflexion, only output the well refined answer to the user's query.
    6. Write your response in clear and well-structured Markdown.
    7. Organize the content with small, subtle sections or bullet points, and highlight key terms with bold or italics where helpful.
    8. Avoid large titles or headers — keep the tone natural and conversational, as if speaking directly to the user.
    9. Output formulas as LaTeX, do not use Unicode escape sequences.
    10. Ensure the response is visually easy to read without breaking the flow of a conversation.
"""
    user_prompt = f"""The user's query : {query}
The agent dialog with the LLM experts : {agent_messages}
"""
    messages = [{'role':'system', 'content':system_prompt}]
    if history : 
        messages.extend(history)
    messages += [{'role':'user', 'content':user_prompt}]
    final_answer = llm.invoke(messages).content
    log_text = log_step("ProduceFinalAnswer", {"messages": messages}, final_answer)
    log.append(log_text)
    return({"final_answer": final_answer, "log":log})

# Step 7 : Get the pages referenced in the final answer
def get_referenced_pages(state) :
    final_answer = state["final_answer"]
    log = state["log"]
    retrieval_statistics = state['retrieval_statistics']
    system_prompt = f"""Your task is to output a list of the sources referenced in the following response.  
**INSTRUCTIONS:**  
- DO NOT invent any references or page numbers.  
- DO NOT add any code tags.  
- If no document is referenced, output [].  
- Respect the order in which the references appear in the response.  
- ONLY OUTPUT a Python list formatted exactly like this: [(<string: document name>, <int: page number>), ...]  
- If the referenced document is a .html or an Excel file (.xls or .xlsx), you MUST add (<string: document name>, 1) to the list.  

For example:  
[("doc1", 10), ("doc2", 5), ("doc3.xls", 1), ("doc4.html", 1), ...]  

The references MUST be among the following potential references given.
    """ 
    messages = [{"role":"system", "content": system_prompt}]
    prompt = f"""
Get the list of the sources referenced in the following response : \n{final_answer}

Potential references : \n{retrieval_statistics}
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
    images_by_doc = defaultdict(list)
    doc_names = list(set([doc_name for doc_name, _ in ref_list]))
    final_answer = rendre_citation_cliquable(final_answer, doc_names)
    for doc_name, page_number in ref_list:
        try:
            from my_paths import RAW_DATA_DIR, RAW_DATA_DIR_B
            if not(Path(doc_name).suffix in [".xls",".xlsx",".html"]):
        
                doc_path = find_document_path(doc_name, data_dirs=[RAW_DATA_DIR, RAW_DATA_DIR_B])
                page_PIL = get_page_image(doc_path, page_number)
                # On stocke avec le numéro de page pour pouvoir trier ensuite
                images_by_doc[doc_name].append((page_number, page_PIL, f"Page {page_number} de {doc_name}"))
        except Exception as e:
            print(f"Page {page_number} du document {doc_name} non trouvée : {e}")

    # Tri des pages pour chaque document
    for doc_name in images_by_doc:
        images_by_doc[doc_name].sort(key=lambda x: x[0])  # tri par numéro de page

    # Reconstruire la liste finale d'images, regroupée par document,
    # en enlevant le numéro de page (premier élément du tuple)
    img_list = []
    for doc_name, pages in images_by_doc.items():
        for _, img, desc in pages:
            img_list.append((img, desc))

    log_text = log_step("GetReferencedPages", {}, answer)
    log.append(log_text)
    return {"final_answer":final_answer, "referenced_pdf_pages": img_list, "log": log}

# Build LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict
    

class GraphState(TypedDict):
    query: str           # La requête de l'utilisateur
    history: list        # Historique de la conversation
    decision: str        # Décision sur si un expert est nécessaire
    final_answer: str    # Réponse finale
    log: list            # Logging
    referenced_pdf_pages: list # Pages utilisées pour répondre
    retrieval_statistics: str
    agent_messages: list 

graph = StateGraph(GraphState)
# Nodes
graph.add_node("ReformulateQuery", reformulate_query)
graph.add_node("DecideExpert", decide_expert)
graph.add_node("DirectAnswer", direct_answer)
graph.add_node("AddAcronymDefinitions", add_acronym_definitions)
graph.add_node("AskSubquestionsWithReactAgent", ask_subquestions_with_react_agent)
graph.add_node("ProduceFinalAnswer", produce_final_answer)
graph.add_node("GetReferencedPages", get_referenced_pages)

# Edges
graph.add_edge("ReformulateQuery", "DecideExpert")

# Conditional path: expert or direct
graph.add_conditional_edges(
    "DecideExpert",
    lambda state: state["decision"],
    {
        "direct": "DirectAnswer",
        "expert": "AddAcronymDefinitions",
    },
)
graph.add_edge("DirectAnswer", END)

# Expert branch sequence
graph.add_edge("AddAcronymDefinitions","AskSubquestionsWithReactAgent")
graph.add_edge("AskSubquestionsWithReactAgent","ProduceFinalAnswer")
graph.add_edge("ProduceFinalAnswer","GetReferencedPages")
graph.add_edge("GetReferencedPages", END)

# Entry point
graph.set_entry_point("ReformulateQuery")

# Compile workflow
workflow = graph.compile()

def invoke_workflow(query, history, analyse_mode, doc_choices) :
    global DOC_CHOICES
    global ANALYSE_MODE
    ANALYSE_MODE = analyse_mode
    DOC_CHOICES = doc_choices
    response = workflow.invoke({"query": query, "history": history, "log":[]})
    return(response["final_answer"], response["log"], response["referenced_pdf_pages"])