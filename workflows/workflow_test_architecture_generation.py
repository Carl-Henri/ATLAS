import os
import json

from agentic_workflow_utils import log_step
from my_paths import *
from mistral_langchain_wrapper import MistralChatWrapper

API_KEY = os.getenv('API_KEY')
llm_mistral_medium = MistralChatWrapper(api_key=API_KEY, model="medium")

llm = llm_mistral_medium

SAVE_FILE = "saved_results/test_architecture_generation.json"

def generate_test_architecture(state) :
    query = state['query']
    HLR_analysis = state['HLR_analysis']
    log = []
    
    # Chargement si existant de l'analyse du HLT coverage déjà faite
    indice = None
    try :
        with open(SAVE_FILE,'r',encoding='utf-8') as f :
            saved_results = json.load(f)
    except : 
        saved_results = []

    response = None
    indice = None
    for i, entry in enumerate(saved_results) :
        if entry['query'] == query :
            response = entry['result']
            indice = i
            break    

    system_prompt = """You are an analytical agent specializing in aerospace software development. You have been provided with a detailed analysis of a High Level Requirement (HLR) for an aerospace project.

Your task is to develop a systematic and comprehensive test architecture that guarantees complete coverage of the specified requirement. 

Organize the test architecture into well-defined test categories and individual test cases.

Prioritize thoroughness, clarity, and verifiability while adhering to aerospace software standards and development plans.

The test architecture MUST address the following aspects:  
1. Each input condition must be tested individually, and all possible combinations of inputs must be exhaustively covered. Output states must be evaluated for every individual input change as well as for every combination of input changes.  
Note : for this aspect, ONLY take into account the explicit input conditions.
2. Timing between input changes and output state transitions must be tested **only if** it is explicitly specified in the HLR.  
3. **Only if** the HLR explicitly specifies performance criteria involving computed data values, their accuracy and validity must be thoroughly verified; otherwise, timing and performance testing are not required.  
4. Data values and their validity that are not computed by the specified action must also be verified.  
5. Priority rules among different actions within the same HLR or across multiple HLRs must be tested comprehensively.  

Additional instructions: 
- Ensure that the tests within each test category are presented in a well-organized and clearly structured table format.
- Write your response in clear and well-structured Markdown.
- Do NOT include discussions of test environments or tools.  
- Do NOT mention test execution or reporting processes.  
- Ensure exhaustive coverage of all possible explicit input combinations without exception.  
- Do NOT mention aerospace software standards and plans. Adhere to them, but do not discuss them."""
    
    prompt = f"""The HLR analysis : {HLR_analysis}"""
    messages = [
        {
            'role':'system',
            'content':system_prompt
        },
        {
            'role':'user',
            'content':prompt
        }
    ]

    if response == None :
        result = llm.invoke(messages)
        response = result.content
        if llm == llm_mistral_medium :
            usage = result.additional_kwargs.get('metadata').get('usage')
        """elif llm == llm_gpt :
            usage = result.response_metadata['token_usage']"""
        prompt_tokens = usage.get('prompt_tokens')
        total_tokens = usage.get('total_tokens')
        completion_tokens = usage.get('completion_tokens')
        log_text = log_step(f"GenerateTestArchitecture", {}, f"prompt_tokens : {prompt_tokens}, total_tokens : {total_tokens}, completion_tokens : {completion_tokens}", log_file="logs/log_test_architecture_generation_workflow_fast.json", log_file_text="logs/log_test_architecture_generation_workflow_fast.txt")
        log.append(log_text)
    
    try :
        with open(SAVE_FILE,'r',encoding='utf-8') as f :
            saved_results = json.load(f)
    except : 
        saved_results = []

    with open(SAVE_FILE,'w',encoding='utf-8') as f :
        if indice != None :
            del saved_results[indice]
        saved_results.append({'query':query, 'result':response})
        json.dump(saved_results, f, ensure_ascii=False, indent=2) 
    
    return {'final_answer':"# Test architecture\n" + response, 'log':log}

# Build LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict
    
class GraphState(TypedDict):
    query: str           # La requête de l'utilisateur (nom du HLR)
    final_answer : str
    log : list           # Listes des actions entreprises par le workflow
    HLR_analysis : str


graph = StateGraph(GraphState)

# Nodes
graph.add_node("GenerateTestArchitecture", generate_test_architecture)

# Edges
graph.add_edge("GenerateTestArchitecture", END)

# Entry point
graph.set_entry_point("GenerateTestArchitecture")

# Compile workflow
workflow = graph.compile()

def generate_test_architecture(query, HLR_analysis) :
    response = workflow.invoke({"query": query, "HLR_analysis": HLR_analysis})
    return(response['final_answer'])
    