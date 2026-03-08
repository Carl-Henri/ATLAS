
import os
import re
import json

from agentic_workflow_utils import log_step, collect_statistics, find_document_path, get_page_image, rendre_citation_cliquable
from my_paths import *

from collections import defaultdict

from pathlib import Path
from mistral_langchain_wrapper import MistralChatWrapper
from hybrid_search import hybrid_search
from workflows.workflow_utils import get_req_chunks

API_KEY = os.getenv('API_KEY')
llm_mistral_medium = MistralChatWrapper(api_key=API_KEY, model="medium")

llm = llm_mistral_medium

from utils import get_filter
ABL_FILTER = get_filter(['ABL'], RAW_DATA_DIR)
FILTER_TO_EXCLUDE_TESTS = [str(f.name).replace('.pdf','') for f in Path(RAW_DATA_DIR).rglob("*") if not ('pages Fitnesse' in str(f))]

SAVE_FILE = "saved_results/HLR_analysis_results.json"

# Étape 1 :
def analyze_HLR(state) :
    query = state['query']
    log = []
    statistics = []

    # Chargement des résultats sauvegardés
    indice = None
    complete = False
    try :
        with open(SAVE_FILE,'r',encoding='utf-8') as f :
            saved_results = json.load(f)
    except : 
        saved_results = []

    workflow_responses = {}
    for i, entry in enumerate(saved_results) :
        if entry['query'] == query :
            workflow_responses = entry['workflow_responses']
            indice = i
            break
            
    steps = [
        ('1. Output the entire text of the HLR found in the context (including everything that is specified e.g. title, DFT, Upper, description, rationale, notes...).',
        ''),
        ('2. Exhaustively identify all input conditions of the HLR.',
        'Be thorough and try to identify the explicit / implicit / contextual input conditions if such input conditions exist.'),
        ('3. Then, thoroughly determine all actions specified for each input condition.',
        ''),
        ('4. Next, comprehensively identify all data modified by the specified actions.',
        ''),
        ('5. Fully describe all output states after action completion and how input conditions are updated.',
        ''),
        ("6. Determine if any timing constraints exist between input state changes and output state changes.", 
        'If it is not specified in the HLR, do not elaborate further and answer that there are no specifications. Timings for this HLR may be specified in other HLRs though. If and ONLY if it is the case, inform the user of it. Otherwise, keep your answer CONCISE.'),
        ("7. Establish the performance targets for the computed data.",
        'If it is not specified in the HLR, do not elaborate further and answer that there are no specifications. Performances for this HLR may be specified in other HLRs though. If and ONLY if it is the case, inform the user of it. Otherwise, keep your answer CONCISE.'),
    ]
    
    for step, _ in steps :
        complete = True
        if not(step in workflow_responses) or 'Erreur API' in workflow_responses[step] :
            complete = False
    
    if not(complete) :
        doc_filter = ABL_FILTER
        context = hybrid_search(query, chroma_dir=CHROMA_DIR, whoosh_dir=WHOOSH_DIR, doc_filter=doc_filter, top_k=400)

    system_prompt = f"""You are an expert in interpreting the role of High-Level Requirements (HLRs) within aerospace projects. 
Respond to the user’s query using the given context, providing thorough and precise information. Avoid redundancy by refraining from adding introductions or conclusions, or by doing a summary while your answer is rather short.
However, a detailed summary presented in a clear, well-structured table is encouraged. 

INSTRUCTIONS :
- Do NOT use any of your knowledge or invent any information. Only answer based on the provided context.
- You MUST explicitly and systematically quote the sources used to generate each paragraph of information.
- For every distinct paragraph or semantic unit you output, indicate clearly the exact document name and the page number from which the information was retrieved.
- This citation must immediately follow the related paragraph. Do not omit or group citations; each informational paragraph must be individually sourced.

Instructions about HOW TO PROCESS THE PROVIDED CONTENT :
- In the context provided, there are <!-- Figure caption: <CAPTION> --> that describes the figure above ; do not quote it.
- There are also tags like <!-- Table markdown 1 start --> MARKDOWN TABLE <!-- Table markdown end --> that indicates a markdown table. DO NOT quote these tags directy.
- You MUST NOT quote the number of the table referenced in these markdown tags.
- Refer to the page number and the name of the document given at the start of each chunk.
- The chunk hierarchy included at the beginning of each context chunk is provided to give you information about the source of the information.
- Keep in mind the user do not have access to the context provided to you, so do not say 'the provided context' as if the user had access to it. 
"""
    responses = {}
    for i, (step, instructions) in enumerate(steps) :
        yield {'step': step, 'response':"Generation in progress..."}
        if complete :
            response = workflow_responses[step]
            responses[step] = response
            yield {'step': step, 'response':response}
            continue
        if i == 0 :
            hlr_chunks = get_req_chunks(ABL_FILTER, query, keywords = ['Title'])
            response = hlr_chunks[0]
            log_text = log_step(f"AnalyzeHLR : {step}", {"query": query}, f"response : {response}", log_file="logs/log_HLR_workflow_fast.json", log_file_text="logs/log_HLR_workflow_fast.txt")
            log.append(log_text)
            responses[step] = response   
        else :
            str_context = "\n\n".join([doc.page_content for doc in context[:60]])
            statistics = [collect_statistics(context[:60])]
            retrieval_statistics = "Retrieval statistics :\n"+ f"Retrieval :\n{statistics[0]}"
            previous_steps = "\n".join([f"{steps[j]}\n{list(responses.values())[j]}" for j in range(i)])
            prompt = f"""Based on the following context and the result of the previous steps, carry out the following task on the following HLR.
The previous steps : {previous_steps}
The HLR : {query}
The task : {step[3:]}"""
            if instructions != '' :
                prompt += f"\nFurther instructions :\n{instructions}"
            prompt_with_context = prompt + f"\nThe context : {str_context}"

            messages = [
                {
                    'role':'system',
                    'content':system_prompt
                },
                {
                    'role':'user',
                    'content':prompt_with_context
                }
            ]
            result = llm.invoke(messages)
            response = result.content
            usage = result.additional_kwargs.get('metadata').get('usage')
            prompt_tokens = usage.get('prompt_tokens')
            total_tokens = usage.get('total_tokens')
            completion_tokens = usage.get('completion_tokens')
            
            log_text = log_step(f"AnalyzeHLR : {step}", {"query": query, "prompt":prompt}, f"prompt_tokens : {prompt_tokens}, total_tokens : {total_tokens}, completion_tokens : {completion_tokens}, response : {response}", retrieval_statistics=retrieval_statistics, log_file="logs/log_HLR_workflow_fast.json", log_file_text="logs/log_HLR_workflow_fast.txt")
            log.append(log_text)
            responses[step] = response
        yield {'step': step, 'response':response}

    HLR_analysis = ''
    for step, response_md in responses.items():
        HLR_analysis += f"# {step}\n\n{response_md}\n\n"

    workflow_responses = workflow_responses | responses 
    state = state | {'HLR_analysis':HLR_analysis, 'statistics':statistics, 'log':log, 'workflow_responses':workflow_responses, 'complete':complete, 'indice':indice}
    yield state

# Étape 2 :
def analyze_HLR_part_2(state) :
    query = state['query']
    HLR_analysis = state['HLR_analysis']
    statistics = state['statistics']
    log = state['log']
    workflow_responses = state['workflow_responses']
    indice = state['indice']
    complete = state['complete']

    steps = [
        ("8. Thoroughly assess, when an action specified in the HLR is carried out, which other HLRs may impact the output states of the action.", "Your goal is to determine, when an action specified in the HLR is carried out, if the input conditions for this action are going to trigger other actions in other HLR that could have an impact on the outputs, and thus affect the correct realization of the action or even prevent it from completing.")
    ]
    system_prompt = f"""You are an expert in interpreting the role of High-Level Requirements (HLRs) within aerospace projects. 
Respond to the user’s query using the given context, providing thorough and precise information. Avoid redundancy by refraining from adding introductions or conclusions.
However, a detailed summary presented in a clear, well-structured table is encouraged.

INSTRUCTIONS :
- Do NOT use any of your knowledge or invent any information. Only answer based on the provided context.
- DO NOT include an introduction, conclusion, or summary in your response. Just follow the steps.
- You MUST explicitly and systematically quote the sources used to generate each paragraph of information.
- For every distinct paragraph or semantic unit you output, indicate clearly the exact document name and the page number from which the information was retrieved.
- This citation must immediately follow the related paragraph. Do not omit or group citations; each informational paragraph must be individually sourced.

Instructions about HOW TO PROCESS THE PROVIDED CONTENT :
- In the context provided, there are <!-- Figure caption: <CAPTION> --> that describes the figure above ; do not quote it.
- There are also tags like <!-- Table markdown 1 start --> MARKDOWN TABLE <!-- Table markdown end --> that indicates a markdown table. DO NOT quote these tags directy.
- You MUST NOT quote the number of the table referenced in these markdown tags.
- Refer to the page number and the name of the document given at the start of each chunk.
- The chunk hierarchy included at the beginning of each context chunk is provided to give you information about the source of the information.
- Keep in mind the user do not have access to the context provided to you, so do not say 'the provided context' as if the user had access to it. 
"""
    already_done = True
    if not(complete) :
        already_done = False
    else :
        for step, _ in steps :
            if not(step in workflow_responses) : 
                already_done = False
            elif 'Erreur API' in workflow_responses[step] :
                already_done = False

    if not(already_done) :
        context = hybrid_search(query, chroma_dir=CHROMA_DIR, whoosh_dir=WHOOSH_DIR, doc_filter=FILTER_TO_EXCLUDE_TESTS, top_k=400)
        str_context = "\n\n".join([doc.page_content for doc in context[:60]])
        current_statistics = [collect_statistics(context[:60])]
        retrieval_statistics = "Retrieval statistics :\n"+ "\n\n".join([f"Retrieval {i+1} :\n{current_statistics[i]}" for i in range(len(current_statistics))])
    
    responses = {}
    for step, instructions in steps :
        yield {'step': step, 'response':"Generation in progress..."}

        if step in workflow_responses : 
            if not('Erreur API' in workflow_responses[step]) and complete : 
                response = workflow_responses[step]
                responses[step] = response
                yield {'step': step, 'response':response}
                continue
        prompt = f"""Based on the following context and the analysis of the HLR {query}, please fully answer the following query.

The HLR analysis : 
{HLR_analysis}

The query : 
{step[3:]}
"""
        if instructions != '' :
            prompt += f"\nFurther instructions :\n{instructions}"
        prompt += f"""
The context : 
{str_context}
"""
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
        result = llm.invoke(messages)
        response = result.content
        usage = result.additional_kwargs.get('metadata').get('usage')
        prompt_tokens = usage.get('prompt_tokens')
        total_tokens = usage.get('total_tokens')
        completion_tokens = usage.get('completion_tokens')
        responses[step] = response
        log_text = log_step(f"AnalyzeHLRPart2 : {step}", {}, f"prompt_tokens : {prompt_tokens}, total_tokens : {total_tokens}, completion_tokens : {completion_tokens}, response : {response}", retrieval_statistics=retrieval_statistics, log_file="logs/log_HLR_workflow_fast.json", log_file_text="logs/log_HLR_workflow_fast.txt")
        log.append(log_text)
        statistics.extend(current_statistics)
        yield {'step': step, 'response':response}
    
    HLR_analysis_part_2 = responses
    final_analysis = ""
    # Ajout du HLR_analysis initial
    final_analysis += HLR_analysis + "\n\n" 

    # Pour chaque étape dans HLR_analysis_part_2, ajouter le step suivi de la response
    for step, response_md in HLR_analysis_part_2.items():
        final_analysis += f"## {step}\n\n{response_md}\n\n"

    workflow_responses = workflow_responses | responses
    try :
        with open(SAVE_FILE,'r',encoding='utf-8') as f :
            saved_results = json.load(f)
    except : 
        saved_results = []

    with open(SAVE_FILE,'w',encoding='utf-8') as f :
        if indice != None :
            del saved_results[indice]
        saved_results.append({'query':query, 'workflow_responses':workflow_responses})
        json.dump(saved_results, f, ensure_ascii=False, indent=2) 

    state = state | {'HLR_analysis_part_2':responses, 'final_answer':final_analysis, 'statistics':statistics, 'log':log, 'workflow_responses': workflow_responses}
    yield(state)

# Étape 3 :
def get_referenced_pages(state) :
    final_answer = state["final_answer"]
    statistics = state['statistics']
    if statistics :
        retrieval_statistics = "Retrieval statistics :\n"+ "\n\n".join([f"Retrieval {i+1} :\n{statistics[i]}" for i in range(len(statistics))])
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

The references MUST be among the following potential references given.
    """ 
        messages = [{"role":"system", "content": system_prompt}]
        prompt = f"""
    Now get the list of sources referenced in the following response : \n{final_answer}

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
            
    state = state | {"final_answer":"# HLR Analysis\n\n"+final_answer}
    return state

def invoke_HLR_analysis_workflow(query) :
    state = {"query": query}
    for result in analyze_HLR(state) :
        if 'step' in result : 
            yield result
        else : 
            state = result
    for result in analyze_HLR_part_2(state) :
        if 'step' in result : 
            yield result
        else : 
            state = result
    final_state = get_referenced_pages(state)
    yield final_state
        
