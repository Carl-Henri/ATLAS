from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import Term, Or
from tqdm import tqdm
from my_paths import *

from mistral_langchain_wrapper import MistralChatWrapper
import os 
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

llm_mistral_medium = MistralChatWrapper(api_key=API_KEY, model="medium")

llm = llm_mistral_medium

TEST_DIR = ""

def get_chunks_of_test_case_containing_HLR_name(doc_name, HLR_name) :
    doc_filter = [doc_name]
    
    ix = index.open_dir(WHOOSH_DIR)
    with ix.searcher() as searcher:
        parser = QueryParser("content", schema=ix.schema)
        q = parser.parse(HLR_name)
        filter_q = None
        if doc_filter:
            terms = [Term("doc_name", name) for name in doc_filter]
            filter_q = Or(terms)
        results_bm25 = searcher.search(q, filter=filter_q, limit=None, terms=True)
        res_bm25 = [r['content'] for r in results_bm25]
    return(res_bm25)

BATCH_SIZE = 2

def generate_report_on_how_a_HLR_is_tested_stream(HLR_name, HLR_full_text, data_dir = TEST_DIR):
    test_case_document_paths = [str(f) for f in Path(data_dir).rglob("*")]
    test_cases = []

    for test_case_path in tqdm(test_case_document_paths):
        if not(Path(test_case_path).is_file()):
            continue
        try:
            with open(test_case_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if HLR_name in content:
                    test_cases.append(Path(test_case_path).name)
        except Exception as e:
            print(f"Impossible d'ouvrir {test_case_path}: {e}")

    system_prompt = """You are a reasoning agent specialized in aerospace software development and verification.

**Your task for each batch of test cases (provided two at a time) is as follows**:
- For each test case, provide a detailed explanation of how and to what extent it verifies or covers the specified High Level Requirement (HLR) within the context of an aerospace project.
- For every test case, systematically present:
   - The relationship between the test steps and the individual statements in the HLR.
   - How the test input data, procedures, and expected outcomes address all aspects of the HLR.
- Analyze the algorithms or code referenced in each test case. Explicitly confirm whether the implementation actually tests the HLR as intended.
   - For every algorithm or method call in the test, describe how it contributes to validating the HLR, and identify any potential gaps or mismatches.
- If any part of the test does not cover aspects of the HLR, mention it concisely and suggest what might be missing.

**Inputs**:
- The full text of the HLR.
- The provided test case documentation or code excerpts referencing this HLR (two per batch).

**Instructions**:
- Base your analysis ONLY on the context provided.
- Provide a rigorous and logical breakdown for EVERY test case given by the user.
- Always explicitly state the full name of each test case as given by the user.
- Do NOT include markdown links in your response.
- Output a detailed, standalone explanation for each test case describing exactly how it tests the HLR, with no introduction, summary, or conclusion.
- Quote SYSTEMATICALLY the sources you use for each paragraph of your analysis, specifying the name of the test case document and, if available, the exact section.

If any necessary information is missing or unclear, state your assumptions before conducting your analysis.
"""

    final_responses = []
    final_usage_infos = []
    contexts_used = []

    for i in range(0, len(test_cases), BATCH_SIZE):
        batch = test_cases[i:i + BATCH_SIZE]
        context = "\n\n".join([s for test_case in batch for s in get_chunks_of_test_case_containing_HLR_name(test_case, HLR_name)])
        user_prompt = f"""Now thoroughly explain how the given tests cover the HLR in the aerospace project.
The full text of the HLR : {HLR_full_text}
The list of test cases to explain : {batch} 
The context : {context}"""
        contexts_used.append(context)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        result = llm.invoke(messages)
        response = result.content
        usage = result.additional_kwargs.get('metadata').get('usage')
        usage_infos = f"prompt_tokens : {usage.get('prompt_tokens')}, total_tokens : {usage.get('total_tokens')}, completion_tokens : {usage.get('completion_tokens')}"
        final_responses.append(response)
        final_usage_infos.append(usage_infos)

        # Stream chaque batch d'analyse avec yield
        yield {
            "batch": batch,
            "context_used": context,
            "response": response,
            "usage_infos": usage_infos,
            "step": "batch_analysis"
        }

    # Quand tous les batchs sont finis, génération du rapport global
    final_response = "\n\n".join(final_responses)
    final_usage_infos_str = "\n".join(final_usage_infos)

    system_prompt_to_generate_the_final_report = """You are a reasoning agent specialized in aerospace software development.

Your task is to produce a comprehensive report detailing how the specified High Level Requirement (HLR) is tested within the aerospace project.  
You are provided with :
- The full text of the HLR.
- A collection of descriptive analyses, each explaining how individual test cases verify the HLR.

**Instructions :**
- Base your response ONLY on the provided context.
- Mention explicitly the full name of every test case referenced in the analyses.
- Do NOT include markdown links.
- For every element in your explanation, SYSTEMATICALLY and FAITHFULLY quote the sources as they appear in the analyses, including document names, page numbers, and section identifiers when available.
- Adhere strictly to the citation style found in the source analyses; each paragraph must end with its explicit list of sources.

Do not include introductory or concluding remarks; provide only the detailed, referenced content as outlined above.
"""
    user_prompt_to_generate_the_final_report = f"""Now thoroughly analyse how the HLR is tested in the aerospace project.
The full text of the HLR : {HLR_full_text}
The context : {final_response}"""

    messages = [
        {'role': 'system', 'content': system_prompt_to_generate_the_final_report},
        {'role': 'user', 'content': user_prompt_to_generate_the_final_report}
    ]
    result = llm.invoke(messages)
    report = result.content
    usage = result.additional_kwargs.get('metadata').get('usage')
    usage_infos = f"prompt_tokens : {usage.get('prompt_tokens')}, total_tokens : {usage.get('total_tokens')}, completion_tokens : {usage.get('completion_tokens')}"
    final_usage_infos_str += "\n" + usage_infos

    # Stream le rapport final (avec yield)
    yield {
        "report": report,
        "usage_infos": final_usage_infos_str,
        "responses": final_responses,
        "contexts_used": contexts_used,
        "step": "final_report"
    }

def compare_test_analysis_to_HLR_analysis(HLR_analysis, report_on_test_cases) :
    system_prompt = """You are a reasoning agent specialized in aerospace software development.  
Your task is to analyze whether a High Level Requirement (HLR) is fully covered by the test cases in the aerospace project.  
To do this, you are given an analysis of the HLR in addition to a report on how the HLR is tested in the project.

Based on this context, you must complete the report by checking if the test cases address the following aspects:  

1. Each input condition must be tested individually, and all possible combinations of inputs must be exhaustively covered. Output states must be evaluated for every individual input change as well as for every combination of input changes.  
*Note: For this aspect, ONLY take into account the explicit input conditions.*  
2. Timing between input changes and output state transitions must be tested **only if** it is explicitly specified in the HLR.  
3. **Only if** the HLR explicitly specifies performance criteria involving computed data values, their accuracy and validity must be thoroughly verified; otherwise, timing and performance testing are not required.  
4. Data values and their validity that are not computed by the specified action must also be verified.  
5. Priority rules among different actions within the same HLR or across multiple HLRs must be tested comprehensively.  

INSTRUCTIONS:  
- Your response must be based ONLY on the context provided.  
- ONLY check the test coverage of the HLR. Any information concerning another HLR must be ignored.  
- Output only your report on the gaps (if any) in the coverage of the HLR.  
- You MUST quote the sources given in the context given, including the document names, pages, and sections if given.
- Stay faithful to the way the sources are quoted. Ensure the sources are explicitly and systematically quoted after each paragraph of information.
"""
        
    user_prompt = f"""Now generate a report on the gaps (if any) in the coverage of the HLR.
The HLR analysis : {HLR_analysis}
The report on how the HLR is tested in the project : {report_on_test_cases}"""

    messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':user_prompt}]
    result = llm.invoke(messages)
    response = result.content
    usage = result.additional_kwargs.get('metadata').get('usage')
    prompt_tokens = usage.get('prompt_tokens')
    total_tokens = usage.get('total_tokens')
    completion_tokens = usage.get('completion_tokens')
    usage_infos = f"prompt_tokens : {prompt_tokens}, total_tokens : {total_tokens}, completion_tokens : {completion_tokens}"
    return(response, usage_infos)

from workflows.workflow_utils import add_links
import json
import os
def workflow_HLT_coverage(HLR_name, HLR_analysis, HLR_full_text):
    save_file = 'saved_results/HLT_coverage.json'

    # Charger/initialiser l’état de la sauvegarde
    if os.path.exists(save_file):
        with open(save_file, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
    else:
        save_data = {}

    if HLR_name not in save_data:
        save_data[HLR_name] = {}
    responses = save_data[HLR_name]

    log_strings = []

    # 1. Etape : "Report on how the HLR is tested" (streamé)
    step1 = "Report on how the HLR is tested"
    gen_step1 = False
    if step1 in responses:
        report_on_how_the_HLR_is_tested = responses[step1]
        usage_infos, test_cases_analysis, contexts_used = None, None, None
        yield step1, "# Report on how the HLR is tested\n" + add_links(report_on_how_the_HLR_is_tested)
    else:
        yield step1, "Generation of the report (streaming analysis, batch by batch)..."
        batches = []
        report_on_how_the_HLR_is_tested = None
        usage_infos = None
        test_cases_analysis = None
        contexts_used = None

        # Streaming analysis, batch by batch
        for result in generate_report_on_how_a_HLR_is_tested_stream(HLR_name, HLR_full_text):
            if result["step"] == "batch_analysis":
                # Affichage dès qu'un batch est traité
                batches.append(result)
                yield step1, "\n\n".join([f"Batch {" ".join(result['batch'])} analysis:\n{result['response']}" for result in batches])
            elif result["step"] == "final_report":
                report_on_how_the_HLR_is_tested = result["report"]
                usage_infos = result["usage_infos"]
                test_cases_analysis = result["responses"]
                contexts_used = result["contexts_used"]
                responses[step1] = report_on_how_the_HLR_is_tested
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                yield step1, "# Report on how the HLR is tested\n" + add_links(report_on_how_the_HLR_is_tested)
                gen_step1 = True

    # log
    if gen_step1:
        log_step1 = (
            "===== Step 1: Report on how the HLR is tested =====\n"
            f"Usage Infos:\n{usage_infos}\n\n"
            f"Contexts Used:\n{contexts_used}\n"
            f"Test Cases Analysis:\n{test_cases_analysis}\n\n"
            "Generation of the report."
        )
    else:
        log_step1 = (
            "===== Step 1: Report on how the HLR is tested =====\n"
            "(Already loaded from save file)\n"
        )
    log_strings.append(log_step1)

    # 2. Etape : "Report on HLT coverage"
    step2 = "Report on HLT coverage"
    gen_step2 = False
    if step2 in responses:
        report_on_HLT_coverage = responses[step2]
        usage_infos_2 = None
        yield step2, "# Report on HLT coverage\n" + add_links(report_on_HLT_coverage)
    else:
        yield step2, "Generation of the report..."
        prev_report = responses[step1]
        report_on_HLT_coverage, usage_infos_2 = compare_test_analysis_to_HLR_analysis(HLR_analysis, prev_report)
        responses[step2] = report_on_HLT_coverage
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        yield step2, "# Report on HLT coverage\n" + add_links(report_on_HLT_coverage)
        gen_step2 = True

    # logs
    if gen_step2:
        log_step2 = (
            "===== Step 2: Report on HLT coverage =====\n"
            f"Usage Infos:\n{usage_infos_2}\n"
            f"Context given : the HLR analysis and the report on how the HLR is tested."
            "Generation of the report."
        )
    else:
        log_step2 = (
            "===== Step 2: Report on HLT coverage =====\n"
            "(Already loaded from save file)\n"
        )
    log_strings.append(log_step2)

    # 3. Log global
    full_log = "\n".join(log_strings)
    yield "Workflow log", full_log