from ragas import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, LLMContextPrecisionWithoutReference, Faithfulness
import os
import json
from tqdm import tqdm
from ragas.run_config import RunConfig
from mistral_langchain_wrapper import MistralLLMWrapper
from bge_m3_langchain_wrapper import MyAPIEmbedding
from dotenv import load_dotenv

# Charger les variables depuis le fichier .env
load_dotenv()

API_KEY = os.getenv('API_KEY')

def load_dataset(results_path) : 
    if os.path.exists(results_path) : 
        with open(results_path, "r", encoding='utf-8') as f:
            data = json.load(f)

    else : 
        data = []
    dataset = []
    for entry in data : 
        dataset.append({
            "user_input": entry["query"],
            "response": entry["response"],
            "retrieved_contexts": entry["retrieved_content"]
        })
    return(dataset)

import math

def not_done(fichier, query):
    metrics = ["faithfulness", "context_precision", "response_relevancy"]
    if os.path.exists(fichier):
        with open(fichier, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Le fichier JSON doit contenir une liste.")
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    for entry in data:
        # Si l'entrée n'a pas le champ 'query', on considère aucune métrique d'évaluée
        if 'query' not in entry:
            return metrics
        
        if entry['query'] == query:
            missing_metrics = []
            for m in metrics:
                if m not in entry:
                    missing_metrics.append(m)
                else:
                    val = entry[m][0]
                    # Vérifie si la valeur est NaN (float("nan")) ou None
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        missing_metrics.append(m)
            return missing_metrics if missing_metrics else False
    
    # Si aucun entry avec la query n'a été trouvé, on considère comme non évalué
    return metrics

import os
import json

def save_metrics(fichier, query, context_precision=None, response_relevancy=None, faithfulness=None):
    # Charger le contenu du fichier s’il existe, sinon créer une liste vide
    if os.path.exists(fichier):
        with open(fichier, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Le fichier JSON doit contenir une liste.")
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # On cherche l'entrée existante avec la query donnée
    entry = next((e for e in data if e.get("query") == query), None)
    
    if entry is None:
        # Pas d'entrée trouvée, on en crée une nouvelle
        entry = {"query": query}
        data.append(entry)
    
    # Mise à jour uniquement des métriques non None
    if context_precision is not None:
        entry["context_precision"] = context_precision
    if response_relevancy is not None:
        entry["response_relevancy"] = response_relevancy
    if faithfulness is not None:
        entry["faithfulness"] = faithfulness
    
    # Sauvegarder le fichier
    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Métriques sauvegardées avec succès dans '{fichier}'.")


evaluator_llm = MistralLLMWrapper(api_key=API_KEY)
embeddings = MyAPIEmbedding("model")
my_run_config = RunConfig(timeout=6000)
from tqdm import tqdm
from pathlib import Path

def run_ragas_eval(output_path) :
    results_path = Path(output_path) / "responses.json"
    fichier_resultats = Path(output_path) / "metrics.json"


    
    evaluation_dataset = load_dataset(results_path)
    for entry in tqdm(evaluation_dataset):
        missing = not_done(fichier_resultats, entry['user_input'])
        if missing:
            sample = SingleTurnSample(
                user_input=entry['user_input'],
                retrieved_contexts=entry['retrieved_contexts'],
                response=entry['response']
            )
            
            context_precision = None
            response_relevancy = None
            faithfulness = None
            
            if "faithfulness" in missing:
                faithfulness = evaluate(
                    dataset=EvaluationDataset(samples=[sample]),
                    metrics=[Faithfulness()],
                    llm=evaluator_llm
                )
                print("Faithfulness :", faithfulness)

            if "context_precision" in missing:
                context_precision = evaluate(
                    dataset=EvaluationDataset(samples=[sample]),
                    metrics=[LLMContextPrecisionWithoutReference()],
                    llm=evaluator_llm,
                    run_config=my_run_config
                )
                print("Context Precision :", context_precision)

            if "response_relevancy" in missing:
                response_relevancy = evaluate(
                    dataset=EvaluationDataset(samples=[sample]),
                    metrics=[ResponseRelevancy()],
                    llm=evaluator_llm,
                    embeddings=embeddings
                )
                print("Response Relevancy :", response_relevancy)

            # Sauvegarder uniquement les métriques calculées (non None)
            save_metrics(
                fichier_resultats,
                entry['user_input'],
                context_precision=context_precision['llm_context_precision_without_reference'] if context_precision else None,
                response_relevancy=response_relevancy['answer_relevancy'] if response_relevancy else None,
                faithfulness=faithfulness['faithfulness'] if faithfulness else None
            )
        else:
            print(f"Question déjà traitée : {entry['user_input']}")

import sys
if __name__ == "__main__" :
    args = sys.argv[1:]
    output_path = args[0]
    run_ragas_eval(output_path)