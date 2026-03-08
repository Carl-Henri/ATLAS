import os 
import json
import numpy as np

def load_json(file) :
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Le fichier JSON doit contenir une liste.")
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    return(data)

from mistral_common.tokens.tokenizers.tekken  import Tekkenizer

tokenizer =  Tekkenizer.from_file("")
def get_datas(fichier):
    metrics_file = fichier / 'metrics.json'
    responses_file = fichier / 'responses_only.json'
    data = load_json(metrics_file)
    data_responses = load_json(responses_file)
    for i in range(len(data)) :
        if data[i]['query'] == data_responses[i]['query'] : 
            #data[i]['len_response'] = len(data_responses[i]['response'])
            data[i]['token_response'] = len(tokenizer.encode(data_responses[i]['response'],bos=True,eos=True))
        else :
            print("Erreur lors de l'ajout de la longueur de la réponse")

    return(data)

def mean_metrics_on_common_subset(list_of_data):

    n_items = len(list_of_data[0])  # on suppose que tous les datasets ont la même longueur

    # Vérification que tous les datasets ont la même longueur
    for data in list_of_data:
        assert len(data) == n_items, "Tous les datasets doivent avoir la même longueur"

    common_valid_indices = []

    for i in range(n_items):
       
        # Vérifier que les métriques ne sont pas NaN dans aucun dataset
        valid = True
        for data in list_of_data:
            cp = data[i]['context_precision'][0]
            ar = data[i]['response_relevancy'][0]
            f = data[i]['faithfulness'][0]
            if np.isnan(cp) or np.isnan(ar) or np.isnan(f):
                valid = False
                break
        if valid:
            common_valid_indices.append(i)

    print(f"Nombre total de questions avec la même query à l'indice i dans tous les datasets : {len(common_valid_indices)}")

    means_per_dataset = []

    for data in list_of_data:
        l_cp = []
        l_ar = []
        l_f = []
        l_len = []
        for i in common_valid_indices:
            item = data[i]
            l_cp.append(item['context_precision'][0])
            l_ar.append(item['response_relevancy'][0])
            l_f.append(item['faithfulness'][0])
            l_len.append(item['token_response'])
        mean_cp = np.mean(l_cp) if l_cp else float('nan')
        mean_ar = np.mean(l_ar) if l_ar else float('nan')
        mean_f = np.mean(l_f) if l_f else float('nan')
        mean_len = np.mean(l_len) if l_len else float('nan')
        means_per_dataset.append((mean_cp, mean_ar, mean_f, mean_len))

    # On peut aussi retourner les queries communes extraites
    common_queries = [(list_of_data[0][i]['query'],list_of_data[1][i]['query']) for i in common_valid_indices]

    return means_per_dataset, common_queries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def afficher_tableau_metrics(liste_metrics, labels, total_evaluated):
    """
    liste_metrics : liste de tuples, chaque tuple = (mean_precision, mean_relevancy, mean_faithfulness, nan_cp, nan_ar, nan_f)
    labels : liste de chaînes, optionnel, identifiants (ex : noms des fichiers) correspondant à chaque tuple
    
    Affiche un tableau des métriques sous forme d'image.
    """
    # Colonnes du tableau
    colonnes = ['Mean Context Precision', 'Mean Answer Relevancy', 'Mean Faithfulness', 'Mean Tokens']
    
    # Création d'un DataFrame pandas
    df = pd.DataFrame(liste_metrics, columns=colonnes)
    
    if labels is not None and len(labels) == len(liste_metrics):
        df.index = labels
    else:
        df.index = [f'Fichier {i+1}' for i in range(len(liste_metrics))]
    
    # Affichage du tableau en image avec matplotlib
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))  # taille dynamique selon nombre de lignes
    ax.axis('off')
    
    # Création du tableau matplotlib
    tab = ax.table(cellText=np.round(df.values, 4), 
                   colLabels=df.columns, 
                   rowLabels=df.index,
                   cellLoc='center', 
                   loc='center',
                   colColours=["#f2f2f2"]*len(colonnes))
    
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.5)  # échelle cellule
    
    plt.title(f'Tableau des métriques par fichier sur {total_evaluated} questions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

from pathlib import Path

def main() :
    from pipeline_eval import tests_finaux 
    # mode de retrieval (str), rerank (bool), nombre max de chunks récupérés (int)
    tests_evalues = []
    for test in tests_finaux :
        benchmark = Path(test['benchmark_path']).stem
        #benchmark == 'benchmark_anglais_1'
        if 'alpha' in test and test['alpha'] in [0.5, 0.75] and test['retrieval'] == "hybrid" and benchmark == 'benchmark_anglais_1': 
            tests_evalues.append(test)
        if 'alpha' in test and test['retrieval'] == "chroma" and benchmark == 'benchmark_anglais_1': 
            tests_evalues.append(test)
        #tests_evalues.append(test)
    liste_labels = []

    liste_datas = []
    for test in tests_evalues :
            benchmark = Path(test['benchmark_path']).stem
            liste_datas.append(get_datas(test['output_dir']))

            liste_labels.append(str(test['output_dir'].name) + '_' + benchmark)
    
    liste_results, subset = mean_metrics_on_common_subset(liste_datas)
    print(subset)
    afficher_tableau_metrics(liste_results, liste_labels, len(subset))

if __name__ == "__main__" : 
    main()