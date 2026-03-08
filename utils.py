import os
import re 

def save_context_unique(conversation_folder, content):
    # Trouver un fichier context_N.txt libre dans conversation_folder
    file_index = 1
    while True:
        file_path = os.path.join(conversation_folder, f"context_{file_index}.txt")
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return file_path
        file_index += 1

def save_response_unique(conversation_folder, content):
    # Trouver un fichier context_N.txt libre dans conversation_folder
    file_index = 1
    while True:
        file_path = os.path.join(conversation_folder, f"response_{file_index}.txt")
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return file_path
        file_index += 1

def clean_folder_name(name):
    # supprimer les espaces en début/fin, et remplacer les espaces internes par un underscore
    # autoriser uniquement les caractères alphanumériques, tirets et underscores
    name = name.strip().replace(' ', '_')
    name = re.sub(r'[^A-Za-z0-9_\-]', '', name)
    name = re.sub(r'_+','_', name)
    if len(name) > 150 :
        name = name[:150]
    return name

from pathlib import Path
def fetch_documents_name(data_dir) :
    file_names_raw = [f.name.strip() for f in Path(data_dir).rglob("*") if f.suffix=='.xls' or f.suffix=='.xlsx' or f.suffix=='.pdf'  or f.suffix==".html"]
    return(file_names_raw) 


def fetch_folders(data_dir):
    folders = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            folders.append(entry)
    return folders

def get_filter(doc_choices, data_dir) :
    if doc_choices == [] :
        return([])
    filter = []
    for entry in doc_choices :
        if Path(entry).suffix == ".pdf" :
            filter.append(Path(entry).stem)
        elif Path(entry).suffix == ".xls" or Path(entry).suffix == '.xlsx' or Path(entry).suffix == '.html' : 
            filter.append(entry)
        else :
            doc_names = fetch_documents_name(Path(data_dir) / entry)
            filter.extend(get_filter(doc_names, data_dir))
    return(filter)