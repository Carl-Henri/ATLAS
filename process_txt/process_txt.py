import os
import json
from transformers import AutoTokenizer
# Ajouter le répertoire parent au path
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from my_paths import *

from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_PATH)

def token_count(text):
    return len(tokenizer(text)['input_ids'])

def get_leaf_directories(root):
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if filenames:
            leaf_dirs.append(dirpath)
    return leaf_dirs

from tqdm import tqdm

import xml.etree.ElementTree as ET
from utils import xml_to_dict

def get_file_chunks(root, extensions=['.txt', '.xml']):
    chunks = []
    leaf_dirs = get_leaf_directories(root)
    for leaf in tqdm(leaf_dirs):
        print(f"Traitement de {leaf}")
        rel_path = os.path.relpath(leaf, root)
        for fname in os.listdir(leaf):
            if any(fname.endswith(ext) for ext in extensions):
                file_path = os.path.join(leaf, fname)
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = f.read()
                    if fname.endswith('.xml'):
                        # conversion du xml en json pour être plus digeste pour le LLM
                        data = ET.fromstring(data)
                        data = {data.tag: xml_to_dict(data)}
                        data = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception as e:
                    data = f"(erreur de lecture) : {str(e)}"
                # Chaque fichier devient une entrée chunk à traiter
                doc_name = rel_path + '\\' + fname
                chunk_str = f"From {doc_name} :\n{data}"
                chunks.append((rel_path, fname, chunk_str))
    return chunks

def langchain_splitter(prefixed_str, main_content, max_tokens=2000, overlap=200):
    # Splits only the content, then adds prefix at each start
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=token_count
    )
    split_content = splitter.split_text(main_content)
    return [f"{prefixed_str}\n{piece}" for piece in split_content]

def process_and_save(root, output_json_path, extensions=['.txt', '.xml']):
    file_chunks = get_file_chunks(root, extensions=extensions)
    entries = []
    print("Traitement et découpage des chunks...")
    for chunk_hierarchy, fname, chunk_str in tqdm(file_chunks):
        doc_name = chunk_hierarchy + '\\' + fname
        if doc_name in entries : 
            print(f"Document : {doc_name} déjà traité.")
        if len(chunk_str) > 1000000 :
            print(f"Chunk super gros : {chunk_hierarchy + '\\' + fname}. Skip.")
            continue
        total_token_count = token_count(chunk_str)
        prefix = f"From {doc_name} :"
        content_only = chunk_str[len(prefix)+1:] if chunk_str.startswith(prefix) else chunk_str.split('\n',1)[-1]
        # Si chunk ≤8000 tokens, pas de split
        if total_token_count <= 8000:
            chunk_list = [chunk_str]
        else:
            chunk_list = langchain_splitter(prefix, content_only, max_tokens=2000, overlap=200)
        total_chunks = len(chunk_list)
        for idx, sub_chunk in enumerate(chunk_list):
            entry = {
                "content": sub_chunk,
                "metadata": {
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "chunk_token_count": token_count(sub_chunk),
                    "hierarchy": chunk_hierarchy,
                    "doc_name":doc_name,
                    "file": fname
                }
            }
            entries.append(entry)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
