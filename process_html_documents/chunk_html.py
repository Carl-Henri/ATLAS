import html2text
from transformers import AutoTokenizer
import json
import os
import sys
from pathlib import Path
# Ajouter le répertoire parent au path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from my_paths import BGE_M3_TOKENIZER_PATH

tokenizer = AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_PATH)

def get_every_html(data_dir) :
    html_files = [f for f in Path(data_dir).rglob("*") if f.suffix=='.html']
    return(html_files)

def open_test_case_as_md(file_path) :
    with open(file_path, 'r', encoding='utf-8') as html_f :
        content = html_f.read()
    res = html2text.html2text(content)
    return(res)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ('####', "Header 4")
]

def token_count(chunk) :
    return(len(tokenizer(chunk)['input_ids']))

def save_chunks_as_txt(json_path) :
    with open(json_path, 'r', encoding="utf-8") as f :
        chunks = json.load(f)
    save_chunks_as_txt(chunks)
    with open('chunks.txt', 'w', encoding="utf-8") as f :
        for i, chunk in enumerate(chunks) :
            f.write(f"Chunk number {i} : \n" + f"Chunk metadata {chunk['metadata']}\n" + chunk['content'] +"\n\n")

def chunk_html_as_md(file_path) :
    markdown_document = open_test_case_as_md(file_path)
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=True)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    # Char-level splits
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size = 2000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=token_count
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    chunks = []
    for i, doc in enumerate(splits) :
        doc_name = Path(file_path).name
        headers = ''
        for header in doc.metadata :
            if header == 'Header 1' :
                headers += "# "+doc.metadata[header]+"\n"
            elif header == 'Header 2' :
                headers += "## "+doc.metadata[header]+"\n"
            elif header == 'Header 3' :
                headers += "### "+doc.metadata[header]+"\n"
            elif header == 'Header 4' :
                headers += "#### "+doc.metadata[header]+"\n"
        content = f"From {doc_name} (chunk {i+1}/{len(splits)}):\n" + headers + doc.page_content

        chunk = {
            "content": content,
            "metadata": {
                "doc_name": doc_name,
                "hierarchy":", ".join([header + " : " + header_name for header, header_name in (doc.metadata).items()]),
                "chunk_index": i+1,
                "total_chunks": len(splits),
                "token_count": token_count(content)
            }
       }
        chunks.append(chunk)
    return(chunks)

def html_already_processed(chunks, doc_name) :
    for chunk in chunks :
        if chunk['metadata']['doc_name'] == doc_name :
            return(True)
    return(False)


from tqdm import tqdm

def chunk_every_html(data_dir, output_path) :
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
    html_files = get_every_html(data_dir)
    for file in tqdm(html_files) :
        json_path = output_path / str(Path(file).name).replace('.html','.json')
        if not os.path.isfile(json_path):
            chunks = chunk_html_as_md(file)
            with open(json_path, 'w', encoding="utf-8") as f :
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        else : 
            print(f'HTML déjà traité : {file}\n Dans : {json_path}')
            
    

if __name__ == "__main__":
    from my_paths import RAW_DATA_DIR_B, PARSED_DATA_DIR_B, CHROMA_DIR_B, WHOOSH_DIR_B
    chunk_every_html(RAW_DATA_DIR_B, PARSED_DATA_DIR_B / 'html_chunks')
    
    from hybrid_search import fill_databases_with_chunks_from_json
    html_chunks_dir = PARSED_DATA_DIR_B / 'html_chunks'
    files_to_process = html_chunks_dir.glob("*")
    fill_databases_with_chunks_from_json('./chroma_html_test', './whoosh_html_test', json_files=files_to_process)

