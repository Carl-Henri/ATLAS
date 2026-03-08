from pdf_processing import process_pdf
from tqdm import tqdm
import json
import shutil
from pathlib import Path
import fitz
import os

def repair_pdf_force_mediabox(pdf_path):
    doc = fitz.open(pdf_path)
    default_mediabox = fitz.Rect(0, 0, 595, 842)  # A4

    to_fix = []
    for page in doc:
        mediabox = doc.xref_get_key(page.xref, "MediaBox")
        print(mediabox)
        if not mediabox or mediabox == ('null','null'):
            print(f"Page {page.number+1} : pas de MediaBox -> correction forcée.")
            page.set_mediabox(default_mediabox)
            to_fix.append(page.number)
        # On met aussi un CropBox au cas où (optionnel)
        cropbox = doc.xref_get_key(page.xref, "CropBox")
        if not cropbox:
            page.set_cropbox(default_mediabox)

    if to_fix:
        temp_path = pdf_path + '.temp'
        doc.save(temp_path, garbage=3)  # "garbage=3" ré-écrit tout de zéro !
        doc.close()
        os.replace(temp_path, pdf_path)
        print(f"{pdf_path} réparé ! (pages corrigées: {to_fix})")
    else:
        doc.close()
        print(f"Aucune page à corriger dans {pdf_path}.")

def process_raw_database_pdfs(database_path, output_path):
    unprocessed_files = []
    
    all_files = list(database_path.rglob("*"))
    files = []
    for file in all_files : 
        ext = file.suffix.lower() 
        if ext == ".pdf" :
            files.append(file)
    print(f'Parsing avec docling des fichiers pdfs : {files}')

    stats_path = output_path / "stats.json"

    # Charger les stats existantes si le fichier existe
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
    else:
        stats_dict = {}
    
    for file in tqdm(files):

        ext = file.suffix.lower()
        if ext == ".pdf":
            output_dir = output_path / file.relative_to(database_path)
            output_dir = Path(str(output_dir.with_suffix('')).strip())
            if output_dir.is_dir() and (output_dir / "chunks.json").is_file():
                print("Document déjà parsé avec docling")
                continue
            else:
                print(f"Parsing avec docling du document {file}")
                if output_dir.is_dir():
                    # On supprime le dossier pour repartir d'une base propre si le parsing n'était pas arrivé jusqu'à chunks.json
                    shutil.rmtree(output_dir)  # Supprime le dossier et tout son contenu
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    stats = process_pdf(file, output_dir)  # retourne un dictionnaire
                    # Ajouter ou mettre à jour les stats pour ce fichier
                    stats_dict[str(file)] = stats
                    # Sauvegarder immédiatement
                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    try :
                        # Corriger l'erreur potentielle sur les dimensions des pages
                        repair_pdf_force_mediabox(str(file))
                        
                        stats = process_pdf(file, output_dir)  # retourne un dictionnaire
                        # Ajouter ou mettre à jour les stats pour ce fichier
                        stats_dict[str(file)] = stats
                        # Sauvegarder immédiatement
                        with open(stats_path, "w", encoding="utf-8") as f:
                            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
                    except Exception as e2:
                        print(f"Erreurs sur le document {file}: {e} puis {e2}")
                        unprocessed_files.append(file)
    return unprocessed_files

if __name__=="__main__":
    import sys
    import os

    # Ajouter le répertoire parent au path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from my_paths import RAW_DATA_DIR, PARSED_DATA_DIR

    database_path = RAW_DATA_DIR
    output_path = PARSED_DATA_DIR

    unprocessed_files = process_raw_database_pdfs(database_path=database_path, output_path=output_path)
    