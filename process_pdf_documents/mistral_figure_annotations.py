import base64
import requests
import os
import json
from pathlib import Path
import re
from tqdm import tqdm
import io 
from PIL import Image
import fitz

def load_figure_page_image(fig_num, base_path, pdf_path) :
    metadatas_path = os.path.join(base_path, "pictures", f"picture_metadata.json")
    with open(metadatas_path, 'r', encoding='utf-8') as f:
        figures_metadatas = json.load(f)
    figure_metadata = figures_metadatas[fig_num-1]
    page_no = figure_metadata["picture_page_no"]
    doc = fitz.open(pdf_path)
    page = doc[page_no-1]
    pix = page.get_pixmap(dpi=96)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def figure_annotation_with_mistral(chunks_content, base_path, pdf_path):
    """
    Pour chaque chunk, ajoute l'annotation obtenue via create_chat_completion
    juste sous le marqueur <!-- Figure number: ... -->, sous forme
    <!-- Figure caption : {caption} -->. Retourne une liste de chunks texte enrichis.

    Args:
        chunks_content (list of str): Liste de textes chunks avec refs comme <!-- Figure number: 20 -->
        base_path (str): Chemin vers dossier contenant 'pictures' avec images picture-<num>.png

    Returns:
        list of str: Liste des chunks enrichis en texte (avec annotations insérées).
    """

    pattern = r"<!--\s*Figure number:\s*(\d+)\s*-->"
    enriched_chunks = []
    irrelevant_figure_numbers = []
    for chunk_content in tqdm(chunks_content):
        # On va reconstruire le chunk, en insérant annotation après chaque figure
        new_chunk = ""
        last_pos = 0
        matches = list(re.finditer(pattern, chunk_content))

        for match in matches:
            figure_num = int(match.group(1))
            start, end = match.span()
            
            # Construire prompt pour annotation
            """prompt = (
                f"You are looking at a scientific document page that contains a figure."
                f"First, carefully inspect the provided images of the figure and of the full page to see "
                f"if there is a visible title which must start with Figure printed near it."
                f"Only if such a title is visible, transcribe it exactly as it appears."
                f"Then, describe briefly and factually the figure : its nature, content and keywords."
            )"""

            # Prompt 1 : pour classifier l'image
            prompt_1 = (
                "Classify the image with a single digit:\n"
                "0 - Relevant: contains technical content and is understandable by a language model.\n"
                "1 - Irrelevant: examples include logos, signatures, illustrative images, footers, or unrecognizable text.\n"
                "Please respond only with the digit 0 or 1."
            )
            parts_1 = [{
                "type": "text",
                "text": prompt_1
            }]

            # Prompt 2 : pour annoter l'image
            prompt_2 = (
                "Provide a brief, factual description of the image in 3-4 sentences. Be concise and accurate." 
                "Limit your description to the visual elements present in the image. Avoid any interpretations, assumptions, or additional context."   
         )
            parts_2 = [{
                "type": "text",
                "text": prompt_2
            }]

            # Charger image
            img_path = os.path.join(base_path, "pictures", f"picture-{figure_num}.png")
            if os.path.isfile(img_path):
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    b64_img = base64.b64encode(img_bytes).decode("utf-8")
                    # On insère l'image au bon format data URL ; on suppose png ici
                    parts_1.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_img}"
                        }
                    })

                   

                    """# Ajouter l'image de la page entière 
                    b64_page = load_figure_page_image(figure_num, base_path, pdf_path)
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_page}"
                        }
                    })"""
                    # Définir les messages envoyés à l'API
                    messages_1 = [
                        {
                            "role": "user",
                            "content":parts_1
                        }
                    ]
                    answer = create_chat_completion(messages_1)
                    if answer == '0' : 
                        parts_2.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}"
                            }
                        })
                        messages_2 = [
                            {
                                "role":"user",
                                "content":parts_2
                            }
                        ]
                        
                        # Appel à l'API
                        caption = None
                        while True :
                            caption = create_chat_completion(messages_2)
                            if caption : 
                                print(f"Description faite de la figure {figure_num} : {caption}")
                                annotation = f"\n<!-- Figure caption : {caption} -->"
                                new_chunk += chunk_content[last_pos:end]
                                new_chunk += annotation
                                break
                            else :
                                print("Erreur de l'API, nouvelle tentative")
                                                            
                    if answer == '1' : 
                        print(f"Image classifiée comme non pertinente par mistral")
                        irrelevant_figure_numbers.append(str(figure_num))
                        new_chunk += chunk_content[last_pos:start]
            else:
                print(f"Erreur : image {img_path} non trouvée")           

            last_pos = end

        # Ajouter le reste du chunk après la dernière figure
        new_chunk += chunk_content[last_pos:]
        enriched_chunks.append(new_chunk)
    if irrelevant_figure_numbers :
        irr_fig_numbers_str = ", ".join(irrelevant_figure_numbers)
        with open(Path(base_path) / 'irrelevant_figures.txt', 'w', encoding='utf-8') as irr_fig_f :
            irr_fig_f.write(f"Figures {irr_fig_numbers_str} non pertinentes \n")
    return enriched_chunks

def create_chat_completion(messages, model='mistral', temperature=0.7, max_tokens=256):
    pass

def annotate_database_figures(data_dir, pdf_dir):
    all_files = list(Path(data_dir).rglob('*'))
    chunks_files = [file for file in all_files if "\\chunks.json" in str(file)]

    for file in tqdm(chunks_files):
        annotated_file = file.with_name("annotated_chunks.json")
        if annotated_file.exists() :
            print(f"\nFichier {file} déjà annoté")
        else :
            print(f"\nAnnotations des images du fichier: {file}")
            # Charger la liste de chunks (list of dict) depuis JSON
            with open(file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Extraire les contenus des chunks (list des strings)
            chunks_content = [chunk['content'] for chunk in chunks]

            doc_dir_path = file.parent
            pdf_path = Path(pdf_dir) / file.parent.relative_to(data_dir).with_suffix('.pdf')
            # Postprocesser tous les chunks, récupère liste str enrichie
            enriched_contents = figure_annotation_with_mistral(chunks_content, doc_dir_path, pdf_path)
            
            # Remplacer dans la liste original chunks la clé 'content' par la nouvelle valeur enrichie
            for chunk, enriched_text in zip(chunks, enriched_contents):
                chunk['content'] = enriched_text

            # Écrire les chunks modifiés dans un fichier
            with open(annotated_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
                print(f"Fichier {annotated_file} enregistré")

if __name__=="__main__":
    import sys

    # Ajouter le répertoire parent au path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from my_paths import RAW_DATA_DIR, PARSED_DATA_DIR

    data_dir = PARSED_DATA_DIR # Dossier dans lequel se trouve les fichiers chunks.json
    pdf_dir = RAW_DATA_DIR # Dossier dans lequel se trouve les pdfs
    # DATA_DIR et PDF_DIR doivent avoir la même structure de dossiers
    annotate_database_figures(data_dir, pdf_dir)

