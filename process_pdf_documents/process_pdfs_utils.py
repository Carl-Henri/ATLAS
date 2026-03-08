import re
import fitz  # PyMuPDF
from PIL import Image
from docling_core.types.doc import PictureItem, DoclingDocument
from pdf_processing import IMG_MARGIN_TOP, IMG_MARGIN_BOTTOM, ROOT_KEYWORDS

def should_keep_image(image_item: PictureItem,
                      doc:DoclingDocument,
                      min_width: float = 50,
                      min_height: float = 50,
                      top_margin: float = IMG_MARGIN_TOP,
                      bottom_margin: float = IMG_MARGIN_BOTTOM) -> bool:
    """
    Détermine si une image doit être conservée.
    Fonctionne avec un PictureItem de Docling.
    """
    page_no = image_item.prov[0].page_no
    page_height = doc.pages[page_no].size.height
    # Taille réelle
    try:
        width = image_item.image.size.width
        height = image_item.image.size.height
    except AttributeError:
        return True  # pas d'info sur la taille → on garde

    if width < min_width and height < min_height:
        return False

    # Position bbox
    if not image_item.prov:
        return True  # pas d'info sur la position → on garde

    bbox = image_item.prov[0].bbox
    top = getattr(bbox, "t", page_height)
    bottom = getattr(bbox, "b", 0)

    if bottom > page_height - top_margin or top < bottom_margin:
        return False

    return True


def is_toc_table(table_item, doc) -> bool:
    md = table_item.export_to_markdown(doc)
    
    # heuristique : beaucoup de points de suite
    if re.search(r"\.{5,}", md):
        return True
    
    # heuristique : contient énormément de références "Figure" / "Table"
    matches = re.findall(r"(Figure|Table|Section)", md, flags=re.IGNORECASE)
    if len(matches) > 5:  # au moins 5 références
        return True
    
    # heuristique : seulement 2 colonnes dans data
    if table_item.data.num_cols == 2 and table_item.data.num_rows > 5:
        return True
    
    return False

import re


def is_root_title(title: str) -> bool:
    """Vérifie si le titre correspond à une 'racine' logique (Appendix, Section, etc.)."""
    title_low = title.lower()
    return any(re.search(rf"\b{re.escape(k)}\b",title_low) for k in ROOT_KEYWORDS)

def normalize_num(num: str) -> str:
    """Normalise les numéros pour que 5.0 == 5."""
    parts = num.split('.')
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts)

def parse_markdown_paths(md_text: str):
    """
    Construit un dict {titre: 'chemin complet du titre'}.
    - Ignore les roots dans les chemins.
    - Ne stacke pas les titres non numérotés entre eux.
    - Préserve la hiérarchie correcte entre niveaux.
    """
    heading_re = re.compile(r'^(##)\s*(.+)')
    num_re = re.compile(
        r'^\s*(?P<num>\(?[A-Za-z0-9]+(?:\.[0-9]+)*\)?)(?P<punc>[)\.\-\–\—\:]+)?\s*(?P<rest>.*)$'
    )

    headings = []
    num_index = {}
    current_stack = []
    last_numbered_stack = []

    # --- Étape 1 : extraction structurée des titres
    for line in md_text.splitlines():
        m = heading_re.match(line)
        if not m:
            continue

        text = m.group(2).strip()
        nm = num_re.match(text)

        if nm:
            raw_num = nm.group('num').strip("()")
            rest = nm.group('rest').strip()
            if re.match(r'^\d', raw_num):
                num = normalize_num(raw_num)
            else:
                num = raw_num.lower()
            title = text
        else:
            num = None
            title = text

        headings.append({
            "title": title,
            "num": num,
            "is_root": is_root_title(title)
        })

        if num and not is_root_title(title):
            if num in num_index :
                del num_index[num]
            else : 
                num_index[num] = title

    # --- Étape 2 : construction des chemins
    paths = {}
    path_parts = []
    for h in headings:
        title = h["title"]

        if h["is_root"]:
            # On ne garde pas les roots dans les chemins
            current_stack = []
            last_numbered_stack = []
            paths[title] = title
            continue

        num = h["num"]

        if num:
            # Cas hiérarchique numérique (8.4.1.2)
            if re.match(r'^\d+(\.\d+)*$', num):
                parts = num.split('.')
                path_parts = []
                for i in range(1, len(parts) + 1):
                    sub_num = ".".join(parts[:i])
                    if sub_num in num_index:
                        path_parts.append(num_index[sub_num])
                current_stack = path_parts
                last_numbered_stack = current_stack.copy()

            # Cas alphabétique (a, b, etc.)
            elif re.match(r'^[a-z]$', num):
                if last_numbered_stack:
                    path_parts = last_numbered_stack + [title]
                else:
                    path_parts = [title]
                current_stack = path_parts

            else:
                # Cas sans numéro ni root
                if last_numbered_stack:
                    # Hérite uniquement du dernier chemin numéroté
                    path_parts = last_numbered_stack + [title]
                else:
                    path_parts = [title]
                # Ne met PAS à jour last_numbered_stack ici !
                #print(path_parts)
                current_stack = path_parts

        # fallback
        if not path_parts:
            path_parts = [title]

        paths[title] = " > ".join(path_parts)

    return paths

def extract_text_from_bbox(pdf_path, page_no, bbox, coord_origin="BOTTOMLEFT"):
    """
    Extrait le texte d'une zone spécifique d'une page PDF.

    Paramètres
    ----------
    pdf_path : str
        Chemin vers le fichier PDF.
    page_no : int
        Numéro de page (1-indexé, c’est-à-dire que 1 = première page).
    bbox : dict
        Dictionnaire avec les clés 'l', 't', 'r', 'b' représentant les coordonnées du rectangle.
    coord_origin : str, optionnel
        "BOTTOMLEFT" si les coordonnées sont exprimées depuis le coin bas-gauche,
        "TOPLEFT" si elles le sont depuis le coin haut-gauche (par défaut : "BOTTOMLEFT").

    Retourne
    --------
    str
        Le texte extrait de la zone spécifiée.
    """
    # Ouvrir le document
    with fitz.open(pdf_path) as doc:
        page = doc[page_no - 1]  # PyMuPDF est 0-indexé

        # Ajuster les coordonnées verticales selon l’origine
        page_height = page.rect.height
        if coord_origin.upper() == "BOTTOMLEFT":
            top = page_height - bbox["t"]
            bottom = page_height - bbox["b"]
        else:
            top = bbox["t"]
            bottom = bbox["b"]

        # Créer le rectangle de clipping
        rect = fitz.Rect(bbox["l"], top, bbox["r"], bottom)

        # Extraire le texte
        text = page.get_text("text", clip=rect)

    return text.strip()

def save_img_from_bbox(pdf_path, page_no, bbox, output_path, num, coord_origin="BOTTOMLEFT"):
    """
    Extrait le texte d'une zone spécifique d'une page PDF.

    Paramètres
    ----------
    pdf_path : str
        Chemin vers le fichier PDF.
    page_no : int
        Numéro de page (1-indexé, c’est-à-dire que 1 = première page).
    bbox : dict
        Dictionnaire avec les clés 'l', 't', 'r', 'b' représentant les coordonnées du rectangle.
    coord_origin : str, optionnel
        "BOTTOMLEFT" si les coordonnées sont exprimées depuis le coin bas-gauche,
        "TOPLEFT" si elles le sont depuis le coin haut-gauche (par défaut : "BOTTOMLEFT").

    Retourne
    --------
    str
        Le texte extrait de la zone spécifiée.
    """
    # Ouvrir le document
    with fitz.open(pdf_path) as doc:
        page = doc[page_no - 1]  # PyMuPDF est 0-indexé

        # Ajuster les coordonnées verticales selon l’origine
        page_height = page.rect.height
        if coord_origin.upper() == "BOTTOMLEFT":
            top = page_height - bbox["t"]
            bottom = page_height - bbox["b"]
        else:
            top = bbox["t"]
            bottom = bbox["b"]

        # Créer le rectangle de clipping
        rect = fitz.Rect(bbox["l"], top, bbox["r"], bottom)

        # Extraire le texte
        pix = page.get_pixmap(clip=rect, dpi=144)
        if not(output_path.exists()) : 
            output_path.mkdir(parents=True, exist_ok=True)
        pix.save(output_path / f"formula-{num}.png")

import fitz
import re
import torch
import base64
import io
from PIL import Image


# --- Helper: encode PIL image in base64 ---
def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Render full page as PIL image ---
def get_page_image(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=80)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def extract_object_titles(pdf_path, page_number, object_type="figure", max_block_length=100, min_length=15):
    """
    Extract candidate titles from a PDF page for either 'figure' or 'table'.

    Args:
        pdf_path: path to PDF
        page_number: int
        object_type: "figure" or "table"
        max_block_length: maximum number of characters in a block to consider

    Returns:
        list of strings: candidate titles
    """
    import fitz, re

    if object_type.lower() not in ["figure", "table"]:
        raise ValueError("object_type must be 'figure' or 'table'")

    # Regex patterns, must start at beginning or after newline/space
    if object_type.lower() == "figure":
        pattern = r'^\s*(?:fig(?:ure)?\.?)\s*\d[\d\-\.:]*'
    else:
        pattern = r'^\s*table\s*\d[\d\-\.:]*'

    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    blocks = page.get_text("blocks")

    candidates = []
    for block in blocks:
        text = block[4].strip()
        if not text:
            continue
        if len(text) > max_block_length:
            continue  # ignore very long blocks
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            if len(text) > min_length :
                candidates.append(text)
    return candidates

def get_object_caption(pdf_path, page_number, object_type="figure", debug=False):
    candidates = extract_object_titles(pdf_path, page_number, object_type=object_type)

    if debug:
        print(f"[DEBUG] Found {len(candidates)} candidate {object_type} titles on page {page_number}")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c}")
    
    # Case 1 — single candidate
    if len(candidates) == 1:
        if debug:
            print(f"[DEBUG] Single candidate selected: {candidates[0]}")
        return candidates[0]

    else :
        return(None)

