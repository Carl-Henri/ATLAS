
# DEFINIR LES MARGES DE PRISE EN COMPTE DU TEXTE ET DES IMAGES

# Ne prend pas en compte le texte à TEXT_MARGIN_BOTTOM pixels du bas de la page
TEXT_MARGIN_BOTTOM = 70
# Ne prend pas en compte le texte à TEXT_MARGIN_TOP pixels du haut de la page
TEXT_MARGIN_TOP = 70

# Ne prend pas en compte l'image dont le bottom est à IMG_MARGIN_TOP pixels du haut de la page
IMG_MARGIN_TOP = 110
# Ne prend pas en compte l'image dont le top est à IMG_MARGIN_BOTTOM pixels du bas de la page
IMG_MARGIN_BOTTOM = 110

# Texte à ne pas prendre en compte
USELESS_TEXT_LIST = ['']

# Texte pris en compte comme des titres racine (i.e indiquant une nouvelle section, sans numérotation)
ROOT_KEYWORDS = [
    "appendix", "annex", "section", "part", "chapter",
    "summary", "foreword", "introduction",
    "table of contents", "table of figures", "table of tables",
    "list of figures", "list of tables", "signatures",
    "modifications description", "abstract"
]

import re

# FONCTION A ADAPTER POUR DETECTER LES EXIGENCES
def est_titre_req(s):
    # Recherche -REQ- (insensible à la casse)
    if re.search(r"-req-", s, re.IGNORECASE):
        return True

    # Recherche Req avec lettre avant ou après
    # Explication du regex :
    #   [a-zA-Z]Req  -> lettre avant (ex: "xReq")
    #   Req[a-zA-Z]  -> lettre après  (ex: "Reqx")
    # Le | fait l'alternance
    pattern = re.compile(r"[a-zA-Z]Req[a-zA-Z]", re.IGNORECASE)
    if pattern.search(s):
        return True

    return False

# FONCTION A ADAPTER POUR FILTER LES TITRES DETECTES
def is_structural_heading(title: str) -> bool:
    if not title or not isinstance(title, str):
        return False

    t = title.strip()
    lower_t = t.lower()

    # Si le titre contient un numéro hiérarchique ET un mot spécifique qu'on veut exclure
    parts = t.split()
    # Liste des mots "exclus" à considérer comme non titres s'ils suivent la numérotation

    # On ne veut pas que les descriptions de conditions etc. viennent troubler les titres
    excluded_following_words = {"else", "if", "then", "endif", "elif"}

    # pattern
    num_pattern = re.compile(
        r"""(?ix)
        ^                          # Début de ligne
        (?:                        # Groupe non capturant pour le prefixe du titre
            [a-z]|\d+              # Lettre ou chiffre arabe
            (?:[\.\-–—]\d+)*       # Suit . ou -
            |                      # OU
            (?:[ivxlc]+)           # Chiffre romain en minuscules ou majuscules
        )
        [\.\)]?                    # . ou ) optionnel
        [\s\:]                     # espace ou :
        """,
    )

    if len(parts) >= 2:
        num_part, word_part = parts[0], parts[1]
        if word_part.lower().rstrip(':') in excluded_following_words:
            # Vérifier que num_part est bien une numérotation valide
            if num_pattern.match(num_part):
                return False

    if any(kw in lower_t for kw in ROOT_KEYWORDS):
        return True
    
    if num_pattern.match(t):
        return True

    if est_titre_req(t):
        return True

    if re.fullmatch(r'[A-Z]$', t):
        return True

    return False

import json
import logging
import time
from time import perf_counter as timer
from pathlib import Path
from tqdm import tqdm
import os 
from serializers import FilteredTextSerializer, ReferencedPictureSerializer, ReferencedTableSerializer
from process_pdfs_utils import parse_markdown_paths, ROOT_KEYWORDS

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from docling.chunking import HybridChunker

from docling_core.types.doc import DoclingDocument

from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider, ChunkingDocSerializer

_log = logging.getLogger(__name__)

import sys

# Ajouter le répertoire parent au path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from my_paths import BGE_M3_TOKENIZER_PATH, DOCLING_ARTIFACTS_PATH

from docling_core.types.doc import TextItem, SectionHeaderItem, DocItemLabel

def del_some_useless_texts(text) : 
    res = text
    for useless_text in USELESS_TEXT_LIST :
        res = res.replace(useless_text,'')
    return res

def doc_post_processing(doc, output_dir):
    margin_bottom = TEXT_MARGIN_BOTTOM
    margin_top = TEXT_MARGIN_TOP
    # on nettoie certains titres qui ne devraient pas l'être
    texte_hors_marges = []
    titre_transforme_en_texte = []
    for idx, item in tqdm(enumerate(doc.texts)):
        if hasattr(item,'text') :
            item.text = del_some_useless_texts(item.text)

        if isinstance(item, SectionHeaderItem) and not(is_structural_heading(item.text)) :
            # Certains textes sont détectés comme des titres alors qu'ils ne devraient pas
            titre_transforme_en_texte.append(item.text)

            # On supprime le texte s'il est hors marges (trop proche du haut ou du bas de la page)
            bbox = item.prov[0].bbox 
            page_no = item.prov[0].page_no
            page_height = doc.pages[page_no].size.height
            top, bottom, coord_origin = getattr(bbox, "b"), getattr(bbox, "t"), getattr(bbox,'coord_origin')
            if coord_origin == "BOTTOMLEFT" :
                if bottom <= margin_bottom or top >= page_height-margin_top :
                    texte_hors_marges.append(item.text)
                    item.text = ''
            elif coord_origin == "TOPLEFT" :
                if bottom <= page_height-margin_bottom or top >= margin_top :
                    texte_hors_marges.append(item.text)
                    item.text = ''

            # On recrée un item Texte pour remplacer l'item SectionHeaderItem 
            new_item = TextItem(
                text=item.text,
                orig=getattr(item, "orig", item.text),
                prov=getattr(item, "prov", []),
                children=getattr(item, "children", []),
                parent=getattr(item, "parent", None),
                content_layer=getattr(item, "content_layer", "body"),
                self_ref=getattr(item, "self_ref", None),
                label="text",
            )
            doc.texts[idx] = new_item
        
        elif isinstance(item, TextItem) :
            # On supprime le texte s'il est hors marges (trop proche du haut ou du bas de la page)
            bbox = item.prov[0].bbox 
            page_no = item.prov[0].page_no
            page_height = doc.pages[page_no].size.height
            top, bottom, coord_origin = getattr(bbox, "b"), getattr(bbox, "t"), getattr(bbox,'coord_origin')
            if coord_origin == "BOTTOMLEFT" :
                if bottom <= margin_bottom or top >= page_height-margin_top :
                    texte_hors_marges.append(item.text)
                    item.text = ''
            elif coord_origin == "TOPLEFT" :
                if bottom <= page_height-margin_bottom or top >= margin_top :
                    texte_hors_marges.append(item.text)
                    item.text = ''
        
        with open(output_dir / "doc_post_processing_log.txt","w",encoding="utf-8") as log_file :
            txt_h_m = '\n'.join(texte_hors_marges)
            ttt = '\n'.join(titre_transforme_en_texte)
            log_file.write(f"Textes supprimés car hors marges : \n{txt_h_m}\n")
            log_file.write(f"Titres transformés en texte : \n{ttt}\n")

def chunks_post_processing(chunks, hierarchy, doc_name, tokenizer, max_tokens):
    i = 0
    data = []
    chunk_no = 0
    previous_saved_table_start = ""
    saved_table_start = ""

    # Patterns table
    pattern_start = r'<!--\s*Table\s+markdown\s+(\d+)\s+start\s*-->'
    pattern_end = r'<!--\s*Table\s+markdown\s+(\d+)\s+end\s*-->'

    while i < len(chunks):
        # Récupération du texte actuel (avec éventuellement un morceau de table sauvegardé du chunk précédent)
        current_text = previous_saved_table_start + chunks[i].text
        chunks_to_merge = [chunks[i]]

        # On cherche s’il y a une table qui commence
        found_starts = list(re.finditer(pattern_start, current_text))

        # --- Cas 1 : pas de table dans ce chunk ---
        if not found_starts:
            chunk_no += 1
            content = current_text
            tokens = tokenizer(content)
            token_count = len(tokens['input_ids'])
            page_no = list(set(
                [item.prov[0].page_no for ch in chunks_to_merge for item in ch.meta.doc_items]
            ))
            if len(page_no) == 1:
                page_no_str = f"From page {page_no[0]} in "
            else:
                page_no_str = f"From pages {', '.join(map(str, page_no))} in "
            try : 
                chunk_hierarchy = hierarchy[chunks_to_merge[0].meta.headings[0]]                   
            except :
                try : 
                    chunk_hierarchy = hierarchy[chunks_to_merge[0].meta.headings[0].replace('_','\\_')].replace('\\_','_')
                except : 
                    chunk_hierarchy = ""
            
            try : 
                chunk_heading = chunks_to_merge[0].meta.headings[0]            
            except :
                chunk_heading = ""
            entry = {
                "content": page_no_str + doc_name + "\n" + "Chunk hierarchy : " + chunk_hierarchy + "\n" + chunk_heading + "\n" + content,
                "metadata": {
                    'chunk_no': chunk_no,
                    'chunk_token_count': token_count,
                    'page_no': page_no,
                    'hierarchy': chunk_hierarchy,
                    'doc_name': doc_name,
                }
            }
            data.append(entry)
            i += 1
            continue

        # --- Cas 2 : une ou plusieurs tables commencent dans ce chunk ---

        # Fusionner tant qu'on n’a pas trouvé la fin de la dernière table
        while True:
            all_starts = re.findall(pattern_start, current_text)
            all_ends = re.findall(pattern_end, current_text)

            if len(all_starts) > len(all_ends):
                # Table(s) non terminée(s) → continuer à fusionner
                if i + 1 < len(chunks):
                    next_text = current_text + chunks[i + 1].text
                    tokens = tokenizer(next_text)
                    token_count = len(tokens['input_ids'])

                    if token_count > max_tokens:
                        # Trop de tokens, on coupe avant la dernière table incomplète

                        # On garde le texte jusqu'à avant la dernière table incomplète
                        last_start = list(re.finditer(pattern_start, current_text))[-1]
                        saved_table_start = current_text[last_start.start():]
                        current_text = current_text[:last_start.start()]
                        break
                    else:
                        # On fusionne le chunk suivant
                        i += 1
                        chunks_to_merge.append(chunks[i])
                        current_text = next_text
                else:
                    # Dernier chunk et table non terminée — on garde la fin pour plus tard
                    break
            else:
                # Toutes les tables ont une fin
                break

        # Enregistrement du chunk fusionné et/ou croppé 
        if current_text :
            chunk_no += 1
            tokens = tokenizer(current_text)
            token_count = len(tokens['input_ids'])

            page_no = list(set(
                [item.prov[0].page_no for ch in chunks_to_merge for item in ch.meta.doc_items]
            ))
            if len(page_no) == 1:
                page_no_str = f"From page {page_no[0]} in "
            else:
                page_no_str = f"From pages {', '.join(map(str, page_no))} in "
        
            try : 
                chunk_hierarchy = hierarchy[chunks_to_merge[0].meta.headings[0]]                   
            except :
                try : 
                    chunk_hierarchy = hierarchy[chunks_to_merge[0].meta.headings[0].replace('_','\\_')].replace('\\_','_')
                except : 
                    chunk_hierarchy = ""
            try : 
                chunk_heading = chunks_to_merge[0].meta.headings[0]            
            except :
                chunk_heading = ""
            entry = {
                "content": page_no_str + doc_name + "\n" + "Chunk hierarchy : " + chunk_hierarchy + "\n" + chunk_heading + "\n" + current_text,
                "metadata": {
                    'chunk_no': chunk_no,
                    'chunk_token_count': token_count,
                    'page_no': page_no,
                    'hierarchy': chunk_hierarchy,
                    'doc_name': doc_name,
                }
            }
            data.append(entry)

        # On sauvegarde la partie non terminée (s’il y en a une)
        previous_saved_table_start = saved_table_start
        saved_table_start = ""
        i += 1

    return data

IMAGE_RESOLUTION_SCALE = 2.0

from docling.datamodel.base_models import InputFormat  
from docling.datamodel.pipeline_options import PdfPipelineOptions  
from docling.document_converter import DocumentConverter, PdfFormatOption  
from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  
  
from io import BytesIO  
import base64  
from docling.models.base_model import BaseItemAndImageEnrichmentModel  
from docling.datamodel.base_models import ItemAndImageEnrichmentElement  
from docling_core.types.doc import DoclingDocument, NodeItem, FormulaItem

from mistral_langchain_wrapper import MistralChatWrapper
import os 
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

llm = MistralChatWrapper(api_key=API_KEY, model="medium")
  
class ApiCodeFormulaModel(BaseItemAndImageEnrichmentModel):  
    elements_batch_size = 1  
    images_scale = 2.0
  
    def __init__(self, enabled: bool):  
        # Initialize required attributes  
        self.enabled = enabled  
        # Don't call super().__init__(**kwargs) - parent doesn't accept kwargs  
  
    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:  
        return self.enabled and isinstance(element, FormulaItem)  
  
    def __call__(  
        self,  
        doc: DoclingDocument,  
        element_batch: list[ItemAndImageEnrichmentElement],  
    ) -> list[NodeItem]:  
        results = []  
        for el in element_batch:  
            # el.image is the cropped PIL.Image; el.item is the original TextItem  
            buffered = BytesIO()  
            el.image.save(buffered, format="PNG")  
            img_b64 = base64.b64encode(buffered.getvalue()).decode()  
  
            messages = [  
                {  
                    "role": "user",  
                    "content": [  
                        {"type": "text", "text": "Extract LaTeX from this image. ONLY output the LaTex, without comments, without code tags."},  
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},  
                    ],  
                }  
            ]   
            response = llm.invoke(messages).content.replace('```','').replace('```latex','')
  
            # Update the original TextItem's text  
            el.item.text = response  
            results.append(el.item)  
        return results
    
# Version du pipeline en utilisant mistral medium pour les formules latex
class MyStandardPdfPipelineWithAPIFormulaEnrichment(StandardPdfPipeline):  
    def _init_models(self):  
        super()._init_models()  
        # Replace CodeFormulaModel with API version while keeping other enrichments  
        self.enrichment_pipe = [  
            ApiCodeFormulaModel(  
                enabled=self.pipeline_options.do_formula_enrichment,  
            ),  
            *self.enrichment_pipe[1:],  # Keep other enrichment models  
        ]  


# Version du pipeline en réduisant la batch size pour CodeFormulaV2 en local (pour éviter de déborder de la VRAM)
class MyCodeFormulaModel(CodeFormulaModel):  
    elements_batch_size = 1  
  
class MyStandardPdfPipeline(StandardPdfPipeline):  
    def _init_models(self) -> None:  
        # Copy parent initialization but replace CodeFormulaModel  
        super()._init_models()  
        # Replace the CodeFormulaModel entry in enrichment_pipe  
        self.enrichment_pipe = [  
            MyCodeFormulaModel(  
                enabled=self.pipeline_options.do_code_enrichment  
                or self.pipeline_options.do_formula_enrichment,  
                artifacts_path=self.artifacts_path,  
                options=CodeFormulaModelOptions(  
                    do_code_enrichment=self.pipeline_options.do_code_enrichment,  
                    do_formula_enrichment=self.pipeline_options.do_formula_enrichment,  
                ),  
                accelerator_options=self.pipeline_options.accelerator_options,  
            ),  
            # Preserve any other enrichment models added by the parent  
            *self.enrichment_pipe[1:],  
        ]  

from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions  
  
# Version du pipeline en loadant les modèles séparément en fonctions des phases (NON TESTÉ)
class LazyEnrichmentPipeline(StandardPdfPipeline):  
    def __init__(self, pipeline_options: ThreadedPdfPipelineOptions) -> None:  
        self._enrichment_models_initialized = False  
        # Initialize without enrichment models  
        super().__init__(pipeline_options)  
        # Clear enrichment pipe to prevent loading  
        self.enrichment_pipe = []  
      
    def _init_models(self) -> None:  
        # Load only core models (OCR, layout, table, etc.)  
        super()._init_models()  
        # Skip enrichment model initialization  
        self.enrichment_pipe = []  
      
    def _enrich_document(self, conv_res):  
        # Initialize enrichment models on first use  
        if not self._enrichment_models_initialized:  
            self._init_enrichment_models()  
            self._enrichment_models_initialized = True  
          
        # Unload heavy models after assembly  
        self._unload_heavy_models()  
          
        # Run enrichment  
        return super()._enrich_document(conv_res)  
      
    def _init_enrichment_models(self) -> None:  
        # Initialize enrichment models here  
          
        self.enrichment_pipe = [  
            MyCodeFormulaModel(  
                enabled=self.pipeline_options.do_code_enrichment  
                or self.pipeline_options.do_formula_enrichment,  
                artifacts_path=self.artifacts_path,  
                options=CodeFormulaModelOptions(  
                    do_code_enrichment=self.pipeline_options.do_code_enrichment,  
                    do_formula_enrichment=self.pipeline_options.do_formula_enrichment,  
                ),  
                accelerator_options=self.pipeline_options.accelerator_options,  
            ),  
            # Add other enrichment models as needed  
        ]  
      
    def _unload_heavy_models(self) -> None:  
        # Unload OCR, layout, and table models to free VRAM  
        if hasattr(self, 'ocr_model'):  
            del self.ocr_model  
            self.ocr_model = None  
          
        if hasattr(self, 'layout_model'):  
            del self.layout_model  
            self.layout_model = None  
          
        if hasattr(self, 'table_model'):  
            del self.table_model  
            self.table_model = None  
          
        # Clear GPU cache if using torch  
        try:  
            import torch  
            if torch.cuda.is_available():  
                torch.cuda.empty_cache()  
        except ImportError:  
            pass

        
def process_pdf(input_doc_path, output_dir):
    logging.basicConfig(level=logging.INFO)
    doc_name = Path(input_doc_path).stem
    input_doc_path, output_dir = Path(input_doc_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep page/element images so they can be exported. The `images_scale` controls
    # the rendered image resolution (scale=1 ~ 72 DPI). The `generate_*` toggles
    # decide which elements are enriched with images.
    pipeline_options = PdfPipelineOptions(
        artifacts_path=DOCLING_ARTIFACTS_PATH, 
        table_batch_size=4,
        layout_batch_size=4,
        queue_max_size=100,
    )
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.do_table_structure = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_ocr = False
    pipeline_options.table_structure_options.do_cell_matching = True  # uses text cells predicted from table structure model
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model
    pipeline_options.enable_remote_services=True  # <-- this is required!

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=6, # Number of CPU threads to use for model inference. Higher values can improve throughput on multi-core systems but may increase memory usage. Can be set via DOCLING_NUM_THREADS or OMP_NUM_THREADS environment variables. Recommended: number of physical CPU cores.
        device=AcceleratorDevice.CUDA
    )
    from docling.datamodel.settings import settings
    settings.perf.page_batch_size = 4 # default is 4
    #settings.perf.elements_batch_size = 1  # default is 16, apparemment le nombre d'éléments processed en même temps par les enrichments models
    
    doc_converter = DocumentConverter(  
        format_options={  
            InputFormat.PDF: PdfFormatOption(  
                pipeline_cls=MyStandardPdfPipelineWithAPIFormulaEnrichment,  
                pipeline_options=pipeline_options,  
            )  
        }  
    ) 
    start_time = time.time()
    
    start = timer()
    conv_res = doc_converter.convert(input_doc_path)
    end = timer()
    conversion_time = end-start
    print("Début du post processing du doc")
    start = timer()
    doc_post_processing(conv_res.document, output_dir)
    end = timer()
    doc_pp_time = end-start
    print(f"Fin du post processing du doc en {end-start} secondes")
    #print(f"Confiance dans le parsing : {conv_res.confidence}")
    
    doc_filename = conv_res.input.file.stem  
    """with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_res.document.export_to_doctags())
    
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        json.dump(conv_res.document.export_to_dict(), fp, ensure_ascii=False, indent=4)"""

    serializer = MarkdownDocSerializer(
        doc=conv_res.document,
        table_serializer=ReferencedTableSerializer(input_doc_path, output_dir, doc_filename, save_tables=True),
        picture_serializer=ReferencedPictureSerializer(input_doc_path, output_dir, doc_filename, save_images=True),
        text_serializer=FilteredTextSerializer(input_doc_path=input_doc_path, output_path=output_dir, save_formulas=True)
    )
    print("Début de la sérialisation du document")
    start = timer()
    ser_result = serializer.serialize()
    end = timer()
    serialization_time = end-start
    ser_text = ser_result.text
    print(f"Fin de la sérialisation du document en {end-start} secondes")

    """with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(ser_text)"""

    class CustomSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: DoclingDocument):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=ReferencedTableSerializer(input_doc_path, output_dir, doc_filename, add_captions=True),
                picture_serializer=ReferencedPictureSerializer(input_doc_path, output_dir, doc_filename, add_captions=True),
                text_serializer=FilteredTextSerializer(input_doc_path=input_doc_path, output_path=output_dir)
            )

    MAX_TOKENS = 2048 

    tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_PATH), max_tokens=MAX_TOKENS)

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,  # optional, defaults to True
        serializer_provider = CustomSerializerProvider()
    )
    print("Début du chunking hybride")
    start = timer()
    chunk_iter = chunker.chunk(dl_doc=conv_res.document)
    end = timer()
    chunking_time = end-start
    print(f"Fin du chunking hybride en {end-start} secondes")
    chunks = list(chunk_iter)
    hierarchy = parse_markdown_paths(ser_text)
    
    with open(output_dir/ f"chunks.json","w",encoding="utf-8") as chunks_file : 
        print("Début du post processing des chunks")
        start = timer()
        data = chunks_post_processing(chunks, hierarchy, doc_name, tokenizer=AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_PATH), max_tokens=MAX_TOKENS) 
        end = timer()
        chunk_pp_time = end-start
        print(f"Post processing des chunks effectué en {end-start} secondes")
        json.dump(data, chunks_file, ensure_ascii=False, indent=2)
    with open(output_dir / f"chunks.md", "w", encoding="utf-8") as fp :
        for i in range(len(data)) :
            fp.write(f"Chunk numéro {i}:\n\n{data[i]['content']}\n\n")

    end_time = time.time() - start_time

    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")
    figures_path = output_dir / 'pictures'
    if figures_path.exists() :
        n_images = sum(os.path.isfile(os.path.join(str(figures_path),f)) for f in os.listdir(figures_path))-1
    else : 
        n_images = 0
    
    num_formulas = len([  
        t for t in conv_res.document.texts   
        if isinstance(t, TextItem) and t.label == DocItemLabel.FORMULA  
    ])
    stats = {
        'total time': end_time,
        'conversion time': conversion_time,
        'doc postprocessing time': doc_pp_time,
        'doc serialization time': serialization_time,
        'chunking time': chunking_time,
        'chunks postprocessing time': chunk_pp_time,
        'number of pages': len(list(conv_res.document.pages.keys())),
        'number of tables': len(conv_res.document.tables),
        'number of formulas': num_formulas,
        'number of images': n_images
    }
    return stats

