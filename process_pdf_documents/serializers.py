from typing import Any, Optional
from process_pdfs_utils import extract_text_from_bbox, should_keep_image, is_toc_table, get_object_caption
from docling_core.transforms.serializer.markdown import (
    MarkdownTextSerializer,
    BaseDocSerializer,
    create_ser_result,
    SerializationResult,
)
from docling_core.types.doc import DoclingDocument

from pathlib import Path
import re
import json
from typing_extensions import override

from docling_core.types.doc import PictureItem, TableItem, DoclingDocument

#from docling_core.transforms.chunker.hierarchical_chunker import TripletTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, MarkdownPictureSerializer, BaseDocSerializer, create_ser_result, SerializationResult

from docling_core.types.doc import PictureItem, TableItem

class ReferencedPictureSerializer(MarkdownPictureSerializer):
    def __init__(self, pdf_path: Path, output_dir: Path, doc_filename: str, save_images = False, add_captions = False):
        super().__init__()
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.doc_filename = doc_filename
        self.picture_counter = 0
        self.picture_deleted = 0
        self.save_images = save_images
        self.add_captions = add_captions

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:

        if should_keep_image(item, doc) :
            # Sérialisation
            self.picture_counter += 1
            fname = f"picture-{self.picture_counter}.png"
            parts = [f"<!-- Figure number: {self.picture_counter} -->"]
            if self.add_captions : 
                caption = get_object_caption(self.pdf_path, item.prov[0].page_no)
                if caption :
                    parts.append(f"<!-- Figure caption: {caption} -->")
            md_text = "\n".join(parts)

            # Sauvegarde de l'image en png
            if self.save_images :
                save_item_img_and_metadata(folder=self.output_dir / "pictures", item=item, doc=doc, item_counter=self.picture_counter, item_class="picture")

            return create_ser_result(text=md_text, span_source=item)
        else :
            self.picture_deleted += 1
            # Sauvegarde de l'image en png
            if self.save_images : 
                save_item_img_and_metadata(folder=self.output_dir / "pictures_deleted", item=item, doc=doc, item_counter=self.picture_deleted, item_class="picture")
            return create_ser_result(text="", span_source=item)

def save_item_img_and_metadata(folder, item, doc, item_counter, item_class) :
    # On sauvegarde l'image de l'item
    element_image_filename = (
        folder / f"{item_class}-{item_counter}.png"
    )
    folder.mkdir(parents=True, exist_ok=True)

    with element_image_filename.open("wb") as fp:
        item.get_image(doc).save(fp, "PNG")

    # On sauvegarde aussi sa position (bbox + page num) dans le document
    t_metadata_dir = folder / f"{item_class}_metadata.json"
    try :
        with open(t_metadata_dir,"r",encoding="utf-8") as tables_metadatas : 
            try :
                t_metadata = json.load(tables_metadatas)
            except :
                t_metadata = []
    except :
        t_metadata = []
    entry = {
        f"{item_class}_no":item_counter,
        f"{item_class}_bbox":list(item.prov[0].bbox),
        f"{item_class}_page_no":item.prov[0].page_no
    }
    t_metadata.append(entry)
    with open(t_metadata_dir,"w",encoding="utf-8") as tables_metadatas : 
        json.dump(t_metadata, tables_metadatas, ensure_ascii=False, indent=2)

class ReferencedTableSerializer(MarkdownTableSerializer):
    def __init__(self, input_doc_path, output_dir: Path, doc_filename: str, add_captions = False, save_tables = False):
        """
        :param max_len: longueur max du markdown avant fallback vers l'image
        """
        super().__init__()
        self.input_doc_path = input_doc_path
        self.output_dir = output_dir
        self.doc_filename = doc_filename
        self.not_serialized_table_counter = 0
        self.serialized_table_counter = 0
        self.save_tables = save_tables
        self.add_captions = add_captions

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        
        # On essaie d'obtenir la version markdown via le serializer standard
        parent_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **kwargs,
        )
        # 1. Si c'est une table of content, on retourne que le texte
        if is_toc_table(item, doc) :
            #md_text = f"<!-- Table of contents path: {rel_path} -->"
            page_no = item.prov[0].page_no
            bbox = dict(item.prov[0].bbox)
            md_text = extract_text_from_bbox(self.input_doc_path, page_no, bbox, coord_origin="BOTTOMLEFT")
            # Nettoyage du texte
            md_text = md_text.strip()

            md_text = re.sub(r"[ \t]+", " ", md_text)  # espaces multiples → un seul
            md_text = re.sub(r" *\n *", "\n", md_text)  # nettoyer espaces autour des \n
            md_text = re.sub(r"\n{3,}", "\n\n", md_text)  # max 2 retours vides consécutifs

            # Réduire les séquences de points à un maximum de trois
            md_text = re.sub(r"\.{4,}", "...", md_text)
            self.serialized_table_counter += 1 
            if self.save_tables :
                save_item_img_and_metadata(folder=self.output_dir / "tables_serialized", item=item, doc=doc, item_counter=self.serialized_table_counter, item_class="table")

        # 2. Si la représentation markdown n'est pas trop "lourde", on la garde (critère sur le nombre de colonnes, 10 max)
        elif item.data.num_cols <= 10:
            self.serialized_table_counter += 1 
            if self.add_captions : 
                caption = get_object_caption(self.input_doc_path, item.prov[0].page_no, object_type= "table")
                if caption :
                    md_text = f"<!-- Table markdown {self.serialized_table_counter} start -->\n<!-- Table caption: {caption} -->\n{parent_res.text}\n<!-- Table markdown {self.serialized_table_counter} end -->"
                else :    
                    md_text = f"<!-- Table markdown {self.serialized_table_counter} start -->\n{parent_res.text}\n<!-- Table markdown {self.serialized_table_counter} end -->"
            else :
                md_text = f"<!-- Table markdown {self.serialized_table_counter} start -->\n{parent_res.text}\n<!-- Table markdown {self.serialized_table_counter} end -->"
            if self.save_tables :
                save_item_img_and_metadata(folder=self.output_dir / "tables_serialized", item=item, doc=doc, item_counter=self.serialized_table_counter, item_class="table")
        else:
            # 3. Sinon, fallback en image
            self.not_serialized_table_counter += 1      
            md_text = f"<!-- Table number: {self.not_serialized_table_counter} -->"
            if self.add_captions : 
            
                caption = get_object_caption(self.input_doc_path, item.prov[0].page_no, object_type= "table")
                if caption :
                    md_text+=f"\n<!-- Table caption: {caption} -->"
            if self.save_tables :
                save_item_img_and_metadata(folder=self.output_dir / "tables", item=item, doc=doc, item_counter=self.not_serialized_table_counter, item_class="table")


        return create_ser_result(text=md_text, span_source=item)

class FilteredTextSerializer(MarkdownTextSerializer):
    """
    Sérialiseur personnalisé :
      - saute les <page_header> et <page_footer>
      - convertit certains <section_header_level_1> contenant des mots-clés (exceptions)
        en texte simple au lieu de titre
    """
    input_doc_path : Path
    output_path : Path
    num_formula : int = 1
    save_formulas : bool = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def serialize(
        self,
        *,
        item: Any,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """
        Appelée pour chaque élément du document.
        On peut choisir de :
          - sauter certains éléments (return texte vide)
          - modifier le texte de sortie avant de le renvoyer
        """
        # Ignorer headers et footers
        if item.label == "page_header" or item.label == "page_footer" or item.label == "footnote":
            return create_ser_result(text="", span_source=item)
        
        """ if item.label == "formula":
            page_no = item.prov[0].page_no
            bbox = dict(item.prov[0].bbox)
            img_dir_path = self.output_path / "formulas"
            if self.save_formulas :
                save_img_from_bbox(self.input_doc_path, page_no, bbox, img_dir_path, self.num_formula, coord_origin="BOTTOMLEFT")
            self.num_formula += 1
            return create_ser_result(text=f"<!-- Formula number: {self.num_formula} ; Page no : {page_no}-->\n", span_source=item)
            """
        # Tout le reste : comportement standard
        parent_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            separator=separator,
            **kwargs,
        )
        return create_ser_result(text=parent_res.text, span_source=item)