import fitz  # PyMuPDF

def extraire_pages_fitz(pdf_entree, pdf_sortie, pages):
    """
    pdf_entree : chemin du PDF source
    pdf_sortie : chemin du PDF à créer
    pages : liste des numéros de pages à extraire (en base 1)
    """
    doc = fitz.open(pdf_entree)

    # Nouveau PDF vide
    new_pdf = fitz.open()

    for p in pages:
        page_index = p - 1  # fitz utilise des pages 0-based
        new_pdf.insert_pdf(doc, from_page=page_index, to_page=page_index)

    # Sauvegarder le sous-PDF
    new_pdf.save(pdf_sortie)
    new_pdf.close()
    doc.close()

# Exemple : extraire les pages 1, 3 et 5
