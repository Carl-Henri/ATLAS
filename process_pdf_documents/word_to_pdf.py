import os
import comtypes.client

def doc_to_pdf(doc_path, pdf_path, word):
    print(f"Tentative de conversion de {doc_path}")
    try:
        # Ouvrir en lecture seule
        doc = word.Documents.Open(doc_path, ReadOnly=True)
        doc.ExportAsFixedFormat(
            OutputFileName=pdf_path,
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,         # wdExportOptimizeForPrint
            CreateBookmarks=1      # wdExportCreateHeadingBookmarks
        )
        doc.Close(SaveChanges=False)
        print(f"Converti : {doc_path} -> {pdf_path}")
    except Exception as e:
        print(f"Erreur lors de la conversion de {doc_path} : {e}")

def convert_folder(folder_path):
    word = comtypes.client.CreateObject('Word.Application')
    word.Visible = False
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.doc', '.docx')):
                full_doc_path = os.path.normpath(os.path.join(root, file))
                pdf_file = os.path.splitext(full_doc_path)[0] + '.pdf'
                if os.path.exists(pdf_file):
                    print(f"PDF déjà existant, skip : {pdf_file}")
                else:
                    doc_to_pdf(full_doc_path, pdf_file, word)
    word.Quit()