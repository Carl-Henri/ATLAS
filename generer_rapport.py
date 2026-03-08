import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY
from xml.sax.saxutils import escape


class MyDocTemplate(SimpleDocTemplate):
    """Classe pour créer des signets cliquables dans le PDF (table des matières)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._toc_entries = []

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = getattr(flowable.style, 'name', '')
            if style_name == 'Heading2':
                text = flowable.getPlainText()
                key = f"toc_{len(self._toc_entries)}"
                try:
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(text, key, level=0, closed=False)
                except Exception:
                    pass
                # Ajouter un élément à la table des matières
                self.notify('TOCEntry', (0, text, self.page))
                self._toc_entries.append(key)


def _fmt_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _escape_and_break(s: str) -> str:
    """Échapper et ajouter des sauts de ligne dans le texte."""
    if s is None:
        return ""
    return escape(str(s)).replace('\n', '<br/>')

from pathlib import Path
def generer_rapport(json_paths: list, output_pdf: str):
    """Génère un rapport PDF comparant les réponses de plusieurs tests sur différentes questions."""
    # Organiser les données par benchmark et test
    benchmarks = {}

    # Traiter chaque fichier JSON
    for json_path in json_paths:
        # Extraire le nom du benchmark et du test à partir du chemin
        benchmark_name = Path(json_path).parent.parent.name # Le nom du dossier parent
        test_name = Path(json_path).parent.name  # Nom du fichier sans l'extension

        # Charger les données du fichier JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # Organiser les données dans un dictionnaire par benchmark et test
        if benchmark_name not in benchmarks:
            benchmarks[benchmark_name] = {}
        
        benchmarks[benchmark_name][test_name] = test_data

    # Styles de texte
    styles = getSampleStyleSheet()
    justify_style = ParagraphStyle('Justify', parent=styles['Normal'], alignment=TA_JUSTIFY)
    h2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontName='Helvetica-Bold')
    h3_style = ParagraphStyle('Heading3', parent=styles['Heading3'], fontName='Helvetica-Bold')
    h4_style = ParagraphStyle('Heading4', parent=styles['Heading4'], fontName='Helvetica-Bold')

    # Elements du PDF
    elements = []

    # Page de garde
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("<b>Rapport comparatif des tests</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Analyse des réponses de différents tests pour chaque question", h2_style))
    elements.append(Spacer(1, 12))
    today = datetime.today().strftime("%d/%m/%Y")
    elements.append(Paragraph(f"Date de génération : {today}", styles['Normal']))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Ce document présente une comparaison détaillée des résultats de plusieurs tests sur chaque question. Les performances de chaque test sont également comparées.", justify_style))
    elements.append(PageBreak())

    # Table des matières
    elements.append(Paragraph('Table des matières', styles['Title']))
    doc = MyDocTemplate(output_pdf, pagesize=A4)

    # Traiter chaque benchmark
    for benchmark_name, tests in benchmarks.items():
        elements.append(Paragraph(f'Benchmark : {benchmark_name}', h2_style))
        elements.append(Spacer(1, 12))

        # Organiser les résultats par question
        questions = {}

        # Traiter chaque test dans ce benchmark
        for test_name, test_data in tests.items():
            for question_info in test_data:
                # Vérifier si question_info est un dictionnaire
                if isinstance(question_info, dict):  # Si c'est un dictionnaire
                    question = question_info.get('query', 'Question non définie')
                    response = question_info.get('response', 'Réponse non définie')
                elif isinstance(question_info, str):  # Si c'est une chaîne
                    question = 'Question non définie'
                    response = question_info
                else:
                    continue  # Si question_info est un type inattendu, on ignore

                # Ajouter la réponse à la question correspondante
                if question not in questions:
                    questions[question] = []

                questions[question].append({
                    "test_name": test_name,
                    "response": response
                })

        # Pour chaque question
        for question, responses in questions.items():
            # Affichage de la question
            elements.append(Paragraph('Question:', h3_style))
            elements.append(Paragraph(_escape_and_break(question), h2_style))
            elements.append(Spacer(1, 6))

            # Affichage des réponses pour chaque test dans ce benchmark
            for response_data in responses:
                test_label = f"Test : {response_data['test_name']}"
                elements.append(Paragraph(test_label, h4_style))
                elements.append(Paragraph(_escape_and_break(response_data['response']), justify_style))
                elements.append(Spacer(1, 6))

            elements.append(PageBreak())


    # Générer le PDF
    doc.multiBuild(elements)
