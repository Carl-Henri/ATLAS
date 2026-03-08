import gradio as gr
from gradio import ChatMessage

from agentic_workflow_2 import invoke_workflow
from workflows.workflow_HLR_analysis import invoke_HLR_analysis_workflow
from workflows.workflow_HLT_coverage import workflow_HLT_coverage
from workflows.workflow_test_architecture_generation import generate_test_architecture
from workflows.workflow_derived_analysis import workflow_derived_analysis
from workflows.workflow_declination_analysis import workflow_declination_analysis
from workflows.requirement_analysis import analyse_requirement

def get_title(conversation) :
    premier_message = conversation[0]['content']
    title = premier_message
    if len(conversation) >= 2 :
        premiere_reponse = conversation[1]['content']
        if '# HLR Analysis' in premiere_reponse : 
            title = "Analyse de HLR : " + premier_message 
        
        if '# Test architecture' in premiere_reponse  :
            title = 'Architecture de test : ' + premier_message


    if len(conversation) >= 3 :
        if '# Report on how the HLR is tested' in conversation[2]['content'] : 
            title = 'Analyse de couverture : ' + premier_message   
        if "## HLR derivation analysis" in conversation[2]['content'] :
            title = 'Analyse de dérivation : ' + premier_message
        if "## SR declination analysis" in conversation[2]['content'] :
            title = 'Analyse de déclinaison : ' + premier_message

    return title

class CustomChatInterface(gr.ChatInterface):  
    def __init__(self, *args, history_samples_per_page: int = 4, **kwargs):  
        self._history_samples_per_page = history_samples_per_page  
        self._history_textbox_component = gr.Textbox(visible=False)  
        super().__init__(*args, **kwargs)  
  
    def _generate_chat_title(self, conversation: list[dict]) -> str:  
        """  
        Génère un titre personnalisé pour la conversation  
        """  
        title = get_title(conversation)

        return title
    def _render_history_area(self):  
        with gr.Column(scale=1, min_width=100):  
            self.new_chat_button = gr.Button(  
                "New Chat",  
                variant="primary",  
                size="md",  
                icon=gr.utils.get_icon_path("plus.svg"),  
            )  
            self.chat_history_dataset = gr.Dataset(  
                components=[self._history_textbox_component],  
                show_label=False,  
                layout="table",  
                type="index",  
                samples_per_page=self._history_samples_per_page,  
                elem_classes=["history-tile"], 
            )  
  
    def _load_chat_history(self, conversations):  
        return gr.Dataset(  
            samples=[  
                [self._generate_chat_title(conv)]  
                for conv in conversations or []  
                if conv  
            ],  
            components=[self._history_textbox_component],  # reuse same instance  
            show_label=False,  
            layout="table",  
            type="index",  
            samples_per_page=self._history_samples_per_page,
            elem_classes=["history-tile"], 
        )

def get_HLR_analysis(HLR_name) :
    img_list = []
    results = {}
    for res in invoke_HLR_analysis_workflow(HLR_name) :
        messages = []
        if 'step' in res :
            results[res['step']] = res['response']
            for step in results :
                message = ChatMessage(
                    role="assistant",  
                    content=results[step],
                    metadata={"title": step},
                )
                messages.append(message)
            yield(False, (messages, img_list))
        else :
            message = ChatMessage(
                role="assistant",  
                content=res['final_answer'],
            )
            workflow_responses = res['workflow_responses']
            HLR_analysis = res['final_answer']
            HLR_full_text = list(workflow_responses.values())[0]
            messages = [message]
            yield(True, (messages, img_list, HLR_analysis, HLR_full_text))
        
# Différentes fonctions possibles pour le chat
def chat_function(message, history, analyse_mode, current_workflow, doc_choices):
    message = message.strip()
    os.makedirs('saved_results', exist_ok=True)
    if current_workflow == "Assistant projet aéronautique" :
        response, log, img_list = invoke_workflow(message, history, analyse_mode, doc_choices)
        log_message = "\n".join(log)  # Concaténer tous les éléments du log en une seule chaîne

        messages = [
            {"role":"assistant","content":log_message, "metadata":{"title":f"Détail des actions effectuées","status":'done'}},
            {"role": "assistant", "content": response}
        ]
        yield messages, img_list
    
    elif current_workflow == "Analyse de HLR" :
        for done, res in get_HLR_analysis(message) :
            if done :
                messages, img_list, _, _ = res
                yield messages, img_list

            else : 
                messages, img_list = res
                yield messages, img_list
    
    elif current_workflow == "Analyse de couverture d'une HLR" : 
        for done, res in get_HLR_analysis(message) :
            if done :
                _, img_list, HLR_analysis, HLR_full_text = res
                messages = [ChatMessage(
                    role="assistant",
                    content=HLR_analysis,
                    metadata={'title':'HLR Analysis'}
                )]
                yield messages, img_list
            else : 
                messages, img_list = res
                yield messages, img_list
        results = {}
        results['HLR Analysis'] = messages[0].content
        img_list = []
        for step, result in workflow_HLT_coverage(message, HLR_analysis, HLR_full_text) :
            results[step] = result
            messages = []
            for step in results : 
                message = ChatMessage(
                    role="assistant",  
                    content=results[step],
                    metadata={"title": step}  
                )
                messages.append(message)
            yield messages, img_list
        messages = []
        del results['HLR Analysis']
        for key, res in results.items() :
            if key == "Workflow log" :
                messages = [ChatMessage(role="assistant", content=res, metadata={"title": "Détail du workflow effectué","status":"done"})] + messages
            else :
                messages.append(ChatMessage(
                    role="assistant",  
                    content=res,
                ))
        yield messages, img_list
    
    elif current_workflow == "Génération d'une architecture de test d'une HLR" :
        for done, res in get_HLR_analysis(message) :
            if done :
                _, img_list, HLR_analysis, HLR_full_text = res
                messages = [ChatMessage(
                    role="assistant",
                    content=HLR_analysis,
                    metadata={'title':'HLR Analysis'}
                )]
                yield messages, img_list
            else : 
                messages, img_list = res
                yield messages, img_list
        messages += [ChatMessage(
            role="assistant",
            content="Generation in progress...",
            metadata={'title':'Test architecture generation'}
        )]
        yield messages, img_list
        response = generate_test_architecture(message, HLR_analysis)
        img_list = []
        messages = [
            {"role": "assistant", "content": response}
        ]
        yield messages, img_list
    
    elif current_workflow == "Analyse de la dérivation d'une HLR" :
        results = {}
        for done, res1, res2 in workflow_derived_analysis(message) :
            if not(done) :
                results[res1] = res2
                messages = []
                for step in results : 
                    message = ChatMessage(
                        role="assistant",  
                        content=str(results[step]),
                        metadata={"title": step}  
                    )
                    messages.append(message)
                yield messages, []
            else :
                response = res1
                str_log = res2
                messages = [
                    {"role":"assistant","content":str_log, "metadata":{"title":f"Détail du workflow effectué","status":'done'}},
                    {"role": "assistant", "content": response}
                ]
                yield messages, []
    elif current_workflow == "Analyse d'exigence" :
        response = analyse_requirement(message, history)
        yield [{'role':'assistant','content':response}], []
    
    elif current_workflow == "Analyse de déclinaison d'une SR" :
        results = {}
        for done, res1, res2 in workflow_declination_analysis(message) :
            if not(done) :
                results[res1] = res2
                messages = []
                for step in results : 
                    message = ChatMessage(
                        role="assistant",  
                        content=str(results[step]),
                        metadata={"title": step}  
                    )
                    messages.append(message)
                yield messages, []
            else :
                response = res1
                str_log = res2
                messages = [
                    {"role":"assistant","content":str_log, "metadata":{"title":f"Détail du workflow effectué","status":'done'}},
                    {"role": "assistant", "content": response}
                ]
                yield messages, []

import re
import unicodedata

def safe_filename(text: str, replace_space="_") -> str:
    # Normalisation Unicode (é → e)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remplacer les espaces
    text = text.replace(" ", replace_space)

    # Garder uniquement lettres, chiffres, _ et -
    text = re.sub(r"[^a-zA-Z0-9_-]", "", text)

    return text

# Fonction pour transformer la conversation en Markdown
def conversation_to_markdown(conversation):
    md_text = ""
    for msg in conversation:
        if not(msg['metadata']) : 
            md_text += f"# {msg['role'].capitalize()}: \n\n{msg['content']}\n\n"
 
    conv_name = get_title(conversation)
    if len(conv_name) > 100 :
        conv_name = conv_name[:100]+'...'
    else :
        conv_name = conv_name[:100]

    return md_text, conv_name

from markdown import Markdown
import tempfile
import os
import gradio as gr

def chatbot_to_html(conversation):
    if not(conversation) :
        return gr.File(visible=False)

    text, conv_name = conversation_to_markdown(conversation)
    # --- STEP 1: Advanced Math Shielding ---
    # This regex finds everything between delimiters, including newlines (?s flag)
    # and replaces them with a unique placeholder.
    math_blocks = []
    
    def shield_math(match):
        placeholder = f"!!MATH_BLOCK_{len(math_blocks)}!!"
        math_blocks.append(match.group(0))
        return placeholder

    # Shield Block Math \[ ... \]
    text = re.sub(r'\\\[(.*?)\\\]', shield_math, text, flags=re.DOTALL)
    # Shield Inline Math \( ... \)
    text = re.sub(r'\\\((.*?)\\\)', shield_math, text, flags=re.DOTALL)

    # --- STEP 2: Fix Lists & Spacing ---
    # Ensure space after bullets/numbers
    text = re.sub(r'^(\d+)\.(?=[^\s])', r'\1. ', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s*(?=[^\s])', '- ', text, flags=re.MULTILINE)
    # Ensure blank line before lists
    text = re.sub(r'([^\n])\n(\s*(\d+\.|-)\s)', r'\1\n\n\2', text)

    # --- STEP 3: Convert to HTML ---
    md = Markdown(extensions=["fenced_code", "tables", "sane_lists", "nl2br", "md_in_html"])
    html = md.convert(text)

    # --- STEP 4: Restore Math exactly as it was (no <br> tags added) ---
    for i, original_math in enumerate(math_blocks):
        html = html.replace(f"!!MATH_BLOCK_{i}!!", original_math)

    html_body = html

    # HTML final
    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{conv_name}</title>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{ left: '$$', right: '$$', display: true }},
            {{ left: '$', right: '$', display: false }},
            {{ left: '\\\\(', right: '\\\\)', display: false }},
            {{ left: '\\\\[', right: '\\\\]', display: true }}
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    }});"></script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>hljs.highlightAll();</script>

<style>
    /* --- Base Settings --- */
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        background: #ffffff;
        color: #24292f; /* Soft black for better contrast */
        padding: 40px;
        max-width: 900px;
        margin: auto;
        line-height: 1.6; /* Improves general readability */
    }}

    /* --- Headings --- */
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
    }}
    h1 {{ font-size: 2em; border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.5em; border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; }}

    /* --- List Styling (Optimized) --- */
    ul, ol {{
        padding-left: 2em;
        margin-top: 0;
        margin-bottom: 16px;
    }}

    li {{
        margin-bottom: 0.35em; /* Breathing room between items */
        word-wrap: break-word; /* Prevents long words breaking layout */
    }}

    /* Ensure text wraps nicely next to bullet, not under it */
    ul {{ list-style-position: outside; list-style-type: disc; }}
    ol {{ list-style-position: outside; }}

    /* Nested lists styling */
    li > ul, li > ol {{
        margin-top: 0.5em;
        margin-bottom: 0.5em;
    }}
    ul ul, ul ol, ol ul, ol ol {{
        margin-bottom: 0;
    }}

    /* Subtle bullet color */
    li::marker {{
        color: #57606a; 
    }}

    /* --- Code Blocks --- */
    pre {{
        background: #f6f8fa;
        padding: 16px;
        border-radius: 6px;
        overflow-x: auto;
        line-height: 1.45;
    }}
    code {{
        font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
        font-size: 85%;
        background-color: rgba(175, 184, 193, 0.2);
        padding: 0.2em 0.4em;
        border-radius: 6px;
    }}
    pre code {{
        background-color: transparent;
        padding: 0;
    }}

    /* --- Blockquotes --- */
    blockquote {{
        border-left: 4px solid #d0d7de;
        padding: 0 1em;
        color: #57606a;
        margin: 0 0 16px 0;
    }}

    /* --- Tables --- */
    table {{
        border-spacing: 0;
        border-collapse: collapse;
        display: block;
        width: max-content;
        max-width: 100%;
        overflow: auto;
        margin-bottom: 16px;
    }}
    tr {{ border-top: 1px solid #d0d7de; }}
    tr:nth-child(2n) {{ background-color: #f6f8fa; }}
    th, td {{
        padding: 6px 13px;
        border: 1px solid #d0d7de;
    }}
    th {{ font-weight: 600; }}
</style>
</head>

<body>
{html_body}
</body>
</html>
"""
    # dossier temporaire (auto-clean)
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, (safe_filename(conv_name) + ".html"))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return gr.File(value=output_path, visible=True)


import pyperclip 
import markdown
from bs4 import BeautifulSoup

def on_copy(copy_data: gr.CopyData):  
    # Le contenu original est DÉJÀ dans le presse-papiers  
    # On doit le remplacer avec notre version modifiée  
      
    # Convertir le Markdown en HTML
    md_text = copy_data.value
    html = markdown.markdown(md_text)
    
    # Utiliser BeautifulSoup pour analyser le HTML
    soup = BeautifulSoup(html, "html.parser")
    
    # Extraire le texte brut
    plain_text = soup.get_text()
    # Remplacer le contenu du presse-papiers  
    pyperclip.copy(plain_text)  
    return