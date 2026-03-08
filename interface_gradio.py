import gradio as gr
from datetime import datetime
from interface_gradio_utils import CustomChatInterface, chat_function, chatbot_to_html, on_copy

# Stockage global des logs (en mémoire avant sauvegarde)
log_data = []

css = """  
#big-chatbot {  
    max-width: 100% !important;   
    width: 100% !important;
}  
#chatbot { flex-grow: 1; }
"""  

gallery = gr.Gallery(label="Pages PDF utilisées pour répondre", object_fit="fill", render=False)  
with gr.Blocks(css=css, fill_height=True, fill_width=True, title="Assistant projet aéronautique") as app:  
    current_workflow = gr.State('Assistant projet aéronautique')
    doc_choices = gr.State([])
    analyse_mode = gr.State('fast')
    with gr.Row(equal_height=True, scale=0):
        title=gr.HTML("""<h1 style="text-align: center;">Assistant projet aéronautique</h1>""")  
    with gr.Row(equal_height=True, scale=10): 
        init_textbox = gr.Textbox(  
            lines=1,  # Nombre de lignes visibles fixes  
            max_lines=5,  # Nombre maximum de lignes avant défilement  
            submit_btn=True,
            stop_btn=True,
            info="Appuyez sur Entrée ou cliquez sur la flèche pour envoyer votre message. Maj + entrée pour retourner à la ligne.",
            placeholder="Entrez votre question ici. Pour de meilleurs résultats : mettez les noms des exigences entre guillements (\" \") ; filtrez sur les bons documents.",  
        )
        chatbot = gr.Chatbot(  
            label="Assistant IA",   
            elem_id="chatbot",
            type="messages",   
            latex_delimiters=[{"left":"\\[", "right":"\\]", "display": True}, {"left":"\\(", "right":"\\)", "display": False}],  
            show_copy_all_button=False,  
            show_copy_button=True,  
            show_share_button=False,
            group_consecutive_messages=False,
        )
        chatbot.copy(on_copy, None, None)
        with gr.Column(scale=1) as chatbot_col:
            c = CustomChatInterface(
                fn=chat_function,
                additional_inputs=[analyse_mode, current_workflow, doc_choices],
                additional_outputs=[gallery],
                type="messages",
                chatbot=chatbot,
                textbox=init_textbox,
                fill_height=True,
                cache_mode='eager',
                delete_cache=None,
                analytics_enabled=False,
                save_history=True,
            )
            c.saved_conversations.secret = "acfazrlkasd62054516678"
       
        with gr.Column(scale=0, visible=False) as gallery_col:  
            gallery.render()

    with gr.Row(equal_height=True, scale=0) :
        with gr.Column(scale=1) :
            mode_dropdown = gr.Dropdown(["Assistant projet aéronautique", "Analyse d'exigence", "Analyse de HLR", "Analyse de couverture d'une HLR", "Génération d'une architecture de test d'une HLR", "Analyse de la dérivation d'une HLR", "Analyse de déclinaison d'une SR"], label="Choix de l'outil")
        
        with gr.Column(scale=1):
            convert_btn = gr.Button("Exporter en HTML")

            file_output = gr.File(
                label="Fichier exporté",
                visible=False
            )

            convert_btn.click(
                fn=chatbot_to_html,
                inputs=[chatbot],
                outputs=[file_output]
            )

        with gr.Column(scale=1) :
            afficher_images_checkbox = gr.Checkbox(label="Afficher les pages utilisées pour répondre", value=False)
            
        with gr.Column(scale=1) :
            dropdown = gr.Dropdown(["Analyse rapide des documents", "Analyse intermédiaire des documents", "Analyse poussée des documents"], label="Niveau d'analyse des documents")
        
        # Dictionnaire pour gérer les groupes et leurs options
        from utils import fetch_documents_name, fetch_folders
        from my_paths import RAW_DATA_DIR
        docs_choices = []
        docs_choices.extend(fetch_folders(RAW_DATA_DIR))
        docs_choices.extend(fetch_documents_name(RAW_DATA_DIR))
        with gr.Column(scale=3) :
            dropdown_doc_choices = gr.Dropdown(
                label="Liste des documents auxquels se limiter pour répondre (ne concerne pas les questions sur les calculs (SDD), les différentes versions)",
                choices=docs_choices,
                multiselect=True,
                interactive=True,
                value=[]
            )

    def update_mode(selected_option) :
        textbox_hlr = gr.Textbox(  
            lines=1,  # Nombre de lignes visibles fixes  
            max_lines=5,  # Nombre maximum de lignes avant défilement  
            submit_btn=True,
            info="Appuyez sur Entrée ou cliquez sur la flèche pour envoyer votre message",
            placeholder="Entrez le nom exact d'une HLR ici...",  
        )
        textbox_sr = gr.Textbox(  
            lines=1,  # Nombre de lignes visibles fixes  
            max_lines=5,  # Nombre maximum de lignes avant défilement  
            submit_btn=True,
            info="Appuyez sur Entrée ou cliquez sur la flèche pour envoyer votre message",
            placeholder="Entrez le nom exact d'une SR ici...",  
        )
        textbox = gr.Textbox(  
            lines=1,  # Nombre de lignes visibles fixes  
            max_lines=5,  # Nombre maximum de lignes avant défilement  
            submit_btn=True,
            info="Appuyez sur Entrée ou cliquez sur la flèche pour envoyer votre message. Maj + entrée pour retourner à la ligne.",
            placeholder="Entrez votre question ici. Pour de meilleurs résultats : mettez les noms des exigences entre guillements (\" \") ; filtrez sur les bons documents.",  
        )
        title = f"""<h1 style="text-align: center;">{selected_option}</h1>"""

        if selected_option=="Assistant projet aéronautique":
            return(gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), title, textbox, selected_option, gr.update(value=False))
        elif "HLR" in selected_option:
            return(gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), title, textbox_hlr, selected_option, gr.update(value=False))
        elif "SR" in selected_option:
            return(gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), title, textbox_sr, selected_option, gr.update(value=False))
        else :
            return(gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), title, textbox, selected_option, gr.update(value=False))          
        

    mode_dropdown.change(update_mode, inputs=mode_dropdown, outputs=[afficher_images_checkbox, dropdown, dropdown_doc_choices, title, init_textbox, current_workflow, afficher_images_checkbox])

    def update_doc_choices(selected_options) :
        return selected_options
    
    dropdown_doc_choices.change(update_doc_choices, inputs=dropdown_doc_choices, outputs=doc_choices)

    def update_analyse_mode(selected_option) :
        if selected_option == "Analyse rapide des documents" : 
            return 'fast'
        if selected_option == "Analyse intermédiaire des documents" :
            return 'intermediate'
        if selected_option == "Analyse poussée des documents" :
            return 'complex'
    
    dropdown.change(update_analyse_mode, inputs=dropdown, outputs=analyse_mode)

    def toggle_layout(afficher_images):
        # Cette fonction contrôle la visibilité et le scale des colonnes
        if afficher_images:
            # Affiche galerie et réduit taille chatbot
            return gr.update(visible=True), gr.update(scale=1), gr.update(scale=2)
        else:
            # Cache galerie et chatbot prend toute la place
            return gr.update(visible=False), gr.update(scale=1), gr.update(scale=0)

    # Quand on change la checkbox, ajuster layout
    afficher_images_checkbox.change(fn=toggle_layout,
        inputs=afficher_images_checkbox,
        outputs=[gallery_col, chatbot_col, gallery_col])
    
    state = gr.State([])    
    
# Générer une clé unique pour la nouvelle conversation (ex: timestamp)
new_conv_key = datetime.now().strftime("%Y%m%dT%H%M%S%f") 
#app.queue(default_concurrency_limit=4).launch(
#    server_name="0.0.0.0",
#    server_port=7860
#)

app.launch(inbrowser=True, server_port=7860)