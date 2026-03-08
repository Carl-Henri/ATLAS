from mistral_langchain_wrapper import MistralChatWrapper
import os 
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
llm = MistralChatWrapper(api_key=API_KEY, model="medium")

def analyse_requirement(message, history) :
    with open('workflows/req_analysis_system_prompt.txt', 'r', encoding='utf-8') as system_prompt_f :
        system_prompt = system_prompt_f.read()
    
    messages = [{'role':'system', 'content':system_prompt}]
    
    if history :
        messages.extend(history)
    
    messages.append({'role':'user', 'content':message})

    result = llm.invoke(messages)
    return(result.content)