import gradio as gr
import os
from os.path import dirname, join
import re
import time
import requests


wd = os.path.dirname(__file__)

def get_response_rasa(text):
    rasa_server = "http://0.0.0.0:5005/webhooks/rest/webhook"
    # create post request using requests
    data = {"message": text, "client": "default"}
    r = requests.post(rasa_server, json={"message": "hello"})
    output = r.json()
    return output

def task_build_rasa():
    rasa_folder = join(dirname(wd), 'chatbot_template')

    output = ""

    # # output += os.popen(f'cd {rasa_folder}; rasa train').read()
    # regex_pattern = r'\[\d+m'
    # output = re.sub(regex_pattern, '', output)

    output += '\nRun rasa server\n'
    os.popen(f'cd {rasa_folder}; rasa run --enable-api')
    output += os.popen(f'cd {wd}; python wait_rasa_start.py').read()
    
    return output

def task_chat(chatbot, input, context=[]):
    time.sleep(0.3)
    response_data = get_response_rasa(input)
    response = response_data[0]['text']
    chatbot.append((input, response))
    context.append((input, response))
    return chatbot, context

chatbot_block_css = """
#chatbot {
    height: 500px;
}
"""
# add two tab 
with gr.Blocks(css=chatbot_block_css) as app:
    gr.Markdown("Chatbot")

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(elem_id="chatbot").style(color_map=("#1D51EE", "#585A5B"))
        context = gr.State([])

        with gr.Row():
            with gr.Column(scale=12):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            with gr.Column(min_width=50, scale=1):
                submitBtn = gr.Button("🚀", variant="primary")
    with gr.Tab("Build"):
        gr.Markdown("Build models and run chatbot")
        text_output = gr.Textbox()
        text_button = gr.Button("Build")

    

    txt.submit(task_chat, [chatbot, txt, context], [chatbot, context], show_progress=True)
    txt.submit(lambda: "", None, txt)
    submitBtn.click(task_chat, [chatbot, txt, context], [chatbot, context], show_progress=True)
    submitBtn.click(lambda: "", None, txt)
    
    
    text_button.click(task_build_rasa, outputs=text_output)

app.launch()
