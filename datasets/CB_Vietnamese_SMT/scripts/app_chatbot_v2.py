import gradio as gr
import os
from os.path import dirname, join
import re
import time
import requests
from chat_engine import SimpleChatbotEngine, PILEngine


# get folder of this script file
wd = os.path.dirname(os.path.realpath(__file__))
print(wd)


chatbot_engine = SimpleChatbotEngine()
pil_engine = PILEngine()

def get_current_time():
    # get current time with format YYYY-MM-DD HH:MM:SS
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_response_rasa(text):
    rasa_server = "http://0.0.0.0:5005/webhooks/rest/webhook"
    # create post request using requests
    data = {"message": text, "client": "default"}
    r = requests.post(rasa_server, json=data)
    output = r.json()
    return output

def get_response_chatbot_engine(text):
    response = chatbot_engine.get_response(text)
    return response

def reset_chat_rasa():
    sender_id = "default"
    url = f"http://0.0.0.0:5005/conversations/{sender_id}/tracker/events"
    data = {
        "event": "restart"
    }
    requests.post(url, json=data)
    print("Restart covnersation")

def task_build_rasa(state):
    is_build, is_run = state
    rasa_folder = join(dirname(wd), 'chatbot_template')

    output = ""

    if is_build:
        output += f'{get_current_time()} INFO Start build model\n'
        output += os.popen(f'cd {rasa_folder}; rasa train').read()
        output += f'\n{get_current_time()} INFO Finish build model\n'
        regex_pattern = r'\[\d+m'
        output = re.sub(regex_pattern, '', output)

    if is_run:
        output += f'{get_current_time()} INFO Start run rasa service\n'
        os.popen(f'cd {rasa_folder}; rasa run --enable-api')
        command = f'cd {wd}; python wait_rasa_start.py'
        print(command)
        output += os.popen(command).read()
        output += f'{get_current_time()} INFO Finish run rasa service\n'
    
    return output

def reset_chat():
    reset_chat_rasa()

def clear_conversation():
    return gr.update(value=None, visible=True), None, []
    
def task_chat(chatbot, input, context=[]):
    time.sleep(0.3)
    if input.startswith(":"):
        response_data = pil_engine.get_response(input, chatbot_engine)
        display_input = pil_engine.get_display_input(input)
    else:
        response_data = chatbot_engine.get_response(input)
        display_input = input
    try:
        response = response_data[0]['text']
    except:
        response = "..."
    chatbot.append((display_input, response))
    context.append((input, response))
    img = "graph-1.svg"
    return chatbot, context, img

chatbot_block_css = """
#chatbot {
    height: 500px;
}

#graph {
    height: 500px;
}

#graph img {
    height: 500px;
}
"""
wd = os.path.dirname(os.path.realpath(__file__))
# add two tab 
with gr.Blocks(css=chatbot_block_css) as app:
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot").style(color_map=("#1D51EE", "#585A5B"))
                context = gr.State([])
                with gr.Row():
                    with gr.Column(scale=12):
                        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
                    with gr.Column(min_width=50, scale=1):
                        submitBtn = gr.Button("🚀", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        resetBtn = gr.Button("🔄 Reset", variant="danger")
            
            with gr.Column(scale=4):
                img = gr.Image(join(wd, "graph-1.svg"), type="filepath").style(height=500)

    with gr.Tab("Build"):
        gr.Markdown("Build models and run chatbot")
        text_output = gr.Textbox()

        with gr.Row():
            with gr.Column(scale=2):
                buildAndRunBtn = gr.Button("🔄 Build & Run")
            
            with gr.Column(scale=1):
                buildBtn = gr.Button("🛠️ Build")
            
            with gr.Column(scale=1):
                runBtn = gr.Button("🚀 Run")

    txt.submit(task_chat, [chatbot, txt, context], [chatbot, context, img], show_progress=True)
    txt.submit(lambda: "", None, txt)
    submitBtn.click(task_chat, [chatbot, txt, context], [chatbot, context, img], show_progress=True)
    submitBtn.click(lambda: "", None, txt)

    # resetBtn.click(reset_chat)
    # resetBtn.click(clear_conversation, [], [txt, chatbot, context])
    
    # buildAndRunBtn.click(task_build_rasa, gr.State([True, True]), outputs=text_output)
    # buildBtn.click(task_build_rasa, gr.State([True, False]), outputs=text_output)
    # runBtn.click(task_build_rasa, gr.State([False, True]), outputs=text_output)

app.launch(debug=True)
