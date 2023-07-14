from models.ChatGLM2 import ChatGLM
import gradio as gr
import mdtex2html
import time

def LoadLLM():
    llm = ChatGLM(model_path = "YOUR_PATH")
    llm.load_model()
    return llm

# 加载模型
llm = LoadLLM()

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

# 解析文本
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

# 按下按钮发送消息
def onSubmit(input, chatbot):
    text = parse_text(input)
    chatbot.append((text, ""))
    time.sleep(0.05)
    chatbot[-1] = (text, llm(text))
    yield chatbot

# 重置输入框
def reset_user_input():
    return gr.update(value='')

with gr.Blocks() as webpage:
    gr.HTML("""<h1 align="center">550A 智能量子计算机</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale = 4):
            with gr.Column(scale = 12):
                user_input = gr.Textbox(show_label = False, placeholder = "Input...", lines = 1)  # 设置用户输入框

    user_input.submit(onSubmit, [user_input, chatbot], [chatbot], show_progress = False)#
    user_input.submit(reset_user_input, [], [user_input])     # 消息发送之后，重置输入框

# 设置可生成公网访问
webpage.queue()
webpage.launch(share = True)