import os
import gradio as gr
import shutil

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from configs.model_config import *
from local_doc_qa import LocalDocQA

import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint


# 设置OpenAI模型
# prefix_messages = [{"role": "system", "content": "我是我的研究助手——普罗米修斯17号, 你也可以叫我普罗米修斯"}]
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'
llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0301", temperature = 0.9)

# 初始化模型
def init_model():
    try:
        local_doc_qa.init_cfg(llm_model = llm)
        answer_result_stream_result = local_doc_qa.llm_model_chain(
            {"prompt": "你好", "history": [], "streaming": True})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载"""
        return reply


# 本地知识库问答
local_doc_qa = LocalDocQA()

# 获取知识库列表
def get_vs_list():
    lst_default = ["新建知识库"]    # 默认知识库
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst

# 修改对话模式
def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history

# 刷新向量库列表
def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())

# 添加向量库
def add_vs_name(vs_name, chatbot):
    if vs_name is None or vs_name.strip() == "":
        vs_status = "知识库名称不能为空，请重新填写知识库名称"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    elif vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)

# 删除向量库
def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)

# 修改选择的向量库
def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history, \
               gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])

# 获取选择的向量库
def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    if local_doc_qa.llm_model_chain and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])

# 获取答案
def get_answer(query, vs_path, history, mode, score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k = VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size = CHUNK_SIZE, streaming: bool = STREAMING):
    if mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
            vs_path):
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query = query, vs_path = vs_path, chat_history = history, streaming = streaming):
            yield history, ""
    else:
        answer_result_stream_result = local_doc_qa.llm_model_chain({"prompt": query, "history": history, "streaming": streaming})
        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, ""

# 删除文件
def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = local_doc_qa.delete_file_from_vector_store(vs_path=vs_path,
                                                        filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = local_doc_qa.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files) > 0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中无已上传文件，请先上传文件后，再开始提问。"
    logger.info(",".join(files_to_delete) + vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices = local_doc_qa.list_file_from_vector_store(vs_path), value=[]), chatbot

# 设置网页
block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 550A 智能量子计算机
"""

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as webPage:
    vs_path, file_status = gr.State(os.path.join(KB_ROOT_PATH, get_vs_list()[0], "vector_store") if len(get_vs_list()) > 1 else ""), gr.State("")
    gr.Markdown(webui_title)

    with gr.Row():
        with gr.Column(scale = 12):
            chatbot = gr.Chatbot(elem_id = "550A", show_label=  False).style(height = 750)
            query = gr.Textbox(show_label = False,  placeholder = "Input...").style(container = False)
        with gr.Column(scale = 5):
            mode = gr.Radio(["LLM 对话", "知识库问答"], label = "请选择使用模式", value = "知识库问答")
            knowledge_set = gr.Accordion("知识库设定", visible = False) # 知识库设置
            vs_setting = gr.Accordion("配置知识库")                     # 配置知识库
            mode.change(fn = change_mode, inputs=[mode, chatbot], outputs=[vs_setting, knowledge_set, chatbot])
            with vs_setting:        # 配置知识库交互设计
                vs_refresh = gr.Button("更新已有知识库选项")
                select_vs = gr.Dropdown(get_vs_list(), label="请选择要加载的知识库", interactive = True, value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                vs_name = gr.Textbox(label = "请输入新建知识库名称，当前知识库命名暂不支持中文", lines = 1, interactive = True, visible = True)
                vs_add = gr.Button(value = "添加至知识库选项", visible = True)
                vs_delete = gr.Button("删除本知识库", visible = False)
                load_knowlege_button = gr.Button("重新构建知识库")
                LoadFiles = gr.Column(visible = False)           # 文件上传
                with LoadFiles:
                    gr.Markdown("向知识库中添加文件")
                    sentence_size = gr.Number(value = SENTENCE_SIZE, precision = 0, label = "文本入库分句长度限制", interactive = True, visible = True)
                    with gr.Tab("上传文件"):
                        files = gr.File(label = "添加文件", file_types = ['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"], file_count = "multiple", show_label = False)
                        load_file_button = gr.Button("上传文件并加载知识库")
                    with gr.Tab("上传文件夹"):
                        folder_files = gr.File(label = "添加文件", file_count = "directory", show_label = False)
                        load_folder_button = gr.Button("上传文件夹并加载知识库")
                    with gr.Tab("删除文件"):
                        files_to_delete = gr.CheckboxGroup(choices = [], label = "请从知识库已有文件中选择要删除的文件", interactive = True)
                        delete_file_button = gr.Button("从知识库中删除选中文件")
                vs_refresh.click(fn = refresh_vs_list,
                                inputs = [],
                                outputs = select_vs)
                vs_add.click(fn = add_vs_name,
                            inputs = [vs_name, chatbot],
                            outputs = [select_vs, vs_name, vs_add, LoadFiles, chatbot, vs_delete])
                vs_delete.click(fn = delete_vs,
                                inputs = [select_vs, chatbot],
                                outputs = [select_vs, vs_name, vs_add, LoadFiles, chatbot, vs_delete])
                select_vs.change(fn = change_vs_name_input,
                                inputs = [select_vs, chatbot],
                                outputs=[vs_name, vs_add, LoadFiles, vs_path, chatbot, files_to_delete, vs_delete])
                load_file_button.click(get_vector_store,
                                    show_progress = True,
                                    inputs = [select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                    outputs = [vs_path, files, chatbot, files_to_delete])
                load_folder_button.click(get_vector_store,
                                        show_progress = True,
                                        inputs = [select_vs, folder_files, sentence_size, chatbot, vs_add, vs_add],
                                        outputs = [vs_path, folder_files, chatbot, files_to_delete])
                query.submit(get_answer, inputs = [query, vs_path, chatbot, mode], outputs = [chatbot, query])
                delete_file_button.click(delete_file, show_progress=True,
                                        inputs = [select_vs, files_to_delete, chatbot],
                                        outputs = [files_to_delete, chatbot])

# 设置可生成公网访问
webPage.queue()
webPage.launch(share = True)