from typing import List, Optional, Mapping, Any
from functools import partial

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from transformers import AutoModel, AutoTokenizer

class ChatGLM(LLM):
    model_path: str = None              # 模型路径
    max_length: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history: List = []  
    streaming: bool = True              # 流式输出
    model: object = None
    tokenizer: object = None

    @property
    def _llm_type(self) -> str:
        return "ChatGLM2-6B"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history": [],
            "streaming": self.streaming
        }

    # 回调函数，当用户输入prompt时触发
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, add_history: bool = False) -> str:
        if self.model == None or self.tokenizer == None:
            raise RuntimeError("没有加载模型")
        if self.streaming:
            text_callback = partial(StreamingStdOutCallbackHandler().on_llm_new_token, verbose = True)
            response = self.generateAnswer(prompt, text_callback, add_history = add_history)
        else:
            response = self.generateAnswer(self, prompt, add_history = add_history)
        return response

    # 生成回复
    def generateAnswer(self, prompt, text_callback = None, add_history = True) -> str:
        response: str = ""
        index: int = 0
        if text_callback:
            for i, (response, _) in enumerate(self.model.stream_chat(self.tokenizer, prompt, self.history, max_length = self.max_length, top_p = self.top_p, temperature = self.temperature)):
                if add_history:         # 添加历史记录
                    if i == 0:  self.history += [[prompt, response]]        
                    else:       self.history[-1] = [prompt, response]
                text_callback(response[index:])
                index = len(response)
        else:
            response, _ = self.model.stream_chat(self.tokenizer, prompt, self.history, max_length = self.max_length, top_p = self.top_p, temperature = self.temperature)
            if add_history:             # 添加历史记录
                self.history += [[prompt, response]]
        return response

    # 加载模型
    def load_model(self):
        # 当前存在模型，直接返回
        if self.model != None or self.tokenizer != None:
            return 
        # 加载本地模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code = True, revision = "v1.0")
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code = True, revision = "v1.0").cuda()

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._identifying_params:
                self.k = v