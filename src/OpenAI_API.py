import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'OpenAI_API'
llm = OpenAI(temperature = 0.9, model_name = "text-davinci-003", max_tokens = 1024)
question = "用C++写归并排序"
print(question)
print(llm(question))