from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from models.ChatGLM2 import ChatGLM

def main():
    llm = ChatGLM(model_path = "YOUR_PATH")   # 定义模型
    llm.load_model()        # 加载模型
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        else:
            print("\n550A：", end="")
            llm(query)
        print("")

    return 

if __name__ == "__main__":
    main()
