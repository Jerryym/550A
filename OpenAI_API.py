import os
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from typing import List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# 设置OpenAI模型
os.environ["OPENAI_API_KEY"] = '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'
llm = OpenAI(temperature = 0.9, model_name = "gpt-3.5-turbo-0301")

def loadFiles(directoryPath):
    documentList: List = []
    for root, _, files in os.walk(directoryPath):
        for filename in files:
            if filename.endswith(".pdf"):
                filepath = os.path.join(root, filename)
                print(filename)
                loader = PyPDFLoader(filepath)
                document = loader.load_and_split()
                if  document != None:   documentList.extend(document) 
    return documentList

def main():
    # 加载指定路径文件
    directoryPath = "G:\\LLM_Application\\knowledge_base"
    documentList = loadFiles(directoryPath)

    # 初始化 openai 的 embeddings 对象
    model_path = "G:\\LLM_Application\\models\\embeddings\\text2vec-large-chinese"
    embeddings = HuggingFaceEmbeddings(model_name = model_path, model_kwargs={'device': "cuda"})

    # 创建向量库
    db = FAISS.from_documents(documentList, embeddings)
    db.write
    db.save_local("G:\\vector_store")
    FAISS.load_local("G:\\vector_store", embeddings)
    # docSearch.persist()

    # 创建问答对象
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=False)
    # 进行问答
    # result = qa({"query": "介绍下EICAD，用中文回答"})
    # print(result["query"])
    # print(result["result"])

    result = qa({"query": "请写出设计分段文件(*.FD)格式"})
    print(result["query"])
    print(result["result"])

if __name__ == "__main__":
    main()
