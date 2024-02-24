from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from os import environ
from typing import Any
import chromadb

from consts import COLLECTION_NAME

load_dotenv()

chroma_client = chromadb.HttpClient(host="localhost", port="8000")

embeddings = AzureOpenAIEmbeddings(
        openai_api_type="azure",
        azure_endpoint=environ["VM_BDO_AZURE_OPENAI_BASE"],
        openai_api_key=environ["VM_BDO_AZURE_OPENAI_KEY"],
        deployment="text-embedding-ada-002"
)

chat_model = AzureChatOpenAI(
    deployment_name="gpt-4",
    model="gpt-4",
    api_key=environ["VM_BDO_AZURE_OPENAI_KEY"],
    azure_endpoint=environ["VM_BDO_AZURE_OPENAI_BASE"],
)


def run_llm(query: str) -> Any:
    chroma_db = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=chroma_db.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain({"query": query})


'''if __name__ == "__main__":
    print(run_llm(query="How to measure outside temperature?"))'''