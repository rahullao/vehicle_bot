from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from chromadb.api.models import Collection
from os import listdir, environ
from dotenv import load_dotenv
import chromadb
import logging

from consts import COLLECTION_NAME

load_dotenv()

# get root logger
logger = logging.getLogger(__name__)

chroma_client = chromadb.HttpClient(host="localhost", port="8000")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=environ["AZURE_OPENAI_APIKEY"],
    api_base=environ["AZURE_OPENAI_APIBASE"],
    api_type="azure",
    api_version="2023-05-15",
    model_name="text-embedding-ada-002",
)


def process_and_load_docs():
    filename: str = "2019-audi-q3.pdf"
    loader = PyPDFLoader(file_path="vehicle_manuals/" + filename)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages")

    docs = _create_chunks(raw_documents, filename)

    collection = _create_collection(COLLECTION_NAME)

    _add_docs_to_collection(docs, collection)
    print(f"---- Chunks uploaded to chromadb for file [{filename}] ----")


def _create_collection(collection_name: str) -> Collection:
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=openai_ef
    )
    print(f"Created collection = {collection.name}")
    return collection


def _create_chunks(documents: list[Document], filename: str) -> list[Document]:
    chunk_size = 1000
    chunk_overlap = 100

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = text_splitter.split_documents(documents)
    print(f"Number of chunks for file [{filename}] = {len(docs)}")
    return docs


def _add_docs_to_collection(docs: list[Document], collection: Collection) -> None:
    for index, content in enumerate(docs):
        ids = []
        contents = []

        id = index + 1
        # logger.info("Adding index = {0}".format(id))

        ids.append(str(id))
        contents.append(content.page_content)

        collection.add(documents=contents, ids=ids)


if __name__ == "__main__":
    print("Initiating documents processing")
    process_and_load_docs()
