import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX_NAME"]


def Pinecone_Create_Index():
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        while not pc.describe_index(index_name).index.status["ready"]:
            time.sleep(1)

        print("Pinecone Index provisioned")
    else:
        print("Pinecone Index Already Provisioned")


def Pinecone_Delete_Index():
    if index_name in pc.list_indexes().names():
        pc.delete_index(name=index_name)

        print("Pinecone Index Deleted")
    else:
        print("Pinecone Index Had Already been Deleted")


def ingest_docs() -> None:
    base_path = r"D:\ADGO-AI"
    pdf_files = [
        "pdf-docs/Formules.pdf",
        "pdf-docs/Handreiking.pdf",
        "pdf-docs/Opdracht.pdf",
    ]

    raw_documents = []

    for pdf_file in pdf_files:
        full_path = os.path.join(base_path, pdf_file)
        loader = PyPDFLoader(file_path=full_path)
        raw_documents.extend(loader.load())

    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    LangchainPinecone.from_documents(documents, embeddings, index_name=index_name)
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    Pinecone_Delete_Index()
    Pinecone_Create_Index()
    ingest_docs()
