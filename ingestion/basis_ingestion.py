import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from dotenv import load_dotenv

from backend.core import Pinecone_Create_Index

load_dotenv()

pinecone_key = os.environ["PINECONE_API_KEY"]
index_name = os.environ["PINECONE_INDEX_NAME"]

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
    #Dit is de grote hamvraag hoe groot of klein moet de chunk_size zijn?
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    LangchainPinecone.from_documents(documents, embeddings, index_name=index_name)
    print(f"****** Added to Pinecone vectorstore vectors {index_name} ******")

# Deze run je om de documenten te vectoriseren en in Pinecone te zetten
if __name__ == "__main__":
    Pinecone_Create_Index(pinecone_key, index_name)
    ingest_docs()
