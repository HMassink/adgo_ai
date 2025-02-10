import os
import time

from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def Pinecone_Create_Index(pinecone_api_key: str, index_name: str):
    #first delete the index if it exists
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name in pc.list_indexes().names():
        pc.delete_index(name=index_name)

        print(f"Pinecone Index {index_name} Deleted")
    else:
        print(f"Pinecone Index {index_name} Had Already been Deleted")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        while not pc.describe_index(index_name).index.status["ready"]:
            time.sleep(1)

        print(f"Pinecone Index {index_name} provisioned")
    else:
        print(f"Pinecone Index {index_name} Already Provisioned")


def run_llm(index_name : str, query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    chat = ChatOpenAI(model="gpt-4o", verbose=False, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result
