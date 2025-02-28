from typing import Set
import os

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
# Dit is de naam van de index in Pinecone!
from ingestion.index_names import index_name_medewerkersgids as index_name

st.header("Medewerkersgids ADGO")


prompt = st.text_input("Prompt", placeholder="Stel hier je vraag..")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(
            index_name=index_name,
            query=prompt,
            chat_history=st.session_state["chat_history"]
        )
        # print(generated_response)

        for doc in generated_response["context"]:
            source = doc.metadata.get("source", "")
            if source != "":
                source = os.path.basename(source)
            page = int(doc.metadata.get("page", 0))

        # print(sources)

        formatted_response = (
            f"{generated_response['answer']} \n *Gevonden in document {source} op ongeveer pagina {page}*"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)
