import streamlit as st

def main():
    st.header("**Experiment** RAG chatbot ADGO")

    st.write("""
    Als experiment onderzoeken we hoe een chatbot getraind kan worden op ADGO documenten.
    Links zie je verschillende chatbots die getraind zijn op verschillende datasets.
    Bij vragen of opmerkingen kun je contact opnemen met Henk Massink
    
    **NB** De getrainde data bevatten geen privacy gevoelige data  
    De gevectoriseerde data wordt opgeslagen in Pinecone, een service van AWS  
    Als LLM wordt chatbot GPT-4o gebruikt van OpenAI  
    
    """)

if __name__ == "__main__":
    main()
