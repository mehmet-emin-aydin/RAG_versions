import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
log_data = [] 
import sys
sys.path.append('../..')

azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_oai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"
llm_name = "gpt-4o"

# Global variable for the vectorstore
vectorstore = None

def document_data():
    pdf_path = 'test_files/rag.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    doc = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    text = text_splitter.split_documents(documents=doc)
    print("chunk number: ",len(text))
    # Creating embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    global vectorstore
    vectorstore = FAISS.from_documents(text, embeddings)
    print("Embeddings successfully saved in vector Database and saved locally")

def ask(query, chat_history):
    # ConversationalRetrievalChain 
    qa = ConversationalRetrievalChain.from_llm(
        llm=AzureChatOpenAI(model="gpt-4o", temperature=0, api_key=azure_oai_key, azure_endpoint=azure_oai_endpoint, api_version=api_version),
        retriever=vectorstore.as_retriever()
    )
    
    return qa.invoke({"question": query, "chat_history": chat_history})

if __name__ == '__main__':

    st.header("Multilingual RAG Chatbot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Ensure document_data is called only once
    if vectorstore is None:
        document_data()

    if prompt:
        with st.spinner("Generating......"):
            output = ask(query=prompt, chat_history=st.session_state["chat_history"])
            # Storing the questions, answers and chat history
            st.session_state["chat_answers_history"].append(output['answer'])
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt, output['answer']))

    # Displaying the chat history
    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)
