import os
import io
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
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

def ask(query, chat_history):
    # Ensure vectorstore is initialized
    if "vectorstore" not in st.session_state or st.session_state["vectorstore"] is None:
        raise ValueError("Vectorstore is not initialized.")

    system_instruction = "Based on the context provided and chat history, answer the question as an easy-to-understand chat assistant. Ensure that the answer is concise, directly addresses the question, and is in the same language as the question."
    template = (
        f"{system_instruction} "
        "Combine the chat history and follow up question into "
        "a standalone question. Chat History: {{chat_history}}"
        "Follow up question: {{question}}"
    )
    condense_question_prompt = PromptTemplate.from_template(template)
    # ConversationalRetrievalChain 
    qa = ConversationalRetrievalChain.from_llm(
        llm=AzureChatOpenAI(model="gpt-4o", temperature=0, api_key=azure_oai_key, azure_endpoint=azure_oai_endpoint, api_version=api_version),
        retriever=st.session_state["vectorstore"].as_retriever(search_type="mmr", search_kwargs={"k": 3})
    )
    
    return qa.invoke({"question": query, "chat_history": chat_history})

def process_uploaded_files(username, uploaded_files):
    text = ""
    for file in uploaded_files:
        file_name = file.name
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            text += file.read().decode('utf-8')
        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()
        elif file_extension == '.docx':
            doc = Document(io.BytesIO(file.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    # Creating embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": f"{username}:{i}"} for i in range(len(chunks))])
    st.session_state["vectorstore"] = vectorstore
    print("User's documents have been processed and added to the vectorstore.")

if __name__ == '__main__':
    st.header("Multilingual RAG Chatbot")

    if "username" not in st.session_state:
        username = st.text_input("Enter a username (just something that represents you):", key="text_input_username")
        uploaded_files = st.file_uploader("Upload your documents (for now it only works with files that have .txt, .pdf or .docx extension):", accept_multiple_files=True, key="file_uploader")
        if username and uploaded_files:
            process_uploaded_files(username, uploaded_files)
            st.session_state["username"] = username
            st.session_state["vectorstore_initialized"] = True
            st.rerun()
    else:
        if "vectorstore_initialized" not in st.session_state or not st.session_state["vectorstore_initialized"]:
            st.write("Please upload documents first.")
        else:
            prompt = st.chat_input("Enter your questions here")

            if "user_prompt_history" not in st.session_state:
                st.session_state["user_prompt_history"] = []
            if "chat_answers_history" not in st.session_state:
                st.session_state["chat_answers_history"] = []
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

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
