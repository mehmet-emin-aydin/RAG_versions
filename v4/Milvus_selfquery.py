import os
from dotenv import load_dotenv
load_dotenv()
log_data = [] 
import sys
sys.path.append('../..')

azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_oai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-02-15-preview"
llm_name = "gpt-4o"

from llama_index.llms.azure_openai import AzureOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
llm=AzureOpenAI(deployment_name="gpt-4o", model="gpt-4o", temperature=0, api_key=azure_oai_key, azure_endpoint=azure_oai_endpoint, api_version=api_version)


import io
import os
from docx import Document
import PyPDF2
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Dosya yükleyici bileşen
uploaded_files = st.file_uploader("Upload your documents (for now it only works with files that have .txt, .pdf or .docx extension):", 
                                  accept_multiple_files=True, key="file_uploader")

# Yüklenen dosyaları Document formatında okuyarak işleme
documents = []

for file in uploaded_files:
    file_name = file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension == '.txt':
        text_loader =  TextLoader(file, encoding = 'UTF-8')
        documents.extend(text_loader.load())
    elif file_extension == '.pdf':
        pdf_loader = PyPDFLoader(io.BytesIO(file.read()))
        documents.extend(pdf_loader.load())
    elif file_extension == '.docx':
        docx_loader = Document(io.BytesIO(file.read()))
        documents.extend(docx_loader.load())

# Metinleri parçalara ayırma
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(chunks, sep="\n\n\n")

# metadata_field_info = [
#     AttributeInfo(
#         name="source",
#         description="The the name of the file that the chunk is from.",
#         type="string",
#     ),
#     AttributeInfo(
#         name="page",
#         description="The page from the file that the chunk is from.",
#         type="integer",
#     ),
# ]



# document_content_description = "Lecture notes"

# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectordb,
#     document_content_description,
#     metadata_field_info,
#     verbose=True
# )