import streamlit as st
import PyPDF2
from langchain.chains import StuffDocumentsChain,MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
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

def process_uploaded_files(uploaded_files):
    pdf_path = '/home/mehmet/Desktop/Softtech/RAG/RAG_versions/test_files/rag.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    # First, we specify the LLMChain to use for mapping each document to an individual summary:
    llm=AzureChatOpenAI(model="gpt-4o", temperature=0, api_key=azure_oai_key, azure_endpoint=azure_oai_endpoint, api_version=api_version)

    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)


    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)


    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )


    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    result = map_reduce_chain.invoke(split_docs)

    return result["output_text"]





def main():

    st.set_page_config(page_title="PDF Summarizer")
    st.title("PDF Summarazing App")
    st.write("Summarize your pdf files in just a few seconds.")
    # Displaying a description
    st.divider() # Inserting a divider for better layout
    # Creating a file uploader widget to upload PDF files
    pdf = st. file_uploader ('Upload your PDF Document', type = "pdf")
    # Creating a button for users to submit their PDF

    submit = st.button("Generate Summary")
    if submit and pdf is not None:
        summary = process_uploaded_files(pdf)
        st.subheader("Summary of the file:")
        st.write(summary)


if __name__ == '__main__':
    main()