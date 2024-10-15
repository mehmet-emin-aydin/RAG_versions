from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_milvus import Milvus
vector_store= Milvus(
    embeddings,
    connection_args={"uri": "./milvus_demo.db"},
    collection_name='YPZ',
)
retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
print(retriever.invoke("Ailab?"))
