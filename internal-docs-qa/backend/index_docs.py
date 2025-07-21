import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_docs():
    # Load documents from 'docs' folder
    documents = []
    docs_path = "docs"

    for file_name in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file_name)
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings using HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and save FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index created and saved successfully.")

if __name__ == "__main__":
    ingest_docs()
