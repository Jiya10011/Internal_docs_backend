from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def custom_text_loader(file_path):
    return TextLoader(file_path, encoding='utf-8')

def main():
    # Step 1: Load documents with encoding fix
    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=custom_text_loader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")

    # Step 3: Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 4: Save FAISS index
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("faiss_index")
    print("Indexing complete. FAISS index saved to 'faiss_index/'.")

if __name__ == "__main__":
    main()
