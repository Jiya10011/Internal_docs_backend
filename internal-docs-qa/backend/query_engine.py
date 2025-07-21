from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    # Load same embedding model used during indexing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load FAISS index from disk
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Return retriever with top 3 matches
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever
