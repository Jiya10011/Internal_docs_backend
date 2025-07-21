# generate_vectorstore.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Load your documents (from "docs" folder)
loader = DirectoryLoader("./docs", glob="**/*.txt")  # or use .pdf if PDFs
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert docs to vectorstore
vectorstore = FAISS.from_documents(docs, embedding)

# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

print("✅ vectorstore.pkl saved!")
