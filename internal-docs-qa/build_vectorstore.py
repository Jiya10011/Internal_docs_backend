
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load documents from your internal docs file
loader = TextLoader("docs/internal_docs.txt")  # make sure this file exists
documents = loader.load()

# Split long documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text to vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save to folder
db.save_local("faiss_index")
print("✅ Vectorstore saved to 'faiss_index'")
