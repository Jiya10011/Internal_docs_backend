from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load the document
loader = TextLoader("internal_docs.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save index
db.save_local("faiss_index")

print("✅ Index built successfully.")
