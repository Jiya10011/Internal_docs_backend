import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API token from the environment
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

# Load sentence embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vector index
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load Hugging Face LLM with proper configuration
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    temperature=0.5,
    max_length=256,
)

# Load QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Function to query documents
def query_docs(question, k=3):
    docs = db.similarity_search(question, k=k)
    answer = qa_chain.run(input_documents=docs, question=question)
    return answer

# Optional: local test
if __name__ == "__main__":
    query = input("Ask a question: ")
    print("Answer:", query_docs(query))
