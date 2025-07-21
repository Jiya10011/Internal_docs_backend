from flask import Flask, request, jsonify, render_template
from query_engine import get_retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

# Load retriever
retriever = get_retriever()

# Load HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Define prompt
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for answering questions about internal company documents.
    Use the following context to answer the question:
    
    Context:
    {context}

    Question:
    {question}

    Give a short and accurate answer."""
)

# Chain: retrieve → prompt → llm → output
rag_chain = (
    {"context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])), 
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("question")
    if not query:
        return jsonify({"error": "No question provided"}), 400

    answer = rag_chain.invoke(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
