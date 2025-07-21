from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the FAISS index with embeddings
db = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Set up Hugging Face pipeline
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    model_kwargs={"max_length": 512}
)

# Wrap the pipeline in LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Load the QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")


@app.route("/query", methods=["POST"])
def query():
    try:
        question = request.json["question"]
        print(f"🧠 User Question: {question}")

        docs = db.similarity_search(question)
        print(f"🔍 Retrieved Docs: {docs}")

        if not docs:
            return jsonify({"answer": "No relevant information found."})

        # Get result from chain
        result = qa_chain(
            {"input_documents": docs, "question": question},
            return_only_outputs=True
        )
        print(f"✅ Answer: {result}")

        # Build safe JSON response
        response = {
            "question": question,
            "answer": result['output_text'],
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return jsonify({"answer": "No answer found due to an error."}), 500


if __name__ == "__main__":
    app.run(debug=True)
