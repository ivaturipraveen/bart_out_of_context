import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

def load_faiss_and_data(embedding_file, faiss_index_file, data_file):
    embeddings = np.load(embedding_file)
    index = faiss.read_index(faiss_index_file)
    with open(data_file, 'r') as f:
        data = json.load(f)
    return embeddings, index, data["texts"], data["metadata"]

def rerank_results(query, top_texts):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, text) for text in top_texts])
    sorted_indices = np.argsort(scores)[::-1]
    reranked_texts = [top_texts[i] for i in sorted_indices]
    return reranked_texts, scores[sorted_indices]

def generate_response_with_openai(query, top_texts, max_texts=3):
    top_texts = top_texts[:max_texts]
    prompt = (
        f"You are an intelligent assistant trained to provide structured, concise, and contextually accurate answers. "
        f"Based on the following information, summarize the given texts effectively and provide an answer to the query: '{query}'.\n\n"
        + "\n".join([f"{i+1}. {text}" for i, text in enumerate(top_texts)]) +
        "\n\nWhen the information is long or detailed, organize your response in a clear structure. "
        "Ensure the response is directly relevant to the query and avoids introducing any information not present in the provided texts."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return None

def generate_generic_response(query):
    prompt = (
        f"You are an intelligent assistant. Respond to the user's query: '{query}'. "
        f"If it is unrelated to the dataset, provide a helpful and general answer."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return "*Error:* Unable to process the query. Please try again later."

@app.route('/query', methods=['POST'])
def query():
    request_data = request.get_json()
    query = request_data.get('query', '')

    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Load FAISS index and data
    output_dir = './embeddings_output'
    embedding_file = f"{output_dir}/combined_embeddings.npy"
    faiss_index_file = f"{output_dir}/combined_faiss.index"
    data_file = f"{output_dir}/processed_data.json"
    embeddings, index, texts, metadata = load_faiss_and_data(embedding_file, faiss_index_file, data_file)

    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Retrieve top results from FAISS
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, 3)
    top_texts = [texts[idx] for idx in indices[0]]

    # Rerank and generate responses
    reranked_texts, _ = rerank_results(query, top_texts)
    faiss_response = generate_response_with_openai(query, reranked_texts)
    generic_response = generate_generic_response(query)

    # Return the final response
    response = {
        "query": query,
        "faiss_response": faiss_response,
        "generic_response": generic_response
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)