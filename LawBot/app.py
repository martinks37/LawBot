from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import pandas as pd
from deep_translator import GoogleTranslator

# Load the SentenceTransformer model and FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index('faq_index.faiss')  # Renamed to avoid conflicts
data = pd.read_csv('faq_data.csv')  # Contains questions and answers

# Initialize the text-generation pipeline
generator = pipeline("text-generation", model="distilgpt2")  # Using distilgpt2 for lightweight generation

# Flask app setup
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chatbot.html')

@app.route('/get_greeting', methods=['GET'])
def get_greeting():
    return "Hello, I'm a chatbot! How can I assist you today?"

@app.route("/get", methods=["POST"])
def chat():
    query = request.form["msg"]
    target_language = request.form.get("lang", "en")

    print(f"User Query: {query}, Target Language: {target_language}")  # Debugging
    
    # Retrieve relevant documents
    context_docs = retrieve(query)
    print(f"Retrieved Context: {context_docs}")  # Debugging

    if not context_docs:
        return jsonify({"response": "Sorry, I couldn't find relevant information."})

    # Use the most relevant answer directly
    response = context_docs[0]
    print(f"Direct Response: {response}")  # Debugging

    # Translate response
    translated_response = translate_text(response, target_language)
    print(f"Translated Response: {translated_response}")  # Debugging

    return jsonify({"response": translated_response})

def retrieve(query, k=1):
    """Retrieve top-k relevant documents using FAISS."""
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)  # Use the renamed variable
    results = data.iloc[indices[0]]['Answers'].tolist()
    print(f"Distances: {distances}, Indices: {indices}")  # Debugging
    return results

def translate_text(text, target_language):
    """Translate text to the target language."""
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return "Sorry, I couldn't translate the response."

if __name__ == '__main__':
    app.run(debug=True)