import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load data
data = pd.read_csv('chatbot_data.csv')  # Ensure this file contains 'Questions' and 'Answers' columns

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the questions
print("Embedding questions...")
embeddings = embedding_model.encode(data['Questions'].tolist(), show_progress_bar=True)

# Create FAISS index
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, 'faq_index.faiss')
print("FAISS index saved as 'faq_index.faiss'.")

# Save the data for retrieval during runtime
data.to_csv('faq_data.csv', index=False)
print("FAQ data saved as 'faq_data.csv'.")
