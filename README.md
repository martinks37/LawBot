# LawBot 

LawBot is an AI-powered legal chatbot that helps users find relevant legal information using Natural Language Processing (NLP). Instead of relying on keyword matching, it performs semantic search using Sentence Transformers and FAISS to retrieve the most relevant legal answers. The chatbot also supports multilingual responses through automatic translation.

## Features
- Semantic search using Sentence Transformers
- Fast similarity search with FAISS
- AI-powered legal question answering
- Multilingual response translation
- Flask-based web interface
- Lightweight and easy to run locally

## Tech Stack
- Python
- Flask
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- Pandas
- Deep Translator

## How it Works
1. User enters a legal question.
2. The question is converted into an embedding using Sentence Transformers.
3. FAISS retrieves the most relevant legal information from the knowledge base.
4. The chatbot returns the best matching answer.
5. The response can be translated into the user's preferred language.

> **Note:** This project is intended for educational and research purposes only. It does not provide legal advice and should not replace consultation with a qualified legal professional.
