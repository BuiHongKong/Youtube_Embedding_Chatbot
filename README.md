# Youtube_Embedding_Chatbot

YouTube-Based RAG Chatbot using ChromaDB &amp; DeepSeek This project is an end-to-end Retrieval-Augmented Generation (RAG) system built to simulate intelligent Q&amp;A over transcribed YouTube videos. It combines open-source embedding models with a lightweight vector store and integrates with the DeepSeek LLM API to produce rich, contextual responses.

# Key Features

Audio-to-Text Transcription: Downloads YouTube videos and transcribes audio using OpenAI's Whisper model.

HuggingFace Embeddings: Uses all-MiniLM-L6-v2 transformer model for document embedding, ensuring independence from proprietary APIs.

Vector Database with ChromaDB: Embeddings are stored and queried via ChromaDB for fast, local retrieval.

DeepSeek API Integration: Chat interface uses DeepSeek LLM to answer user questions with context retrieved from the vector store.

Terminal Chat Interface: Interactive CLI bot supports continuous conversation with chat history and markdown rendering.

# Purpose & Learning Goals

This project was built to understand how modern LLM systems retrieve and synthesize external knowledge using embeddings and vector search. It demonstrates:

Low-cost alternatives to OpenAI embeddings using HuggingFace models.

Custom embedding workflows and chunked document storage.

Prompt engineering with injected context.

Full RAG loop implementation: ingest → embed → store → retrieve → augment → answer.

