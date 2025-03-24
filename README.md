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

# Models Used

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Transcription: openai/whisper (or can switch to local whisper)

LLM: deepseek-chat via DeepSeek API

Vector Store: ChromaDB

# Features

Fully local embeddings and vector store

Fast similarity search with Chroma

DeepSeek LLM for intelligent responses

Persistent chat history in terminal session

Modular Python scripts

# Modules

# 1.embedding_YoutubeVideoExplanation.py

Handles ingestion, transcription, summarization, chunking, and embedding into ChromaDB.

Key Steps:

Downloads YouTube audio using yt_dlp

Transcribes using whisper

Explains transcript with DeepSeek

Splits long text into chunks

Embeds chunks using HuggingFace sentence-transformer

Stores chunks into ChromaDB with metadata

Result: Your Embedded_Database/ folder now contains all processed chunks and their semantic vectors.

# 2.inpsect_vector_database.py

Allows inspection of what's inside ChromaDB.

Features:

Loads Chroma with the correct embedding function

Prints all:

Document texts

Metadata

Embedding shapes

Useful for checking if embeddings were stored correctly.

# 3.chatbot_with_RAG.py

Runs a terminal chatbot using:

Chroma for vector search

DeepSeek for chat responses

HuggingFace for embedding user queries

Rich for terminal markdown rendering

Features:

Retrieves top-k relevant documents from Chroma

Sends them as context to DeepSeek

Streams reply

Maintains session with full chat history
