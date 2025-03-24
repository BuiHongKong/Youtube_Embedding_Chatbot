from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Define the embedding function (must match the one used when creating the ChromaDB)
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts):
        """Embed a list of texts."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.numpy()

    def embed_query(self, text):
        """Embed a single query text."""
        return self.embed_documents([text])[0]

# Initialize Chroma vector store
persist_directory = "Embedded_Database"  # Path to your ChromaDB file
embedding_function = HuggingFaceEmbeddings()
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory=persist_directory
)

# Inspect the contents of the vector store
def inspect_chromadb(vectorstore):
    """Print the documents, embeddings, and metadata stored in ChromaDB."""
    # Get all documents and metadata
    collection = vectorstore.get()
    documents = collection["documents"]
    metadatas = collection["metadatas"]
    ids = collection["ids"]

    # Print the contents
    print("=== Documents ===")
    for i, (doc, meta, id) in enumerate(zip(documents, metadatas, ids)):
        print(f"ID: {id}")
        print(f"Metadata: {meta}")
        print(f"Document: {doc}")
        print("-" * 50)

    # Get embeddings (optional, can be large)
    embeddings = vectorstore._collection.get(include=["embeddings"])["embeddings"]
    if embeddings is not None and len(embeddings) > 0:  # Check if embeddings exist and are not empty
        print("\n=== Embeddings ===")
        for i, embedding in enumerate(embeddings):
            print(f"Embedding {i + 1}: Shape = {np.array(embedding).shape}")

# Run the inspection
inspect_chromadb(vectorstore)