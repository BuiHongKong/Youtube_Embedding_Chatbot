import logging
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from youtube_transcripts import YouTube_Transcripts
from langchain_core.embeddings import Embeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class HuggingFaceEmbeddings(Embeddings):
    """Wrapper for Hugging Face Transformers to work with LangChain."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts):
        """Embed a list of texts."""
        # Tokenize input texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform mean pooling to get sentence embeddings
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.numpy()  # Convert to NumPy array for Chroma

    def embed_query(self, text):
        """Embed a single query text."""
        return self.embed_documents([text])[0]

def check_existing_video(vectorstore, url):
    """Check if the video has already been processed."""
    existing = vectorstore.get()
    existing_sources = {meta.get("source") for meta in existing["metadatas"] if meta.get("source")}
    return url in existing_sources

def main():

    processor = YouTube_Transcripts()
    url = "https://youtu.be/Xdv83MFJd7U?si=_Di4q93-JUGKTubQ" # Your url of Youtube video


    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory="Embedded_Database"
    )

    # Check if the video has already been processed
    if check_existing_video(vectorstore, url):
        logger.info("This video has already been processed. Skipping.")
        return

    try:
        # Process the YouTube video and generate the transcript
        explanation = processor.process(url)

        # Split the explanation into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        docs = splitter.create_documents([explanation])

        # Add metadata to each document
        for doc in docs:
            doc.metadata = {"source": url}

        # Store new chunks in the vector store
        vectorstore.add_documents(docs)
        logger.info(f"Stored {len(docs)} new chunks.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()