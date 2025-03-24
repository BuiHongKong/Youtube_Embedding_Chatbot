import logging
logging.disable(logging.CRITICAL)

import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from embedding_YoutubeVideoExplanation import HuggingFaceEmbeddings
from rich.markdown import Markdown
from rich.console import Console


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Setup DeepSeek client

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# Initialize Chroma vector store

embedding_function = HuggingFaceEmbeddings()
vectorstore = Chroma(
    persist_directory="Embedded_Database",
    embedding_function=embedding_function
)


system_prompt = (
    "You are a helpful assistant who answers clearly and intelligently. "
    "Use context if provided, otherwise use your own reasoning."
)


console = Console()

chat_history = []
print("RAG ChatBot is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break

    # RAG fetch
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    full_prompt = f"""{system_prompt}

Context:
{context if context else "No relevant documents found."}

Question:
{user_input}
""".strip()

    # Full message stack
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": full_prompt})

    print("\nBot:\n")
    reply = ""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        stream=True
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            reply += delta.content


    console.print(Markdown(reply))
    print()

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})
