from fastapi import FastAPI
import uvicorn
import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

def chatbot(question):
    docs = vectorstore.similarity_search(question, k = 3)
    context = "\n".join([doc.page.content for doc in docs])

    response = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": "You are a helpful assistant answering based on provided documents"},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
    ])

    return response['message']['content']

app = FastAPI()


@app.get("/ask")
def ask(question: str):
    return {"answer": chatbot(question)}

uvicorn.run(app, host="0.0.0.0", port=8000)
