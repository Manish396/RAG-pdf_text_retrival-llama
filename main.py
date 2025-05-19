from fastapi import FastAPI
import uvicorn
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

app = FastAPI()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
llm = ChatOllama(model="llama3", temperature=0)


def chatbot(question: str) -> str:
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content="You are a helpful assistant answering based on provided documents."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)
    return response.content


@app.get("/ask")
def ask(question: str):
    return {"answer": chatbot(question)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)