from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdfs/Alerts.pdf")

documents = loader.load()

vectorstore = Chroma.from_documents(documents, embeddings=OpenAIEmbeddings())

vectorstore.persist()