from fastapi import FastAPI, Request
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import os
from pathlib import Path

app = FastAPI()

CHROMA_DIR = "/app/chroma"
DOC_DIR = "/app/data"

@app.on_event("startup")
def load_docs():
    print("🚀 Loading and processing documents...")

    if not os.path.exists(DOC_DIR):
        raise FileNotFoundError(f"{DOC_DIR} does not exist inside the container")

    doc_files = list(Path(DOC_DIR).glob("*.docx"))
    if not doc_files:
        print("📂 No .docx files found — skipping RAG load.")
        return

    all_docs = []
    for file_path in doc_files:
        print(f"📄 Loading file: {file_path}")
        loader = UnstructuredLoader(str(file_path))
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # ✅ Eliminar metadatos no compatibles
    filtered_chunks = filter_complex_metadata(chunks)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(filtered_chunks, embedding, persist_directory=CHROMA_DIR)

    print("✅ Documents loaded and vector store created.")

@app.post("/rag")
async def query_rag(req: Request):
    data = await req.json()
    question = data["question"]

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama2:7b", base_url="http://ollama:11434")

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = chain.invoke({"query": question})
    return {"answer": answer}
