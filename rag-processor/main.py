from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.utils import filter_complex_metadata

import os
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        if file_path.name.startswith("~$"):
            continue
        print(f"📄 Loading file: {file_path}")
        loader = UnstructuredLoader(str(file_path))
        docs = loader.load()
        filtered_docs = filter_complex_metadata(docs)
        all_docs.extend(filtered_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)
    print("✅ Documents loaded and vector store created.")

@app.post("/rag")
async def query_rag(req: Request):
    data = await req.json()
    question = data["question"]

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(
        model="gouranshitera/bloom-1b1:latest",
        base_url="http://ollama:11434"
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Eres un experto en inteligencia artificial y adopción de IA.
Responde en español de forma clara, breve y precisa, usando la información de los documentos.

Contexto:
{context}

Pregunta:
{question}
""".strip()
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"},
        return_source_documents=False,
    )

    answer = chain.invoke({"query": question})
    return {"answer": answer}

