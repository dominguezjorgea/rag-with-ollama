FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-ollama \
    langchain-huggingface \
    langchain-unstructured \
    langchain-chroma \
    chromadb \
    unstructured \
    pypdf \
    python-docx \
    tiktoken \
    sentence-transformers \
    openai \
    fastapi \
    uvicorn


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
