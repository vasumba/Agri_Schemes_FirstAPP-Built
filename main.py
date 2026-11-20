
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
import hashlib
import json
import os
from dotenv import load_dotenv

from typing import Annotated
from pydantic import WithJsonSchema
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="Agricultural Schemes RAG API")
class QuestionRequest(BaseModel):
    question: str
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

FAISS_DIR = BASE_DIR / "faiss_index"
TRACKING_FILE = BASE_DIR / "processed_files.json"

EMBEDDING_MODEL = "text-embedding-3-small"


@app.get("/")
def home():
    return {
        "message": "Agricultural Schemes FastAPI backend is running"
    }
BinaryUploadFile = Annotated[
    UploadFile,
    WithJsonSchema({"type": "string", "format": "binary"})
]
@app.post("/upload")
async def upload_pdfs(files: list[BinaryUploadFile] = File(...)):

    # Read previously processed PDF hashes
    if TRACKING_FILE.exists():
        processed_files = json.loads(
            TRACKING_FILE.read_text(encoding="utf-8")
        )
    else:
        processed_files = {}

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Load existing FAISS database if available
    if (
        (FAISS_DIR / "index.faiss").exists()
        and (FAISS_DIR / "index.pkl").exists()
    ):
        vector_store = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = None

    new_documents = []
    new_files = {}

    with tempfile.TemporaryDirectory() as temp_dir:

        for uploaded_file in files:

            filename = Path(uploaded_file.filename or "").name

            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are allowed."
                )

            content = await uploaded_file.read()

            # Identify duplicate PDFs using their content hash
            file_hash = hashlib.sha256(content).hexdigest()

            if (
                file_hash in processed_files
                or file_hash in new_files
            ):
                continue

            pdf_path = Path(temp_dir) / (
                file_hash + ".pdf"
            )
            pdf_path.write_bytes(content)

            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = filename

            new_documents.extend(documents)
            new_files[file_hash] = filename

    if not new_documents:
        return {
            "message": "No new PDFs to process",
            "new_files": 0,
            "chunks_added": 0
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(new_documents)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable PDF text found."
        )

    # Create or update FAISS
    if vector_store is None:
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )
    else:
        vector_store.add_documents(chunks)

    # Save the updated database permanently
    vector_store.save_local(str(FAISS_DIR))

    # Update the processed-file record only after saving
    processed_files.update(new_files)

    TRACKING_FILE.write_text(
        json.dumps(processed_files, indent=2),
        encoding="utf-8"
    )

    return {
        "message": "FAISS database updated successfully",
        "new_files": len(new_files),
        "chunks_added": len(chunks),
        "database_folder": "faiss_index"
    }

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    # Check whether the saved FAISS database exists
    if not (
        (FAISS_DIR / "index.faiss").exists()
        and (FAISS_DIR / "index.pkl").exists()
    ):
        raise HTTPException(
            status_code=404,
            detail="No FAISS database found. Upload PDFs first."
        )

    # Use the same embedding model as the upload pipeline
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Reload our own trusted, locally saved FAISS database
    vector_store = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Find the three most relevant PDF chunks
    documents = vector_store.similarity_search(
        question,
        k=3
    )

    if not documents:
        return {
            "answer": "No relevant information was found.",
            "sources": []
        }

    # Prepare retrieved text for the language model
    context_parts = []
    sources = []

    for doc in documents:

        filename = doc.metadata.get(
            "source", "Unknown PDF"
        )

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Source: {filename}, Page: {page}\n"
            f"{doc.page_content}"
        )

        sources.append({
            "filename": filename,
            "page": page
        })

    context = "\n\n".join(context_parts)

    # Generate an answer grounded in the retrieved PDFs
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are an assistant explaining agricultural welfare schemes.

Answer the user's question using only the PDF context below.

If the context does not contain enough information,
say that the answer is not available in the uploaded documents.

Do not invent scheme details or eligibility conditions.

PDF Context:
{context}

Question:
{question}

Provide a clear, simple answer.
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "sources": sources
    }
