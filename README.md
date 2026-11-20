
# Agricultural Schemes Multi-PDF RAG Assistant

## Project Overview

An AI-powered application that answers questions about agricultural schemes using information retrieved from uploaded PDF documents.

The application uses FastAPI, Streamlit, LangChain, OpenAI, and FAISS.

## Features

- Upload multiple agricultural scheme PDFs
- Extract and split PDF text into chunks
- Generate embeddings using OpenAI
- Store and retrieve document embeddings using FAISS
- Ask questions about the uploaded documents
- Generate answers grounded in retrieved PDF content
- Display source PDF filenames and page numbers
- Save the FAISS index for reuse
- Skip PDFs already processed

## Technology Stack

- Python
- FastAPI — Backend API
- Streamlit — User interface
- LangChain — RAG pipeline
- FAISS — Vector database
- OpenAI — Embeddings and answer generation
- PyPDF — PDF document processing

## Installation

1. Clone or download this repository.
2. Open the project folder.
3. Create and activate a Python virtual environment.
4. Install the required packages:

   pip install -r requirements.txt

5. Create a `.env` file containing:

   OPENAI_API_KEY=your_api_key_here

## Run the Application

Start the FastAPI backend:

    python -m uvicorn main:app --host 127.0.0.1 --port 8001

Open another terminal and start Streamlit:

    python -m streamlit run app.py

Open the Streamlit application:

http://localhost:8501

## How It Works

1. Upload agricultural scheme PDFs through Streamlit.
2. FastAPI extracts the text and creates chunks.
3. OpenAI generates embeddings.
4. FAISS stores the embeddings locally.
5. The user enters a question.
6. The system retrieves relevant PDF chunks.
7. OpenAI generates an answer from the retrieved context.
8. The application displays the answer and document sources.

## Project Status

The application has been tested locally with PDF uploads, question answering, and source references.

## Security

The OpenAI API key is stored in a local `.env` file, which is excluded from GitHub using `.gitignore`.
