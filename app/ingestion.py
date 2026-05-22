import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_chroma_client():
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client


def _get_collection(client: chromadb.PersistentClient):
    settings = get_settings()
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict]:
    text_parts = []
    metadata = {"source": os.path.basename(pdf_path), "path": pdf_path}

    with pdfplumber.open(pdf_path) as pdf:
        metadata["pages"] = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {page_num}]\n{page_text}")

    return "\n\n".join(text_parts), metadata


def chunk_document(text: str, metadata: dict) -> list[dict]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    result = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{metadata['source']}-{i}".encode()).hexdigest()
        result.append(
            {
                "id": chunk_id,
                "text": chunk,
                "metadata": {**metadata, "chunk_index": i, "total_chunks": len(chunks)},
            }
        )
    return result


def ingest_pdf(pdf_path: str) -> int:
    client = _get_chroma_client()
    collection = _get_collection(client)

    text, metadata = extract_text_from_pdf(pdf_path)
    if not text.strip():
        logger.warning(f"No text extracted from {pdf_path}")
        return 0

    chunks = chunk_document(text, metadata)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    existing = collection.get(ids=ids)
    existing_ids = set(existing["ids"])
    new_chunks = [(i, d, m) for i, d, m in zip(ids, documents, metadatas) if i not in existing_ids]

    if new_chunks:
        new_ids, new_docs, new_metas = zip(*new_chunks)
        collection.add(ids=list(new_ids), documents=list(new_docs), metadatas=list(new_metas))
        logger.info(f"Stored {len(new_chunks)} new chunks from {metadata['source']}")
    else:
        logger.info(f"All chunks from {metadata['source']} already exist — skipping.")

    return len(new_chunks)


def ingest_directory(directory: str) -> tuple[int, int]:
    pdf_files = list(Path(directory).glob("**/*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return 0, 0

    total_docs = 0
    total_chunks = 0
    for pdf_path in pdf_files:
        try:
            chunks = ingest_pdf(str(pdf_path))
            total_docs += 1
            total_chunks += chunks
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path}: {e}")

    return total_docs, total_chunks


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    client = _get_chroma_client()
    collection = _get_collection(client)

    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty.")
        return []

    actual_k = min(top_k, count)
    results = collection.query(
        query_texts=[query],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = max(0.0, 1.0 - dist)
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("chunk_index", 0),
                    "similarity": round(similarity, 4),
                }
            )
    return chunks


def get_collection_count() -> int:
    client = _get_chroma_client()
    collection = _get_collection(client)
    return collection.count()
