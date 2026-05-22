import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import run_research_pipeline
from app.config import get_settings
from app.ingestion import get_collection_count, ingest_directory, ingest_pdf
from app.models import HealthResponse, IngestResponse, ResearchReport, ResearchRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    docs_dir = "./documents"
    if os.path.isdir(docs_dir):
        pdf_count = len(list(__import__("pathlib").Path(docs_dir).glob("**/*.pdf")))
        if pdf_count > 0:
            logger.info(f"Auto-ingesting {pdf_count} PDFs from {docs_dir} …")
            docs, chunks = ingest_directory(docs_dir)
            logger.info(f"Startup ingestion complete: {docs} docs, {chunks} chunks")
    yield
    logger.info("ResearchAgent shutting down.")


app = FastAPI(
    title="ResearchAgent API",
    description="Multi-agent AI system for deep research using RAG and LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    settings = get_settings()
    try:
        doc_count = get_collection_count()
    except Exception:
        doc_count = -1
    return HealthResponse(
        status="ok",
        chroma_documents=doc_count,
        model=settings.llm_model,
    )


@app.post("/research", response_model=ResearchReport, tags=["Research"])
async def research(request: ResearchRequest):
    try:
        report = await run_research_pipeline(
            question=request.question,
            top_k=request.top_k,
        )
        return report
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {str(e)}")


@app.post("/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_upload(files: list[UploadFile] = File(...)):
    settings = get_settings()
    total_chunks = 0
    docs_processed = 0

    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a PDF.")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = ingest_pdf(tmp_path)
            total_chunks += chunks
            docs_processed += 1
        except Exception as e:
            logger.error(f"Ingest error for {upload.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to ingest {upload.filename}: {e}")
        finally:
            os.unlink(tmp_path)

    return IngestResponse(
        status="success",
        documents_processed=docs_processed,
        chunks_stored=total_chunks,
        collection=settings.chroma_collection_name,
    )


@app.post("/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_dir(directory: str = "./documents"):
    settings = get_settings()
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    try:
        docs, chunks = ingest_directory(directory)
    except Exception as e:
        logger.error(f"Directory ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        status="success",
        documents_processed=docs,
        chunks_stored=chunks,
        collection=settings.chroma_collection_name,
    )
