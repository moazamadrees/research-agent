import pytest

from app.ingestion import chunk_document, extract_text_from_pdf


class TestChunkDocument:
    def test_basic_chunking(self):
        text = "word " * 500
        metadata = {"source": "test.pdf", "pages": 1}
        chunks = chunk_document(text, metadata)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == "test.pdf"

    def test_short_text_single_chunk(self):
        text = "This is a short text."
        metadata = {"source": "short.pdf", "pages": 1}
        chunks = chunk_document(text, metadata)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_chunk_ids_are_unique(self):
        text = "sentence. " * 200
        metadata = {"source": "unique.pdf", "pages": 2}
        chunks = chunk_document(text, metadata)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self):
        text = "content " * 100
        metadata = {"source": "meta.pdf", "pages": 5, "extra_field": "value"}
        chunks = chunk_document(text, metadata)
        for chunk in chunks:
            assert chunk["metadata"]["extra_field"] == "value"
            assert "chunk_index" in chunk["metadata"]
            assert "total_chunks" in chunk["metadata"]
