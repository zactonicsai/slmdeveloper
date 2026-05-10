"""Tests for ChromaDB retrieval. Requires a running ChromaDB seeded via load_chromadb.py.

Skipped in environments without one. CI should run docker compose up -d chromadb
and `python src/load_chromadb.py` before invoking pytest.
"""

import os
import socket

import pytest

from src.rag import Retriever


def _chroma_alive() -> bool:
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _chroma_alive(),
    reason="ChromaDB not reachable; start it with docker compose up -d chromadb",
)


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def test_retrieve_returns_top_k(retriever):
    results = retriever.retrieve("DTO with email validation", top_k=3)
    assert len(results) <= 3
    assert all("content" in r and "metadata" in r for r in results)


def test_artifact_filter_dto(retriever):
    results = retriever.retrieve("validate fields with Lombok", top_k=5, artifact_type="dto")
    assert len(results) > 0
    assert all(r["metadata"]["artifact_type"] == "dto" for r in results)


def test_artifact_filter_controller(retriever):
    results = retriever.retrieve("REST endpoints CRUD", top_k=5, artifact_type="controller")
    assert len(results) > 0
    assert all(r["metadata"]["artifact_type"] == "controller" for r in results)


def test_email_query_pulls_email_dto(retriever):
    results = retriever.retrieve("email validation user account", top_k=5, artifact_type="dto")
    assert any("@Email" in r["content"] for r in results), \
        "Expected an @Email-using DTO in top-5 for email-related query"


def test_distance_is_finite(retriever):
    results = retriever.retrieve("BigDecimal price product", top_k=3)
    assert all(isinstance(r["distance"], (int, float)) for r in results)
