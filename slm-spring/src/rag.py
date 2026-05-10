"""ChromaDB retrieval. Used at inference to ground generations in canonical examples."""

import os
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class Retriever:
    """Thin wrapper over ChromaDB with our embedding model."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection: str = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        host = host or os.getenv("CHROMA_HOST", "localhost")
        port = port or int(os.getenv("CHROMA_PORT", "8000"))
        collection = collection or os.getenv("CHROMA_COLLECTION", "spring_java_examples")

        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection)
        self.embedder = SentenceTransformer(embed_model)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        artifact_type: Optional[str] = None,
    ) -> List[Dict]:
        """Return top-k examples. Filter by artifact_type ('controller' or 'dto') if given."""
        embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        where = {"artifact_type": artifact_type} if artifact_type else None

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=where,
        )

        out: List[Dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for content, metadata, distance in zip(documents, metadatas, distances):
            out.append({
                "content": content,
                "metadata": metadata,
                "distance": distance,
            })
        return out
