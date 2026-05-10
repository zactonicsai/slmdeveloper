"""Seed ChromaDB with canonical Java reference examples.

Each chunk is stored with metadata so retrieval can be filtered by artifact
type (controller vs DTO) and entity name when relevant.
"""

import os
import re
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "spring_java_examples")
EXAMPLES_DIR = Path(os.getenv("EXAMPLES_DIR", "./examples"))
DATASET_PATH = Path(os.getenv("DATASET_PATH", "./data/dto_controller_dataset.jsonl"))

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def classify_java(content: str, filename: str) -> dict:
    """Heuristic metadata extraction so retrieval can be filtered."""
    artifact_type = "unknown"
    if "@RestController" in content or "@Controller" in content:
        artifact_type = "controller"
    elif filename.endswith("DTO.java") or "@Data" in content:
        artifact_type = "dto"

    # extract entity name: UserDTO -> User, UserController -> User
    stem = Path(filename).stem
    entity = re.sub(r"(DTO|Controller)$", "", stem) or stem

    validation_features = []
    for marker in [
        "@NotNull", "@NotBlank", "@Size", "@Email", "@Pattern",
        "@Min", "@Max", "@Positive", "@PositiveOrZero", "@DecimalMin",
        "@Digits", "@Past", "@PastOrPresent", "@Future", "@Valid",
    ]:
        if marker in content:
            validation_features.append(marker.lstrip("@"))

    return {
        "artifact_type": artifact_type,
        "entity": entity,
        "validation_features": ",".join(validation_features),
        "uses_lombok": "@Data" in content or "@Builder" in content or "@RequiredArgsConstructor" in content,
    }


def main() -> None:
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )

    # reset collection so reloads are deterministic
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []

    # 1) load standalone .java reference files
    java_files = sorted(EXAMPLES_DIR.glob("*.java"))
    print(f"Found {len(java_files)} reference Java files")
    for jf in java_files:
        content = jf.read_text(encoding="utf-8")
        meta = classify_java(content, jf.name)
        meta["source"] = f"examples/{jf.name}"
        documents.append(content)
        metadatas.append(meta)
        ids.append(f"java::{jf.name}")

    # 2) load training dataset rows as additional retrievable references
    if DATASET_PATH.exists():
        print(f"Loading dataset rows from {DATASET_PATH}")
        with DATASET_PATH.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                output = row["output"]
                meta = classify_java(output, f"dataset_{i}.java")
                meta["source"] = f"dataset:{i}"
                meta["instruction"] = row["instruction"][:500]
                documents.append(output)
                metadatas.append(meta)
                ids.append(f"dataset::{i}")

    print(f"Embedding {len(documents)} documents...")
    embeddings = embedder.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Loaded {collection.count()} documents into '{COLLECTION}'")


if __name__ == "__main__":
    main()
