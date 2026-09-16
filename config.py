"""Shared configuration, read from .env.

Both ingest.py and main.py read from here rather than defining their own
copies, because the embedding model and dimension MUST be identical on both
sides: if ingestion and querying ever drift to different models, the query
vector lands in a different vector space and retrieval degrades silently —
no error, just bad results.
"""
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# --- Pinecone ---
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-homework")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

# --- Models ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o")

# --- Paths ---
BM25_PARAMS_PATH = os.environ.get("BM25_PARAMS_PATH", "bm25_params.json")


def build_embeddings() -> OpenAIEmbeddings:
    """EMBEDDING_DIM drives both the index width and the vectors themselves,
    so the two can't disagree.

    text-embedding-3-* models can emit shorter-than-native vectors via the
    `dimensions` parameter; older models (e.g. ada-002) only emit their
    native width and reject the parameter, so it's passed only where it's
    supported — set EMBEDDING_DIM to that model's native size instead.
    """
    if EMBEDDING_MODEL.startswith("text-embedding-3"):
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM)
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
