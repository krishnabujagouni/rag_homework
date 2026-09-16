"""Page chunking + hybrid ingestion.

Loads PDFs from DATA_DIR and treats each page as one chunk (PyPDFLoader
already paginates, so no splitting or overlap is applied). Each chunk is
embedded twice — dense via OpenAI for semantic similarity, sparse via BM25
for keyword match — and both vectors are upserted to a Pinecone dotproduct
index, which is what lets main.py blend the two at query time.

Run: python ingest.py
"""
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from config import (
    BM25_PARAMS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    build_embeddings,
)

DATA_DIR = "data"


def load_pages():
    pages = []
    for path in sorted(Path(DATA_DIR).rglob("*.pdf")):
        pages.extend(PyPDFLoader(str(path)).load())
    return pages


def ensure_hybrid_index(pc: Pinecone):
    """Create the index, or recreate it if its metric or dimension no longer
    match the config — hybrid search requires 'dotproduct', and neither can
    be changed in place.

    delete_index() and create_index() both block until settled, so it's safe
    to upsert as soon as this returns.
    """
    index = {i.name: i for i in pc.list_indexes()}.get(INDEX_NAME)

    if index is not None:
        mismatch = None
        if index.metric != "dotproduct":
            mismatch = f"metric='{index.metric}' (hybrid search needs 'dotproduct')"
        elif index.dimension != EMBEDDING_DIM:
            mismatch = f"dimension={index.dimension} ({EMBEDDING_MODEL} needs {EMBEDDING_DIM})"
        if mismatch:
            print(f"Index '{INDEX_NAME}' has {mismatch} — recreating ...")
            pc.delete_index(INDEX_NAME)
            index = None

    if index is None:
        print(f"Creating index '{INDEX_NAME}' (dotproduct, dim={EMBEDDING_DIM}) ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        # Every run re-ingests from scratch, so drop stale vectors instead of
        # letting new IDs pile up alongside them.
        print(f"Clearing existing vectors from '{INDEX_NAME}' ...")
        try:
            pc.Index(INDEX_NAME).delete(delete_all=True)
        except Exception as e:
            print(f"  (nothing to clear: {e})")


def main():
    pages = load_pages()
    if not pages:
        raise SystemExit(f"No .pdf files found in {DATA_DIR}/")

    # Drop pages with no extractable text (scanned/image-only): BM25 encodes
    # those to an empty sparse vector, which Pinecone rejects — failing the
    # entire upsert batch.
    chunks = [p for p in pages if p.page_content.strip()]
    blank = len(pages) - len(chunks)
    if blank:
        print(f"Skipped {blank} page(s) with no extractable text.")
    if not chunks:
        raise SystemExit("Every page was blank — nothing to ingest.")
    print(f"Page chunking produced {len(chunks)} chunk(s) from {len(pages)} page(s).")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_hybrid_index(pc)

    print(f"Fitting BM25 sparse encoder -> {BM25_PARAMS_PATH} ...")
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(texts)
    bm25_encoder.dump(BM25_PARAMS_PATH)

    retriever = PineconeHybridSearchRetriever(
        embeddings=build_embeddings(),
        sparse_encoder=bm25_encoder,
        index=pc.Index(INDEX_NAME),
    )
    retriever.add_texts(texts=texts, metadatas=metadatas)
    print(f"Upserted {len(chunks)} chunks (dense + sparse) into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
