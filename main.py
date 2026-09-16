"""Chat with the documents ingested into Pinecone (hybrid dense + sparse search).

Run: python main.py
Keeps asking questions until you press Ctrl+C.
"""
import os

from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from config import (
    BM25_PARAMS_PATH,
    CHAT_MODEL,
    INDEX_NAME,
    build_embeddings,
)

PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions about the user's own documents.

RULES (in priority order — earlier rules override later ones if they conflict):
1. Use ONLY the CONTEXT below. If the context does not contain the answer, respond with exactly this and nothing else: "I don't know — the provided documents don't cover this." Do not guess.
2. If the context fully answers the question, answer it directly and concisely, quoting specific figures/terms verbatim where relevant.
3. If the context only partially answers the question, answer what is supported and explicitly state what information is missing.
4. If two context chunks conflict, point out the conflict and cite both instead of silently picking one.
5. Never invent sources, numbers, or facts not present in the context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:"""
)


def ask(question: str, retriever, chain):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    answer = chain.invoke({"context": context, "question": question})

    print("\nSources:")
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        location = source + (f" (page {int(page) + 1})" if page is not None else "")
        snippet = d.page_content.strip().replace("\n", " ")
        print(f"\n  [{i}] {location}")
        print(f"      {snippet}")
    print(f"\nAnswer:\n{answer}")


def main():
    if not os.path.exists(BM25_PARAMS_PATH):
        raise SystemExit(
            f"'{BM25_PARAMS_PATH}' not found — run `python ingest.py` first to fit "
            "and save the BM25 sparse encoder."
        )

    bm25_encoder = BM25Encoder()
    bm25_encoder.load(BM25_PARAMS_PATH)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    retriever = PineconeHybridSearchRetriever(
        embeddings=build_embeddings(),
        sparse_encoder=bm25_encoder,
        index=pc.Index(INDEX_NAME),
        top_k=4,
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = PROMPT | llm | StrOutputParser()

    print("Chat with your documents. Press Ctrl+C to quit.")
    try:
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                continue
            ask(question, retriever, chain)
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
