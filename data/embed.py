"""
embed.py — NRI Equity RAG: Embed chunks into ChromaDB

Usage:
    python3 embed.py

Reads data/processed/chunks.json, embeds each chunk using
sentence-transformers, and stores vectors in data/vectorstore/

Dependencies:
    pip install chromadb sentence-transformers tqdm
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

BASE        = Path("/Users/jiteshyadav/Desktop/data")
CHUNKS_FILE = BASE / "processed/chunks.json"
VECTOR_DIR  = BASE / "vectorstore"
BATCH_SIZE  = 64        # embed N chunks at a time (RAM-friendly)
MODEL_NAME  = "BAAI/bge-small-en-v1.5"   # fast, good quality, 33M params

# Collection names — one per category for targeted retrieval
COLLECTIONS = {
    "sebi":       "nri_sebi",
    "rbi_fema":   "nri_rbi_fema",
    "tax_dtaa":   "nri_tax_dtaa",
    "structured": "nri_structured",
    "general":    "nri_general",
}


# ── Load chunks ───────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"📦 Loaded {len(chunks)} chunks from {path.name}")
    return chunks


# ── Group by collection ───────────────────────────────────────────────────────

def group_by_collection(chunks: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {name: [] for name in COLLECTIONS.values()}
    for c in chunks:
        cat        = c.get("category", "general")
        collection = COLLECTIONS.get(cat, COLLECTIONS["general"])
        groups[collection].append(c)
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


# ── Embed + store ─────────────────────────────────────────────────────────────

def embed_collection(
    collection_name: str,
    chunks: list[dict],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
):
    print(f"\n📚 Collection: {collection_name} ({len(chunks)} chunks)")

    col = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},   # cosine similarity
    )

    # Skip already-embedded chunks (allows resume on crash)
    existing_ids = set(col.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"  ✅ Already fully embedded — skipping")
        return

    print(f"  → Embedding {len(new_chunks)} new chunks "
          f"(skipping {len(existing_ids)} existing)")

    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE),
                  desc=f"  Batches", unit="batch"):
        batch  = new_chunks[i : i + BATCH_SIZE]
        texts  = [c["text"] for c in batch]
        ids    = [c["chunk_id"] for c in batch]

        # Build metadata dicts (ChromaDB only accepts str/int/float/bool)
        metas  = [
            {
                "source":    c["source"],
                "doc_type":  c["doc_type"],
                "category":  c["category"],
                "page":      int(c["page"]),
                "chunk_idx": int(c["chunk_idx"]),
            }
            for c in batch
        ]

        # Embed
        embeddings = model.encode(
            texts,
            batch_size       = BATCH_SIZE,
            show_progress_bar= False,
            normalize_embeddings = True,   # required for cosine similarity
        ).tolist()

        # Upsert into ChromaDB
        col.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metas,
        )

    print(f"  ✅ Done — collection now has {col.count()} vectors")


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(client: chromadb.PersistentClient, model: SentenceTransformer):
    """Run 3 test queries to verify retrieval works."""
    print("\n🔍 Sanity check — running 3 test queries\n")

    test_queries = [
        ("Can NRI invest in Indian stock market?",  "nri_rbi_fema"),
        ("What is DTAA benefit for NRI in USA?",    "nri_tax_dtaa"),
        ("What is Reliance Industries ISIN?",       "nri_structured"),
    ]

    for query, collection_name in test_queries:
        try:
            col = client.get_collection(collection_name)
        except Exception:
            print(f"  ⚠️  Collection {collection_name} not found — skipping")
            continue

        q_emb = model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = col.query(
            query_embeddings = q_emb,
            n_results        = 2,
            include          = ["documents", "metadatas", "distances"],
        )

        print(f"  Query: \"{query}\"")
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1 - dist, 3)   # cosine similarity
            print(f"    [{score}] {meta['source']} p{meta['page']} "
                  f"| {doc[:120].strip()}...")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # Load model (downloads ~130MB on first run)
    print(f"\n🤖 Loading embedding model: {MODEL_NAME}")
    print("   (Downloads ~130MB on first run — please wait)\n")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✅ Model loaded — embedding dim: {model.get_sentence_embedding_dimension()}")

    # ChromaDB persistent client
    client = chromadb.PersistentClient(
        path     = str(VECTOR_DIR),
        settings = Settings(anonymized_telemetry=False),
    )

    # Load and group chunks
    chunks = load_chunks(CHUNKS_FILE)
    groups = group_by_collection(chunks)

    print(f"\n📊 Distribution across collections:")
    for name, items in groups.items():
        print(f"   {name}: {len(items)} chunks")

    # Embed each collection
    t0 = time.time()
    for collection_name, col_chunks in groups.items():
        embed_collection(collection_name, col_chunks, model, client)

    elapsed = round(time.time() - t0, 1)
    print(f"\n⏱️  Total embedding time: {elapsed}s")

    # Summary
    print("\n📦 Final vector store summary:")
    for col in client.list_collections():
        print(f"   {col.name}: {col.count()} vectors")

    # Sanity check
    sanity_check(client, model)
    print("✅ Vector store ready at:", VECTOR_DIR)


if __name__ == "__main__":
    main()