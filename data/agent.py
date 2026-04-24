"""
agent.py — NRI Equity RAG: Query Agent with Knowledge Graph (Ollama version)

Setup:
    1. ollama pull llama3.2
    2. ollama serve
    3. pip install chromadb sentence-transformers ollama networkx
    4. python3 knowledge_graph.py   <- build graph.pkl first
    5. python3 agent.py
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama

try:
    from knowledge_graph import KnowledgeGraph
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

BASE         = Path("/Users/jiteshyadav/Desktop/data")
VECTOR_DIR   = BASE / "vectorstore"
GRAPH_FILE   = BASE / "graph.pkl"
MODEL_NAME   = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"
TOP_K        = 5
MAX_TOKENS   = 1024

ROUTING_RULES = {
    "nri_structured": [
        "isin", "symbol", "nse", "bse", "stock", "share", "equity",
        "company", "sector", "industry", "nifty", "reliance", "tcs",
        "infosys", "hdfc", "icici", "wipro", "bajaj", "listed",
    ],
    "nri_tax_dtaa": [
        "dtaa", "tax", "double taxation", "tds", "capital gains",
        "dividend tax", "treaty", "usa", "uae", "uk", "singapore",
        "canada", "withholding", "exemption", "deduction", "115e",
        "115f", "115g", "115h", "income tax",
    ],
    "nri_rbi_fema": [
        "fema", "rbi", "nre", "nro", "fcnr", "repatriate", "remit",
        "lrs", "liberalised", "invest", "portfolio", "pis", "fdi",
        "foreign exchange", "account", "banking", "transfer",
        "send money", "wire", "swift",
    ],
    "nri_sebi": [
        "sebi", "fpi", "registration", "kyc", "depository", "demat",
        "broker", "circular", "regulation", "compliance", "disclosure",
        "mutual fund", "etf", "derivative", "futures", "options",
    ],
}

SYSTEM_PROMPT = """You are an expert financial advisor specializing in NRI (Non-Resident Indian) investments.

Answer questions based ONLY on the provided context from official documents and Knowledge Graph facts.

Rules:
1. Answer strictly from context and KG facts. Do not hallucinate.
2. Cite documents as [Source: filename, p.X] and KG facts as [KG].
3. If context is insufficient, say so clearly.
4. For tax questions add: "Consult a tax advisor for your specific situation."
5. Use bullet points for multi-part answers.
6. For stock queries include ISIN and sector if available.
"""


@dataclass
class RetrievedChunk:
    text:     str
    source:   str
    page:     int
    score:    float
    category: str


@dataclass
class AgentResponse:
    query:       str
    answer:      str
    sources:     list
    collection:  str
    chunks_used: int
    kg_context:  str = ""


def route_query(query):
    q = query.lower()
    scores = {col: sum(1 for kw in kws if kw in q)
              for col, kws in ROUTING_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "nri_rbi_fema"


def retrieve(query, col_name, embed_model, chroma_client, top_k=TOP_K):
    try:
        col = chroma_client.get_collection(col_name)
    except Exception:
        return []
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(
        query_embeddings=q_emb,
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = [
        RetrievedChunk(
            text=doc, source=meta.get("source", "unknown"),
            page=meta.get("page", 0), score=round(1 - dist, 3),
            category=meta.get("category", ""),
        )
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
    return [c for c in chunks if c.score > 0.35]


def retrieve_multi(query, embed_model, chroma_client, top_k=3):
    all_chunks = []
    for col in ROUTING_RULES:
        all_chunks.extend(retrieve(query, col, embed_model, chroma_client, top_k))
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    return all_chunks[:TOP_K]


def build_context(chunks, kg_context=""):
    rag = "\n\n---\n\n".join(
        f"[{i}] Source: {c.source}, Page {c.page} (score: {c.score})\n{c.text.strip()}"
        for i, c in enumerate(chunks, 1)
    )
    if kg_context:
        return f"=== Knowledge Graph Facts ===\n{kg_context}\n\n=== Document Context ===\n{rag}"
    return rag


def call_ollama(query, context):
    user_msg = (
        f"Context:\n\n{context}\n\n---\n\n"
        f"Question: {query}\n\n"
        f"Answer from context. Cite docs as [Source: filename, p.X] and KG as [KG]."
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        options={"num_predict": MAX_TOKENS, "temperature": 0.1},
    )
    return response["message"]["content"]


class NRIAgent:
    def __init__(self):
        print("Loading NRI Equity Agent...")

        try:
            models = ollama.list()
            names  = [m.model for m in models.models]
            if not any(OLLAMA_MODEL in n for n in names):
                print(f"\n  '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}\n")
                raise SystemExit(1)
        except ollama.ResponseError:
            print("\n  Ollama not running. Run: ollama serve\n")
            raise

        self.embed_model = SentenceTransformer(MODEL_NAME)
        self.chroma = chromadb.PersistentClient(
            path=str(VECTOR_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # Knowledge Graph
        self.kg = None
        if KG_AVAILABLE and GRAPH_FILE.exists():
            try:
                self.kg = KnowledgeGraph(GRAPH_FILE)
                print(f"  KG loaded  : {self.kg.G.number_of_nodes()} nodes")
            except Exception as e:
                print(f"  KG skipped : {e}")
        else:
            print("  KG skipped : run knowledge_graph.py to build graph.pkl")

        print(f"Agent ready  (LLM: {OLLAMA_MODEL})\n")

    def ask(self, query, verbose=True):
        collection = route_query(query)
        if verbose:
            print(f"  Routed to  : {collection}")

        # KG lookup
        kg_context = ""
        if self.kg:
            kg_context = self.kg.query(query)
            if verbose and kg_context:
                print(f"  KG hit     : {kg_context[:80]}...")

        # RAG retrieval
        chunks = retrieve(query, collection, self.embed_model, self.chroma)
        if not chunks or chunks[0].score < 0.45:
            if verbose:
                print("  Low confidence — searching all collections")
            chunks = retrieve_multi(query, self.embed_model, self.chroma)

        if not chunks and not kg_context:
            return AgentResponse(
                query=query, answer="No relevant information found.",
                sources=[], collection=collection, chunks_used=0, kg_context="",
            )

        if verbose:
            top = chunks[0].score if chunks else 0
            print(f"  Chunks     : {len(chunks)} (top: {top})")
            print(f"  Generating answer...")

        context = build_context(chunks, kg_context)
        answer  = call_ollama(query, context)
        sources = list({f"{c.source} p.{c.page}" for c in chunks})

        return AgentResponse(
            query=query, answer=answer, sources=sources,
            collection=collection, chunks_used=len(chunks),
            kg_context=kg_context,
        )


def print_response(resp):
    print("\n" + "=" * 60)
    print(f"Q: {resp.query}")
    print("=" * 60)
    print(resp.answer)
    if resp.kg_context:
        print(f"\n[KG]: {resp.kg_context[:120]}")
    print("\nSources:")
    for s in resp.sources:
        print(f"  - {s}")
    print(f"\n[collection: {resp.collection} | chunks: {resp.chunks_used}]")
    print("=" * 60 + "\n")


def interactive_mode(agent):
    print("NRI Equity Agent — type 'quit' to exit\n")
    samples = [
        "Can an NRI invest in Indian stocks through an NRE account?",
        "What is the DTAA benefit for an NRI in UAE on dividends?",
        "What is the ISIN for Reliance Industries?",
        "What are SEBI KYC requirements for NRI investors?",
        "How much can an NRI repatriate from India per year under LRS?",
    ]
    for i, q in enumerate(samples, 1):
        print(f"  {i}. {q}")
    print()
    while True:
        try:
            query = input("Ask > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        print_response(agent.ask(query))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str)
    args = parser.parse_args()
    agent = NRIAgent()
    if args.query:
        print_response(agent.ask(args.query))
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
