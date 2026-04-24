🇮🇳 NRI Equity Agent

An end-to-end AI-powered RAG system for Non-Resident Indian investment queries — combining semantic retrieval over official regulatory documents with a structured knowledge graph and a local LLM.


What this project does
NRIs investing in India face a uniquely complex problem: their questions span four completely different knowledge domains simultaneously — SEBI regulations, RBI/FEMA rules, bilateral tax treaties (DTAA), and live stock market data. No single document or search engine answers them well.
This project builds an AI agent that retrieves answers from 19 official government documents, augments retrieval with a structured knowledge graph of 2,404 nodes, uses a fully local LLM (no API costs), and includes a custom evaluation harness measuring system performance.

**Evaluation Results
Metric                  Score**
Retrieval Relevance     5.0/5.0
Faithfulness            4.0/5.0
Routing Accuracy        100% across all 4 collections
KG Coverage             75% of queries enriched with KG facts
Avg Latency             21s per query (local Llama 3.2)

**Architecture**
User Query
    │
    ▼
Query Router (keyword-scored collection selector)
    │
    ├─────────────────────┐
    ▼                     ▼
ChromaDB (4 colls)    Knowledge Graph
RAG chunks            NetworkX, 2404 nodes
    │                     │
    └──────────┬──────────┘
               │  merged context
               ▼
          Llama 3.2
          via Ollama
          (local, free)
               │
               ▼
    Cited Answer + Sources

**Why each component was built**
**ingest.py** — Document-aware chunking pipeline
Different document types need different chunking strategies. A FAQ should split by Q&A pairs. A FEMA Master Direction needs sliding window chunking with overlap to preserve legal context across chunk boundaries. The pipeline auto-detects document type from filename and content, routes to the right chunker, and tags every chunk with source, doc_type, category, page, and a content hash for deduplication.
Result: 3,714 chunks from 19 PDFs and 3 NSE CSVs.
**embed.py** — Vector store builder
Chunks are embedded using BAAI/bge-small-en-v1.5 (33M params, free, best-in-class small model) and stored in 4 separate ChromaDB collections — one per knowledge domain. Keeping collections separate lets the query router target only the relevant domain, improving precision and cutting retrieval time.
Result: 3,714 vectors across 4 collections, persisted to disk.
**knowledge_graph.py** — Structured knowledge graph
RAG retrieves text well but fails on relational queries. "Which IT sector stocks can an NRI invest in?" requires traversing Stock→Sector relationships — a vector search cannot do this reliably. The knowledge graph fills this gap.
Built with NetworkX from three sources: NSE CSVs give 2,364 Stock nodes and 20 Sector nodes. Hardcoded DTAA data gives 5 Country→TaxTreaty relationships with exact TDS rates. Hardcoded regulatory knowledge gives NRE/NRO/FCNR account nodes linked to investment types and regulatory bodies.
Result: 2,404 nodes, 1,037 edges, saved as graph.pkl.
**agent.py** — RAG + KG query agent
Four stages: (1) keyword-scored router picks the best ChromaDB collection, (2) semantic search returns top-5 chunks filtered by minimum relevance score, (3) KG intent detection runs in parallel for structured facts, (4) KG facts are prepended to RAG chunks in the context window and Llama 3.2 generates a cited answer. Answers cite both document sources and KG facts.
**eval.py** — Custom evaluation harness
25 hand-crafted test questions across all 4 knowledge domains. For each question, the agent runs and an LLM-as-judge (same local Llama 3.2) scores two metrics on a 1–5 scale: retrieval relevance (did ChromaDB fetch the right chunks?) and faithfulness (does the answer stay within context?). KG coverage and routing accuracy are also measured. Results saved to eval_results.csv and eval_summary.txt.
**app.py** — Streamlit chat UI
Lazy-loads the agent on first button click. Sidebar shows knowledge base stats and 7 clickable sample queries. Each response shows cited sources and KG metadata.

**Data Sources**
SEBI (circulars)

FPI Regulations 2019
NRI Position Limits in ETDs (2025)
Geo-tagging relaxation for NRI re-KYC (2025)
NRIs/OCIs in SEBI-registered FPIs (2024)
SWAGAT-FI Framework for FPIs (2026)
Simplified FPI Registration (2024)
Streamlining FPI Onboarding (2023)
Operational Guidelines for FPIs (2019)

RBI / FEMA (Master Directions)

Foreign Investment in India (updated Jan 2025)
Overseas Investment (updated Apr 2026)
Borrowing and Lending — NRI (updated Feb 2026)
Liberalised Remittance Scheme (updated Sep 2024)
Deposits and Accounts — NRE/NRO/FCNR (updated Oct 2025)
Non-resident Investment in Debt Instruments (updated Apr 2026)

DTAA Tax Treaties (4 countries)

UAE — 0% dividend TDS, 0% capital gains (most favorable)
UK — 15% dividend, 10% capital gains
Singapore — 15% dividend, 0% capital gains
Canada — 25% dividend, 15% capital gains

NSE Structured Data (3 files)

nse_equity_list.csv — ~2,000 stocks with Symbol and ISIN
nifty50_constituents.csv — Nifty 50 with sector and weightage
sector_mapping.csv — Company to sector mapping


**Project Structure**
nri-equity-agent/
├── ingest.py              # PDF chunking pipeline
├── embed.py               # Vector store builder
├── knowledge_graph.py     # NetworkX knowledge graph
├── agent.py               # RAG + KG query agent
├── eval.py                # Evaluation harness
├── app.py                 # Streamlit chat UI
├── requirements.txt
├── eval_results.csv       # Per-question eval scores
└── eval_summary.txt       # Aggregate eval report

**Setup**
1. Install dependencies
bashpip install -r requirements.txt
2. Add source documents
data/raw_pdfs/sebi/
data/raw_pdfs/rbi_fema/
data/raw_pdfs/tax_dtaa/
data/structured/
3. Run ingestion pipeline
bashpython3 ingest.py
python3 embed.py
python3 knowledge_graph.py
4. Start Ollama
bashollama pull llama3.2
ollama serve
5. Launch
bashstreamlit run app.py
# or CLI:
python3 agent.py --query "Can NRI invest in Indian stocks via NRE account?"
6. Run evaluation
bashpython3 eval.py --quick    # 5 questions
python3 eval.py            # full 25 questions
python3 eval.py --show     # print previous results

**Sample Queries**

Can an NRI invest in Indian stocks through an NRE account?
What is the DTAA benefit for an NRI in UAE on Indian dividends?
What is the ISIN for Reliance Industries?
What are SEBI KYC requirements for NRI investors?
How much can an NRI repatriate from India per year under LRS?
What is the difference between NRE and NRO accounts?
Which IT sector stocks are listed on NSE?
What is TDS on dividends for NRI under India-UK DTAA?


**Tech Stack**
ComponentToolWhyPDF parsingpdfplumberReliable text + table extractionEmbeddingsBAAI/bge-small-en-v1.5Best small embedding model, freeVector storeChromaDBPersistent, no infrastructure neededKnowledge graphNetworkXPure Python, no database neededLLMLlama 3.2 via OllamaFree, local, no API costUIStreamlitFast to build, clean interfaceEvaluationCustom LLM-as-judgeRAGAS-style but fully custom

Author
Jitesh Kumar Yadav
Former Researcher, Reserve Bank of India (DSIM)
Email me: jitesh3777yadav@gmail.com
