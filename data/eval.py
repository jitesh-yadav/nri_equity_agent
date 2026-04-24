"""
eval.py — NRI Equity RAG: Evaluation Harness

Usage:
    python3 eval.py                  # run full eval (25 questions)
    python3 eval.py --quick          # run 5 questions only
    python3 eval.py --show           # print results from last run

Metrics:
    - Retrieval Relevance : did we fetch the right chunks? (score 1-5)
    - Faithfulness        : does answer stay within context? (score 1-5)
    - Coverage            : did KG add info RAG alone missed? (yes/no + note)

Saves:
    data/eval_results.csv   <- per-question scores
    data/eval_summary.txt   <- aggregate report

Dependencies:
    pip install ollama chromadb sentence-transformers pandas
"""

import csv
import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import ollama
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from agent import NRIAgent

BASE         = Path("/Users/jiteshyadav/Desktop/data")
RESULTS_FILE = BASE / "eval_results.csv"
SUMMARY_FILE = BASE / "eval_summary.txt"
JUDGE_MODEL  = "llama3.2"   # same model as agent — free, local

# ── 25 Test questions ─────────────────────────────────────────────────────────
# Covers all 4 collections + KG. Hand-crafted to have clear right answers.

TEST_QUESTIONS = [
    # ── SEBI (5) ──────────────────────────────────────────────────────────────
    {
        "id": "S01",
        "category": "sebi",
        "question": "What are SEBI KYC requirements for NRI investors?",
        "expected_keywords": ["kyc", "geo", "tagging", "onboarding", "re-kyc"],
        "expected_source": "sebi",
    },
    {
        "id": "S02",
        "category": "sebi",
        "question": "What is the SEBI FPI registration process for NRIs?",
        "expected_keywords": ["fpi", "registration", "category", "sebi"],
        "expected_source": "sebi",
    },
    {
        "id": "S03",
        "category": "sebi",
        "question": "Can NRI participate in exchange traded derivatives in India?",
        "expected_keywords": ["derivative", "nri", "exchange", "etcd", "position"],
        "expected_source": "sebi",
    },
    {
        "id": "S04",
        "category": "sebi",
        "question": "What is the SWAGAT-FI framework for foreign investors?",
        "expected_keywords": ["swagat", "fpi", "trusted", "single window"],
        "expected_source": "sebi",
    },
    {
        "id": "S05",
        "category": "sebi",
        "question": "What disclosures are required by Foreign Portfolio Investors?",
        "expected_keywords": ["disclosure", "fpi", "criteria", "objective"],
        "expected_source": "sebi",
    },

    # ── RBI / FEMA (7) ────────────────────────────────────────────────────────
    {
        "id": "R01",
        "category": "rbi_fema",
        "question": "Can an NRI invest in Indian equities through an NRE account?",
        "expected_keywords": ["nre", "equity", "invest", "repatriable", "pis"],
        "expected_source": "rbi_fema",
    },
    {
        "id": "R02",
        "category": "rbi_fema",
        "question": "What is the LRS limit for NRI remittances per year?",
        "expected_keywords": ["lrs", "liberalised", "remittance", "250000", "limit"],
        "expected_source": "rbi_fema",
    },
    {
        "id": "R03",
        "category": "rbi_fema",
        "question": "What is the difference between NRE and NRO accounts?",
        "expected_keywords": ["nre", "nro", "repatriable", "taxable", "foreign"],
        "expected_source": "rbi_fema",
    },
    {
        "id": "R04",
        "category": "rbi_fema",
        "question": "How much can an NRI repatriate from an NRO account per year?",
        "expected_keywords": ["nro", "repatriate", "million", "1000000", "limit"],
        "expected_source": "rbi_fema",
    },
    {
        "id": "R05",
        "category": "rbi_fema",
        "question": "What is an FCNR account and who can open it?",
        "expected_keywords": ["fcnr", "foreign currency", "nri", "deposit", "bank"],
        "expected_source": "rbi_fema",
    },
    {
        "id": "R06",
        "category": "rbi_fema",
        "question": "What are the FEMA rules for NRI investment in India?",
        "expected_keywords": ["fema", "nri", "investment", "foreign exchange", "rbi"],
        "expected_source": "rbi_fema",
    },
    {
        "id": "R07",
        "category": "rbi_fema",
        "question": "Can NRI invest in mutual funds in India?",
        "expected_keywords": ["mutual fund", "nri", "invest", "nre", "nro"],
        "expected_source": "rbi_fema",
    },

    # ── DTAA / Tax (7) ────────────────────────────────────────────────────────
    {
        "id": "T01",
        "category": "tax_dtaa",
        "question": "What is the DTAA benefit for NRI in UAE on Indian dividends?",
        "expected_keywords": ["dtaa", "uae", "dividend", "tax", "exemption"],
        "expected_source": "tax_dtaa",
    },
    {
        "id": "T02",
        "category": "tax_dtaa",
        "question": "How does the India-UK DTAA affect NRI capital gains?",
        "expected_keywords": ["uk", "dtaa", "capital gains", "tax", "india"],
        "expected_source": "tax_dtaa",
    },
    {
        "id": "T03",
        "category": "tax_dtaa",
        "question": "What tax does an NRI in Singapore pay on Indian interest income?",
        "expected_keywords": ["singapore", "interest", "tax", "dtaa", "income"],
        "expected_source": "tax_dtaa",
    },
    {
        "id": "T04",
        "category": "tax_dtaa",
        "question": "Does India have a DTAA with Canada for NRI investors?",
        "expected_keywords": ["canada", "dtaa", "india", "tax", "treaty"],
        "expected_source": "tax_dtaa",
    },
    {
        "id": "T05",
        "category": "tax_dtaa",
        "question": "What is TDS rate on dividends for NRI under India-USA DTAA?",
        "expected_keywords": ["usa", "tds", "dividend", "dtaa", "rate"],
        "expected_source": "tax_dtaa",
    },
    {
        "id": "T06",
        "category": "tax_dtaa",
        "question": "What are the special tax provisions for NRI under section 115E?",
        "expected_keywords": ["115e", "nri", "investment", "income", "tax"],
        "expected_source": "tax_dtaa",
    },
    {
        "id": "T07",
        "category": "tax_dtaa",
        "question": "How are NRI capital gains from Indian stocks taxed?",
        "expected_keywords": ["capital gains", "nri", "tax", "stock", "india"],
        "expected_source": "tax_dtaa",
    },

    # ── Structured / NSE (6) ──────────────────────────────────────────────────
    {
        "id": "N01",
        "category": "structured",
        "question": "What is the ISIN for Reliance Industries?",
        "expected_keywords": ["ine002a01018", "reliance", "isin"],
        "expected_source": "structured",
    },
    {
        "id": "N02",
        "category": "structured",
        "question": "What sector does Infosys belong to on NSE?",
        "expected_keywords": ["infosys", "it", "technology", "sector"],
        "expected_source": "structured",
    },
    {
        "id": "N03",
        "category": "structured",
        "question": "What is the ISIN for HDFC Bank?",
        "expected_keywords": ["hdfc", "isin", "bank"],
        "expected_source": "structured",
    },
    {
        "id": "N04",
        "category": "structured",
        "question": "List some Nifty 50 companies in the banking sector",
        "expected_keywords": ["bank", "nifty", "hdfc", "icici", "sbi"],
        "expected_source": "structured",
    },
    {
        "id": "N05",
        "category": "structured",
        "question": "What is TCS stock symbol on NSE?",
        "expected_keywords": ["tcs", "symbol", "nse", "tata"],
        "expected_source": "structured",
    },
    {
        "id": "N06",
        "category": "structured",
        "question": "Which sector does Wipro belong to?",
        "expected_keywords": ["wipro", "it", "technology", "software"],
        "expected_source": "structured",
    },
]


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluator for a RAG (Retrieval-Augmented Generation) system about NRI investments.

Evaluate the following response on TWO metrics. Respond ONLY with valid JSON, no other text.

Question: {question}

Retrieved Context:
{context}

Agent Answer:
{answer}

Expected keywords that should appear: {keywords}

Evaluate:
1. retrieval_relevance (1-5): Did the retrieved context contain information relevant to answer the question?
   1=completely irrelevant, 3=partially relevant, 5=perfectly relevant

2. faithfulness (1-5): Does the answer stay within the retrieved context without hallucinating?
   1=major hallucinations, 3=some unsupported claims, 5=fully grounded in context

Respond with ONLY this JSON (no markdown, no explanation):
{{"retrieval_relevance": <int>, "faithfulness": <int>, "retrieval_note": "<one sentence>", "faithfulness_note": "<one sentence>"}}"""


# ── Eval result ───────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    id:                   str
    category:             str
    question:             str
    answer:               str
    retrieval_relevance:  int
    faithfulness:         int
    kg_coverage:          int    # 1 if KG added facts, 0 if not
    chunks_used:          int
    top_score:            float
    collection_routed:    str
    retrieval_note:       str
    faithfulness_note:    str
    latency_s:            float


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_response(question: str, context: str,
                   answer: str, keywords: list) -> dict:
    """Use LLM-as-judge to score retrieval relevance and faithfulness."""
    prompt = JUDGE_PROMPT.format(
        question = question,
        context  = context[:2000],   # truncate to avoid token limits
        answer   = answer[:1000],
        keywords = ", ".join(keywords),
    )
    try:
        response = ollama.chat(
            model    = JUDGE_MODEL,
            messages = [{"role": "user", "content": prompt}],
            options  = {"num_predict": 200, "temperature": 0.0},
        )
        raw = response["message"]["content"].strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"    Judge error: {e}")
        return {
            "retrieval_relevance": 0,
            "faithfulness":        0,
            "retrieval_note":      f"Judge failed: {e}",
            "faithfulness_note":   "",
        }


# ── Keyword coverage check ────────────────────────────────────────────────────

def check_keyword_coverage(answer: str, keywords: list) -> float:
    """Simple keyword hit rate in answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 2) if keywords else 0.0


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(questions: list, agent: NRIAgent) -> list[EvalResult]:
    results = []
    total   = len(questions)

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{total}] {q['id']} — {q['question'][:60]}...")

        t0 = time.time()
        try:
            resp = agent.ask(q["question"], verbose=False)
        except Exception as e:
            print(f"  Agent error: {e}")
            continue
        latency = round(time.time() - t0, 2)

        # Get top chunk score
        top_score = 0.0
        context   = ""
        try:
            from agent import retrieve, route_query
            col    = route_query(q["question"])
            chunks = retrieve(q["question"], col,
                              agent.embed_model, agent.chroma, top_k=3)
            if chunks:
                top_score = chunks[0].score
                context   = "\n\n".join(c.text[:300] for c in chunks)
        except Exception:
            context = resp.answer[:500]

        # Judge the response
        print(f"  Judging... (latency so far: {latency}s)")
        scores = judge_response(
            question = q["question"],
            context  = context,
            answer   = resp.answer,
            keywords = q["expected_keywords"],
        )

        # KG coverage: did KG contribute?
        kg_hit = 1 if resp.kg_context and len(resp.kg_context) > 10 else 0

        result = EvalResult(
            id                  = q["id"],
            category            = q["category"],
            question            = q["question"],
            answer              = resp.answer[:300],
            retrieval_relevance = scores.get("retrieval_relevance", 0),
            faithfulness        = scores.get("faithfulness", 0),
            kg_coverage         = kg_hit,
            chunks_used         = resp.chunks_used,
            top_score           = top_score,
            collection_routed   = resp.collection,
            retrieval_note      = scores.get("retrieval_note", ""),
            faithfulness_note   = scores.get("faithfulness_note", ""),
            latency_s           = latency,
        )

        print(f"  Relevance: {result.retrieval_relevance}/5 | "
              f"Faithfulness: {result.faithfulness}/5 | "
              f"KG: {'✅' if kg_hit else '❌'} | "
              f"Latency: {latency}s")

        results.append(result)

    return results


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: list[EvalResult]):
    # CSV
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\n💾 Results saved to {RESULTS_FILE}")


def save_summary(results: list[EvalResult]):
    df = pd.DataFrame([asdict(r) for r in results])

    avg_relevance   = df["retrieval_relevance"].mean()
    avg_faithfulness= df["faithfulness"].mean()
    kg_coverage_pct = df["kg_coverage"].mean() * 100
    avg_latency     = df["latency_s"].mean()
    avg_chunks      = df["chunks_used"].mean()

    lines = [
        "=" * 60,
        "NRI EQUITY RAG — EVALUATION SUMMARY",
        f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Questions evaluated: {len(results)}",
        "=" * 60,
        "",
        "OVERALL SCORES",
        f"  Retrieval Relevance : {avg_relevance:.2f} / 5.0",
        f"  Faithfulness        : {avg_faithfulness:.2f} / 5.0",
        f"  KG Coverage         : {kg_coverage_pct:.0f}% of queries got KG facts",
        f"  Avg Latency         : {avg_latency:.1f}s per query",
        f"  Avg Chunks Used     : {avg_chunks:.1f}",
        "",
        "BY CATEGORY",
    ]

    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        lines.append(
            f"  {cat:15s} | "
            f"Relevance: {cat_df['retrieval_relevance'].mean():.2f} | "
            f"Faithfulness: {cat_df['faithfulness'].mean():.2f} | "
            f"n={len(cat_df)}"
        )

    lines += [
        "",
        "ROUTING ACCURACY",
    ]
    for cat in df["category"].unique():
        cat_df     = df[df["category"] == cat]
        correct    = cat_df[cat_df["collection_routed"].str.contains(
                        cat.split("_")[0], na=False)]
        accuracy   = len(correct) / len(cat_df) * 100
        lines.append(f"  {cat:15s} | Routed correctly: {accuracy:.0f}%")

    lines += [
        "",
        "LOW SCORING QUESTIONS (relevance < 3)",
    ]
    low = df[df["retrieval_relevance"] < 3]
    if len(low) == 0:
        lines.append("  None — all questions scored >= 3")
    else:
        for _, row in low.iterrows():
            lines.append(f"  [{row['id']}] {row['question'][:60]}")
            lines.append(f"       Note: {row['retrieval_note']}")

    lines.append("=" * 60)
    summary = "\n".join(lines)

    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)

    print(summary)
    print(f"\n💾 Summary saved to {SUMMARY_FILE}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true",
                        help="Run only 5 questions (one per category)")
    parser.add_argument("--show",   action="store_true",
                        help="Print results from last run without re-running")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only questions from one category")
    args = parser.parse_args()

    # Show previous results
    if args.show:
        if RESULTS_FILE.exists():
            df = pd.read_csv(RESULTS_FILE)
            print(df[["id", "category", "retrieval_relevance",
                       "faithfulness", "kg_coverage", "latency_s"]].to_string())
            print(f"\nAvg Relevance  : {df['retrieval_relevance'].mean():.2f}")
            print(f"Avg Faithfulness: {df['faithfulness'].mean():.2f}")
        else:
            print("No results file found. Run eval first.")
        return

    # Select questions
    questions = TEST_QUESTIONS
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.quick:
        # One per category
        seen = set()
        quick_qs = []
        for q in questions:
            if q["category"] not in seen:
                quick_qs.append(q)
                seen.add(q["category"])
        questions = quick_qs

    print(f"\n🧪 NRI Equity RAG — Evaluation Harness")
    print(f"   Questions : {len(questions)}")
    print(f"   Judge     : {JUDGE_MODEL} (local)\n")

    # Load agent
    agent = NRIAgent()

    # Run eval
    results = run_eval(questions, agent)

    if not results:
        print("No results — something went wrong.")
        return

    # Save + summarise
    save_results(results)
    save_summary(results)


if __name__ == "__main__":
    main()