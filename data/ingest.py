"""
ingest.py — NRI Equity RAG: PDF ingestion + chunking pipeline

Usage:
    python ingest.py

Reads all PDFs from data/raw_pdfs/ (recursive), chunks them with
document-type-aware strategies, and saves chunks to data/processed/chunks.json

Dependencies:
    pip install pdfplumber chromadb sentence-transformers pandas tqdm
"""

import os
import json
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import pdfplumber
import pandas as pd
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR    = Path("data/raw_pdfs")
STRUCT_DIR = Path("data/structured")
OUT_DIR    = Path("data/processed")
OUT_FILE   = OUT_DIR / "chunks.json"

# Chunk sizes by document type (in characters)
CHUNK_CONFIG = {
    "faq":        {"size": 800,  "overlap": 100},   # Q&A pairs — keep tight
    "circular":   {"size": 1200, "overlap": 150},   # Short circulars
    "regulation": {"size": 1500, "overlap": 200},   # Dense legal text — larger
    "dtaa":       {"size": 1200, "overlap": 150},   # Treaty articles
    "default":    {"size": 1200, "overlap": 150},
}

# Keywords to auto-detect document type from filename or content
TYPE_HINTS = {
    "faq":        ["faq", "frequently asked", "question"],
    "circular":   ["circular", "sebi", "rbi", "notification"],
    "regulation": ["regulation", "fema", "act", "direction", "master direction"],
    "dtaa":       ["dtaa", "tax treaty", "double taxation", "synthesised"],
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:   str       # sha256 of content
    source:     str       # filename
    doc_type:   str       # faq / circular / regulation / dtaa
    category:   str       # sebi / rbi_fema / tax_dtaa / structured
    page:       int
    chunk_idx:  int
    text:       str
    metadata:   dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def detect_doc_type(filename: str, sample_text: str) -> str:
    """Infer document type from filename and first 500 chars of text."""
    haystack = (filename + " " + sample_text[:500]).lower()
    for doc_type, keywords in TYPE_HINTS.items():
        if any(kw in haystack for kw in keywords):
            return doc_type
    return "default"


def detect_category(filepath: Path) -> str:
    """Infer category from folder path."""
    parts = filepath.parts
    for part in parts:
        p = part.lower()
        if "sebi" in p:
            return "sebi"
        if "rbi" in p or "fema" in p:
            return "rbi_fema"
        if "tax" in p or "dtaa" in p:
            return "tax_dtaa"
    return "general"


def clean_text(text: str) -> str:
    """Remove excessive whitespace and page artifacts."""
    text = re.sub(r'\n{3,}', '\n\n', text)          # collapse blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)           # collapse spaces
    text = re.sub(r'\f', '\n\n', text)               # form feeds
    text = re.sub(r'Page \d+ of \d+', '', text)      # page numbers
    text = text.strip()
    return text


def split_faq(text: str, source: str, category: str, page: int) -> List[Chunk]:
    """
    Special chunker for FAQ documents.
    Splits on Q: / A: or numbered question patterns.
    Each Q+A pair becomes one chunk.
    """
    chunks = []
    # Match patterns like "Q1.", "Q:", "1.", "Question 1"
    pattern = re.compile(
        r'(?:^|\n)(?:Q\d*[\.\:]|Question\s*\d+[\.\:]|\d+[\.\)])\s',
        re.MULTILINE | re.IGNORECASE
    )
    splits = pattern.split(text)
    matches = pattern.findall(text)

    for i, block in enumerate(splits):
        block = block.strip()
        if len(block) < 50:  # skip tiny fragments
            continue
        prefix = matches[i - 1].strip() if i > 0 and i - 1 < len(matches) else ""
        chunk_text = (prefix + " " + block).strip()
        chunks.append(Chunk(
            chunk_id  = make_id(chunk_text),
            source    = source,
            doc_type  = "faq",
            category  = category,
            page      = page,
            chunk_idx = i,
            text      = chunk_text,
            metadata  = {"length": len(chunk_text)},
        ))
    return chunks


def split_by_size(text: str, source: str, doc_type: str,
                  category: str, page: int, start_idx: int = 0) -> List[Chunk]:
    """
    Sliding window chunker for regulatory / circular / DTAA text.
    Tries to break at sentence boundaries within the window.
    """
    cfg   = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["default"])
    size  = cfg["size"]
    over  = cfg["overlap"]
    chunks = []
    idx   = 0
    ci    = start_idx

    while idx < len(text):
        end = idx + size
        if end < len(text):
            # Try to break at a sentence boundary within last 20% of chunk
            boundary_zone = text[idx + int(size * 0.8): end]
            # Look for ". " or ".\n" as sentence end
            m = re.search(r'\.\s', boundary_zone[::-1])  # search backwards
            if m:
                end = end - m.start()

        chunk_text = text[idx:end].strip()
        if len(chunk_text) > 80:  # skip tiny trailing chunks
            chunks.append(Chunk(
                chunk_id  = make_id(chunk_text),
                source    = source,
                doc_type  = doc_type,
                category  = category,
                page      = page,
                chunk_idx = ci,
                text      = chunk_text,
                metadata  = {"length": len(chunk_text), "char_start": idx},
            ))
            ci += 1

        idx = end - over  # slide back by overlap
        if idx >= len(text):
            break

    return chunks


# ── PDF processor ─────────────────────────────────────────────────────────────

def process_pdf(filepath: Path) -> List[Chunk]:
    """Extract text from PDF and chunk it with the right strategy."""
    all_chunks = []
    category   = detect_category(filepath)
    source     = filepath.name

    try:
        with pdfplumber.open(filepath) as pdf:
            full_text_sample = ""

            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text()
                if not raw:
                    continue  # skip empty / image-only pages

                text = clean_text(raw)

                # Use first page to detect doc type
                if page_num == 1:
                    full_text_sample = text
                    doc_type = detect_doc_type(source, full_text_sample)
                    print(f"  └─ type: {doc_type} | pages: {len(pdf.pages)}")

                # Route to correct chunker
                if doc_type == "faq":
                    page_chunks = split_faq(text, source, category, page_num)
                    if not page_chunks:
                        # fallback if FAQ pattern not found
                        page_chunks = split_by_size(text, source, doc_type,
                                                    category, page_num)
                else:
                    page_chunks = split_by_size(text, source, doc_type,
                                                category, page_num)

                all_chunks.extend(page_chunks)

    except Exception as e:
        print(f"  ❌ Error processing {filepath.name}: {e}")

    return all_chunks


# ── CSV processor ─────────────────────────────────────────────────────────────

def process_csv(filepath: Path) -> List[Chunk]:
    """
    Convert NSE equity CSVs into text chunks.
    Each row = one chunk (stock entry).
    """
    chunks = []
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]

        for i, row in df.iterrows():
            # Build a natural language description of the stock
            parts = []
            for col in df.columns:
                val = str(row[col]).strip()
                if val and val.lower() != "nan":
                    parts.append(f"{col}: {val}")

            text = filepath.stem.upper() + " | " + " | ".join(parts)
            chunks.append(Chunk(
                chunk_id  = make_id(text),
                source    = filepath.name,
                doc_type  = "structured",
                category  = "structured",
                page      = 0,
                chunk_idx = i,
                text      = text,
                metadata  = {"row": i, "file": filepath.name},
            ))

    except Exception as e:
        print(f"  ❌ Error processing {filepath.name}: {e}")

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    # 1. Process PDFs
    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"\n📄 Found {len(pdf_files)} PDFs\n")

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\n→ {pdf_path.relative_to(PDF_DIR)}")
        chunks = process_pdf(pdf_path)
        print(f"  └─ {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # 2. Process CSVs
    csv_files = sorted(STRUCT_DIR.glob("*.csv"))
    print(f"\n📊 Found {len(csv_files)} CSVs\n")

    for csv_path in tqdm(csv_files, desc="Processing CSVs"):
        print(f"\n→ {csv_path.name}")
        chunks = process_csv(csv_path)
        print(f"  └─ {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # 3. Deduplicate by chunk_id
    seen = set()
    deduped = []
    for c in all_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            deduped.append(c)

    print(f"\n✅ Total chunks: {len(all_chunks)} → after dedup: {len(deduped)}")

    # 4. Save to JSON
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in deduped], f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {OUT_FILE}")

    # 5. Print summary by category
    from collections import Counter
    cats = Counter(c.category for c in deduped)
    types = Counter(c.doc_type for c in deduped)
    print("\n📊 Chunks by category:", dict(cats))
    print("📊 Chunks by doc_type:", dict(types))


if __name__ == "__main__":
    main()