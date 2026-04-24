"""
knowledge_graph.py — NRI Equity Knowledge Graph

Usage:
    python3 knowledge_graph.py           # build and save graph
    python3 knowledge_graph.py --query   # interactive query mode
    python3 knowledge_graph.py --rebuild # force rebuild

Builds a NetworkX graph from:
  - NSE equity CSVs  → Stock + Sector nodes
  - Hardcoded data   → NRICategory, Country, TaxTreaty, RegulatoryBody nodes

Saves to: data/graph.pkl

Dependencies:
    pip install networkx pandas
"""

import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd

BASE       = Path("/Users/jiteshyadav/Desktop/data")
STRUCT_DIR = BASE / "structured"
GRAPH_FILE = BASE / "graph.pkl"

# ── Hardcoded knowledge ───────────────────────────────────────────────────────

NRI_ACCOUNTS = {
    "NRE": {
        "full_name":     "Non-Resident External Account",
        "currency":      "INR",
        "repatriable":   True,
        "taxable_india": False,
        "use_case":      "Foreign earnings brought to India, fully repatriable",
        "can_invest":    ["equities", "mutual_funds", "fixed_deposits", "etf"],
        "governed_by":   "RBI",
    },
    "NRO": {
        "full_name":     "Non-Resident Ordinary Account",
        "currency":      "INR",
        "repatriable":   False,
        "taxable_india": True,
        "use_case":      "Indian income like rent, dividends, pension",
        "can_invest":    ["equities", "mutual_funds", "fixed_deposits"],
        "governed_by":   "RBI",
        "repatriation_limit_usd": 1000000,
    },
    "FCNR": {
        "full_name":     "Foreign Currency Non-Resident Account",
        "currency":      "Foreign (USD/GBP/EUR/JPY/AUD/CAD/SGD)",
        "repatriable":   True,
        "taxable_india": False,
        "use_case":      "Fixed deposits in foreign currency, no exchange risk",
        "can_invest":    ["fixed_deposits"],
        "governed_by":   "RBI",
    },
}

DTAA_COUNTRIES = {
    "UAE":       {"dividend_tds": 0,  "capital_gains_tds": 0,  "interest_tds": 12.5,
                  "notes": "UAE has no personal income tax. Very favorable for NRIs."},
    "USA":       {"dividend_tds": 25, "capital_gains_tds": 20, "interest_tds": 15,
                  "notes": "US taxes worldwide income. Consult tax advisor."},
    "UK":        {"dividend_tds": 15, "capital_gains_tds": 10, "interest_tds": 15,
                  "notes": "India-UK DTAA. UK taxes residents on worldwide income."},
    "Singapore": {"dividend_tds": 15, "capital_gains_tds": 0,  "interest_tds": 15,
                  "notes": "Singapore has no capital gains tax. Favorable for NRIs."},
    "Canada":    {"dividend_tds": 25, "capital_gains_tds": 15, "interest_tds": 15,
                  "notes": "Canada taxes worldwide income. DTAA provides relief."},
}

INVESTMENT_RULES = {
    "equities":      {"allowed_accounts": ["NRE", "NRO"], "regulator": "SEBI+RBI",
                      "route": "Portfolio Investment Scheme (PIS)",
                      "limit": "5% of paid-up capital per NRI, 10% aggregate"},
    "mutual_funds":  {"allowed_accounts": ["NRE", "NRO"], "regulator": "SEBI",
                      "route": "Direct or via distributor with KYC"},
    "etf":           {"allowed_accounts": ["NRE", "NRO"], "regulator": "SEBI",
                      "route": "Through demat account linked to NRE/NRO"},
    "fixed_deposits":{"allowed_accounts": ["NRE", "NRO", "FCNR"], "regulator": "RBI",
                      "route": "Direct with bank"},
}

REGULATORY_BODIES = {
    "SEBI": {"full_name": "Securities and Exchange Board of India",
             "regulates": ["equities", "mutual_funds", "etf", "fpi"]},
    "RBI":  {"full_name": "Reserve Bank of India",
             "regulates": ["banking", "forex", "fema", "lrs", "repatriation"]},
    "CBDT": {"full_name": "Central Board of Direct Taxes",
             "regulates": ["income_tax", "dtaa", "tds", "capital_gains_tax"]},
}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # 1. NSE stocks + sectors
    print("📈 Loading NSE equity data...")
    stocks_added = 0
    sectors_added = set()

    for csv_file in ["nse_equity_list.csv", "nifty50_constituents.csv",
                     "sector_mapping.csv"]:
        fpath = STRUCT_DIR / csv_file
        if not fpath.exists():
            # try double extension
            fpath2 = STRUCT_DIR / (csv_file + ".csv")
            if fpath2.exists():
                fpath = fpath2
            else:
                continue
        try:
            df = pd.read_csv(fpath)
            df.columns = [c.strip().upper() for c in df.columns]
            sym_col    = next((c for c in df.columns if "SYMBOL" in c), None)
            name_col   = next((c for c in df.columns if "NAME" in c or "COMPANY" in c), None)
            sector_col = next((c for c in df.columns if "SECTOR" in c or "INDUSTRY" in c), None)
            isin_col   = next((c for c in df.columns if "ISIN" in c), None)
            if not sym_col:
                continue
            for _, row in df.iterrows():
                symbol = str(row[sym_col]).strip()
                if not symbol or symbol == "nan":
                    continue
                name   = str(row[name_col]).strip() if name_col else symbol
                sector = str(row[sector_col]).strip() if sector_col else "Unknown"
                isin   = str(row[isin_col]).strip() if isin_col else ""
                G.add_node(symbol, node_type="Stock", name=name,
                           isin=isin, sector=sector, exchange="NSE")
                stocks_added += 1
                if sector and sector not in ("nan", "Unknown"):
                    if sector not in sectors_added:
                        G.add_node(sector, node_type="Sector", name=sector)
                        sectors_added.add(sector)
                    G.add_edge(symbol, sector, relation="belongs_to_sector")
                    G.add_edge(sector, symbol, relation="contains_stock")
        except Exception as e:
            print(f"  Warning: {csv_file}: {e}")

    print(f"  ✅ {stocks_added} stocks, {len(sectors_added)} sectors")

    # 2. NRI account nodes
    print("🏦 Adding NRI account types...")
    for acct, attrs in NRI_ACCOUNTS.items():
        G.add_node(acct, node_type="NRICategory", **attrs)
    for inv, rules in INVESTMENT_RULES.items():
        G.add_node(inv, node_type="InvestmentType", **rules)
        for acct in rules["allowed_accounts"]:
            G.add_edge(acct, inv, relation="can_invest_in",
                       route=rules.get("route",""))
            G.add_edge(inv, acct, relation="accessible_via")

    # 3. DTAA country-treaty nodes
    print("🌍 Adding DTAA treaties...")
    for country, treaty in DTAA_COUNTRIES.items():
        G.add_node(country, node_type="Country", name=country)
        tid = f"DTAA_India_{country}"
        G.add_node(tid, node_type="TaxTreaty", country=country, **treaty)
        G.add_edge(country, tid, relation="has_dtaa_with_india")
        G.add_edge(tid, country, relation="applies_to_residents_of")

    # 4. Regulatory bodies
    print("🏛️  Adding regulatory bodies...")
    for body, attrs in REGULATORY_BODIES.items():
        G.add_node(body, node_type="RegulatoryBody", **attrs)
    for acct, attrs in NRI_ACCOUNTS.items():
        gov = attrs["governed_by"]
        G.add_edge(gov, acct, relation="regulates")
        G.add_edge(acct, gov, relation="regulated_by")
    for inv in ["equities", "mutual_funds", "etf"]:
        if inv in G:
            G.add_edge("SEBI", inv, relation="regulates")

    # Summary
    type_counts = defaultdict(int)
    for _, d in G.nodes(data=True):
        type_counts[d.get("node_type", "Unknown")] += 1
    print(f"\n📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for t, c in sorted(type_counts.items()):
        print(f"   {t}: {c}")
    return G


# ── KnowledgeGraph class ──────────────────────────────────────────────────────

class KnowledgeGraph:
    def __init__(self, graph_file: Path = GRAPH_FILE):
        if graph_file.exists():
            with open(graph_file, "rb") as f:
                self.G = pickle.load(f)
            print(f"✅ Loaded KG: {self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")
        else:
            print("Building graph from scratch...")
            self.G = build_graph()
            self.save(graph_file)

    def save(self, path=GRAPH_FILE):
        with open(path, "wb") as f:
            pickle.dump(self.G, f)
        print(f"💾 Saved to {path}")

    def get_stock_info(self, query: str) -> dict:
        query = query.upper().strip()
        if query in self.G and self.G.nodes[query].get("node_type") == "Stock":
            return {**self.G.nodes[query], "symbol": query}
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "Stock":
                if (query in node or query in data.get("name","").upper()
                        or query == data.get("isin","")):
                    return {**data, "symbol": node}
        return {}

    def get_stocks_by_sector(self, sector: str, limit=10) -> list:
        results = []
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "Sector" and sector.lower() in node.lower():
                for nb in self.G.successors(node):
                    nd = self.G.nodes[nb]
                    if nd.get("node_type") == "Stock":
                        results.append({"symbol": nb, "name": nd.get("name",""),
                                        "isin": nd.get("isin",""), "sector": node})
                if results:
                    break
        return results[:limit]

    def get_dtaa_info(self, country: str) -> dict:
        country = country.strip().title()
        if country in self.G:
            for nb in self.G.successors(country):
                if self.G.nodes[nb].get("node_type") == "TaxTreaty":
                    return {**self.G.nodes[nb], "country": country}
        return {}

    def get_account_info(self, acct: str) -> dict:
        acct = acct.upper().strip()
        if acct in self.G and self.G.nodes[acct].get("node_type") == "NRICategory":
            investments = [nb for nb in self.G.successors(acct)
                           if self.G.nodes[nb].get("node_type") == "InvestmentType"]
            return {**self.G.nodes[acct], "investments": investments}
        return {}

    def query(self, question: str) -> str:
        """Route natural language question to correct KG method, return context string."""
        q = question.lower()
        results = []

        # Stock lookup
        if any(kw in q for kw in ["isin", "symbol", "stock", "share"]):
            for word in question.upper().split():
                if len(word) >= 3 and word.isalpha():
                    info = self.get_stock_info(word)
                    if info:
                        results.append(
                            f"[KG] Stock: {info.get('name',word)} | "
                            f"Symbol: {info.get('symbol',word)} | "
                            f"ISIN: {info.get('isin','N/A')} | "
                            f"Sector: {info.get('sector','N/A')}"
                        )
                        break

        # Sector lookup
        for sector in ["it", "technology", "banking", "finance", "pharma",
                       "auto", "fmcg", "energy", "telecom", "metal"]:
            if sector in q:
                stocks = self.get_stocks_by_sector(sector, limit=5)
                if stocks:
                    sl = ", ".join(f"{s['symbol']} ({s['name'][:15]})" for s in stocks)
                    results.append(f"[KG] {sector.upper()} sector stocks (top 5): {sl}")
                break

        # DTAA lookup
        for country in DTAA_COUNTRIES:
            if country.lower() in q:
                info = self.get_dtaa_info(country)
                if info:
                    results.append(
                        f"[KG] DTAA India-{country}: "
                        f"Dividend TDS={info.get('dividend_tds')}%, "
                        f"Capital Gains={info.get('capital_gains_tds')}%, "
                        f"Interest={info.get('interest_tds')}% | "
                        f"{info.get('notes','')}"
                    )

        # Account type lookup
        for acct in ["NRE", "NRO", "FCNR"]:
            if acct.lower() in q:
                info = self.get_account_info(acct)
                if info:
                    results.append(
                        f"[KG] {acct}: {info.get('full_name')} | "
                        f"Repatriable: {info.get('repatriable')} | "
                        f"Taxable India: {info.get('taxable_india')} | "
                        f"Invests in: {', '.join(info.get('investments',[]))}"
                    )

        return "\n".join(results)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild or not GRAPH_FILE.exists():
        G = build_graph()
        with open(GRAPH_FILE, "wb") as f:
            pickle.dump(G, f)
        print(f"\n✅ Graph saved to {GRAPH_FILE}")
    else:
        print(f"Graph exists at {GRAPH_FILE} — use --rebuild to rebuild")

    if args.query:
        kg = KnowledgeGraph()
        print("\nKG Query Mode — type 'quit' to exit")
        print("Try: 'ISIN for Reliance' / 'IT sector stocks' / 'DTAA UAE' / 'NRE account'\n")
        while True:
            try:
                q = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "quit":
                break
            result = kg.query(q)
            print(result or "No KG match")
            print()


if __name__ == "__main__":
    main()