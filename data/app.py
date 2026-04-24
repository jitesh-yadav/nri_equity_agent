import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="NRI Equity Agent", page_icon="🇮🇳", layout="wide")

st.title("🇮🇳 NRI Equity Agent")
st.caption("AI-powered investment guidance · SEBI · RBI · FEMA · DTAA · NSE")

# Lazy load agent only when needed
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load agent button
if st.session_state.agent is None:
    if st.button("🚀 Initialize Agent", type="primary"):
        with st.spinner("Loading models... (30-60 seconds on first load)"):
            try:
                from agent import NRIAgent
                st.session_state.agent = NRIAgent()
                st.success("Agent ready!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    st.info("Click the button above to initialize the agent.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Knowledge Base")
    st.markdown("- **3,714** chunks indexed")
    st.markdown("- **19** source documents")
    st.markdown("- SEBI · RBI · DTAA · NSE")

    st.markdown("### 💡 Sample Queries")
    samples = [
        "Can NRI invest via NRE account?",
        "DTAA benefit for NRI in UAE?",
        "ISIN for Reliance Industries?",
        "SEBI KYC requirements for NRI?",
        "LRS repatriation limit?",
    ]
    for q in samples:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending = q

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.caption(f"• {s}")

# Handle pending query from sidebar
query = None
if "pending" in st.session_state and st.session_state.pending:
    query = st.session_state.pending
    st.session_state.pending = None

# Chat input
user_input = st.chat_input("Ask about NRI investments, DTAA, stocks...")
if user_input:
    query = user_input

# Process query
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                resp = st.session_state.agent.ask(query, verbose=False)
                st.write(resp.answer)
                with st.expander(f"📄 Sources ({len(resp.sources)}) · {resp.collection} · {resp.chunks_used} chunks"):
                    for s in resp.sources:
                        st.caption(f"• {s}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp.answer,
                    "sources": resp.sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")
    st.rerun()
