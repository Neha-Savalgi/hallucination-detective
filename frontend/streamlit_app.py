import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hallucination Detective", page_icon="🕵️", layout="centered")

st.title("🕵️ Hallucination Detective")
st.caption("RAG-based pipeline that catches LLM hallucinations in real time.")

with st.expander("📚 Step 1 — Add source documents to the knowledge base", expanded=True):
    doc_source = st.text_input("Source name", value="my_document.txt")
    doc_text = st.text_area("Paste the ground-truth text here", height=200,
                             placeholder="Paste the reference document, article, or facts you want to fact-check claims against...")
    if st.button("Ingest document"):
        if doc_text.strip():
            resp = requests.post(f"{API_URL}/ingest", json={"text": doc_text, "source": doc_source})
            data = resp.json()
            st.success(f"Added {data['chunks_added']} chunks. Total indexed: {data['total_documents']}")
        else:
            st.warning("Paste some text first.")

st.divider()

st.subheader("🔍 Step 2 — Check a claim")
claim = st.text_area("Enter a claim or LLM output to fact-check", height=100,
                      placeholder="e.g. The Eiffel Tower is 500 meters tall.")

col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.number_input("Top-K evidence", min_value=1, max_value=10, value=4)

if st.button("Check for hallucination", type="primary"):
    if not claim.strip():
        st.warning("Enter a claim first.")
    else:
        with st.spinner("Retrieving evidence and judging..."):
            resp = requests.post(f"{API_URL}/check", json={"claim": claim, "top_k": int(top_k)})
            result = resp.json()

        verdict = result.get("verdict", "UNCERTAIN")
        confidence = result.get("confidence", 0.0)

        color = {"GROUNDED": "green", "HALLUCINATED": "red", "UNCERTAIN": "orange"}.get(verdict, "gray")
        emoji = {"GROUNDED": "✅", "HALLUCINATED": "🚨", "UNCERTAIN": "❓"}.get(verdict, "❔")

        st.markdown(f"### {emoji} :{color}[{verdict}]  (confidence: {confidence:.2f})")
        st.write(result.get("reasoning", ""))

        supporting = result.get("supporting_evidence", [])
        if supporting:
            st.markdown("**Supporting evidence cited by judge:**")
            for s in supporting:
                st.markdown(f"- {s}")

        evidence_used = result.get("evidence_used", [])
        if evidence_used:
            with st.expander(f"📄 Raw retrieved chunks ({len(evidence_used)})"):
                for e in evidence_used:
                    st.markdown(f"**{e['source']}** — relevance: `{e['relevance_score']}`")
                    st.text(e["text"])
                    st.markdown("---")
