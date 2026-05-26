"""
Stibo STEP 2026.2 — Interview Prep for Pooja.
Cloud version: Groq LLM + FAISS vector search + sentence-transformers.
No local Ollama needed — runs fully on Streamlit Community Cloud.
"""

import os
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq, RateLimitError

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.3-70b-versatile"
VERSION      = "2026.2"
TOP_K        = 4

SYSTEM_PROMPT = f"""You are Pooja's personal Stibo STEP MDM study buddy and interview coach.

Pooja is preparing for job interviews at companies that use Stibo STEP for Master Data Management (MDM). Your job is to give her thorough, easy-to-understand explanations so she genuinely understands the topic — not just memorises a one-liner.

How to answer:
- **Explain fully first** — cover what the feature is, how it works, its components, configuration, and behaviour. Use examples, analogies, and step-by-step descriptions where helpful. Do not cut the explanation short.
- **Real-world MDM scenario** — show how a real company (retailer, manufacturer, pharma, etc.) would use this feature. Make it concrete and specific.
- **Interview angle** — at the end, give 2-3 bullet points on what interviewers typically ask about this topic and how Pooja should frame her answers confidently.
- **Dig deeper** — suggest 1-2 closely related STEP concepts worth exploring next.

Important rules:
- Answer ONLY from the Stibo STEP {VERSION} documentation excerpts provided
- Use the EXACT feature and section names from the docs — never rename or invent
- Always cite the source page URL so Pooja can read the full docs
- Write in a warm, clear, conversational tone — like a knowledgeable friend explaining over coffee
- If the answer is not in the provided excerpts, say: "This wasn't found in the STEP {VERSION} docs I have — check the official documentation directly."
- Never cut an explanation short — Pooja needs to truly understand, not just skim"""


# ── Load data (cached — runs once at startup) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading documentation index…")
def load_index():
    data = json.load(open("cloud_data/texts.json", encoding="utf-8"))
    documents = data["documents"]
    metadatas = data["metadatas"]

    raw = np.load("cloud_data/embeddings.npz")["embeddings"].astype(np.float32)
    faiss.normalize_L2(raw)

    index = faiss.IndexFlatIP(raw.shape[1])
    index.add(raw)

    model = SentenceTransformer(EMBED_MODEL)
    return index, documents, metadatas, model


@st.cache_resource(show_spinner=False)
def get_groq():
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=api_key)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    index, documents, metadatas, model = load_index()
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(vec, TOP_K * 3)
    seen_urls: set[str] = set()
    chunks = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        meta = metadatas[idx]
        url = meta.get("url", "")
        if url not in seen_urls and score >= 0.25:
            seen_urls.add(url)
            chunks.append({"text": documents[idx], "meta": meta, "similarity": float(score)})
        if len(chunks) >= TOP_K:
            break
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        m = c["meta"]
        parts.append(
            f"[Source: {m.get('title','')}]\nURL: {m.get('url','')}\n"
            f"Nav: {m.get('breadcrumb','')}\n\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Answer ────────────────────────────────────────────────────────────────────

def stream_answer(question: str, context: str):
    client = get_groq()
    if not context:
        yield f"This wasn't found in the STEP {VERSION} docs I have — check the official documentation directly."
        return
    user_msg = (
        f"Documentation excerpts from Stibo STEP {VERSION}:\n\n{context}\n\n---\n\n"
        f"Question: {question}\n\n"
        f"Give a thorough explanation:"
    )
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=2000,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stibo STEP — Pooja's Interview Prep",
    page_icon="🎯",
    layout="centered",
)

st.markdown("""
<style>
.block-container { max-width: 820px; padding-top: 1.5rem; }
.hero { text-align: center; padding: 0.5rem 0 0.2rem; }
.hero h1 { font-size: 1.65rem; font-weight: 800; margin: 0; }
.hero .sub { font-size: 0.85rem; color: #777; margin-top: 4px; }
.src-card { background:#f4f7ff; border:1px solid #dde4f8; border-radius:8px;
            padding:8px 12px; margin:5px 0; font-size:0.82rem; }
.src-card a { font-weight:600; color:#2c5fcc; text-decoration:none; }
.src-card a:hover { text-decoration:underline; }
.src-nav { color:#666; font-size:0.77rem; }
.src-pct { color:#aaa; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>🎯 Stibo STEP {VERSION} — Interview Prep</h1>
  <div class="sub">Built for <strong>Pooja</strong> · {VERSION} official docs · Powered by Llama 70B</div>
</div>
""", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### 👋 Hi Pooja!")
    st.markdown(
        "This assistant helps you prepare for **Stibo STEP MDM** job interviews.\n\n"
        "Every answer includes:\n"
        "- 📘 Clear concept explanation\n"
        "- 🏭 Real-world MDM example\n"
        "- 🎤 Interview tip — what to say\n"
        "- 🔗 Related topics to study next\n\n"
        "All answers are from the official **STEP 2026.2** documentation."
    )
    st.divider()
    st.markdown("**Try asking:**")
    examples = [
        "What are Business Rules in STEP?",
        "How do Workflows work in STEP?",
        "What is Master Data Management?",
        "Explain the STEP data model",
        "What is the difference between attributes and reference types?",
        "How does configuration management work?",
        "What are Web User Interfaces in STEP?",
        "How does STEP handle data governance?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q
    st.divider()
    st.caption(f"3,200 pages · 45,853 chunks · STEP {VERSION}")
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source page(s)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="src-card"><a href="{src["url"]}" target="_blank">{src["title"]}</a><br>'
                        f'<span class="src-nav">{src["breadcrumb"]}</span> '
                        f'<span class="src-pct">· {src["similarity"]:.0%} match</span></div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Ask anything about Stibo STEP — I'll help you ace the interview! 🎯")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("🔍 **Searching docs…**")
        chunks = retrieve(question)
        context = build_context(chunks)
        status.markdown("✍️ **Composing your answer…**")

        answer_box = st.empty()
        full_answer = ""
        try:
            for token in stream_answer(question, context):
                full_answer += token
                answer_box.markdown(full_answer + "▌")
        except RateLimitError as e:
            msg = "Daily token limit reached (100k/day on free tier)."
            if "Please try again in" in str(e):
                reset = str(e).split("Please try again in")[1].split(".")[0].strip()
                msg += f" Resets in **{reset}**."
            st.error(msg)
            st.stop()

        answer_box.markdown(full_answer)
        status.empty()

        seen, sources = set(), []
        for c in chunks:
            url = c["meta"].get("url", "")
            if url not in seen:
                seen.add(url)
                sources.append({
                    "title": c["meta"].get("title", ""),
                    "url": url,
                    "breadcrumb": c["meta"].get("breadcrumb", ""),
                    "similarity": c["similarity"],
                })

        if sources:
            with st.expander(f"📄 {len(sources)} source page(s)", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="src-card"><a href="{src["url"]}" target="_blank">{src["title"]}</a><br>'
                        f'<span class="src-nav">{src["breadcrumb"]}</span> '
                        f'<span class="src-pct">· {src["similarity"]:.0%} match</span></div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant", "content": full_answer, "sources": sources,
    })
