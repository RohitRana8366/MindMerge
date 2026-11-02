"""
MindMerge — Personal RAG Assistant
A single-file Streamlit prototype demonstrating Retrieval-Augmented Generation
over a user's own documents (PDF / TXT / MD), built with LangChain.

Pipeline:
  Documents -> chunk -> embed locally (sentence-transformers/all-MiniLM-L6-v2)
            -> store in an in-memory FAISS index -> retrieve top-k on query
            -> ground a Groq-hosted LLM (openai/gpt-oss-120b) answer in the
               retrieved context.

No external servers, no separate backend/frontend — everything runs inside
this one Streamlit process. Vector index lives in memory for the session.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# ─────────────────────────────────────────────────────────────────────────
# Page config + styling
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindMerge · Personal RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top left, #1b1035 0%, #0b0a1a 55%, #060612 100%);
        color: #EDEBFF;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150e2e 0%, #0d0a20 100%);
        border-right: 1px solid rgba(168, 132, 255, 0.15);
    }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(124, 58, 237, 0.35), rgba(56, 189, 248, 0.18));
        border: 1px solid rgba(168, 132, 255, 0.35);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin: 0;
        background: linear-gradient(90deg, #C4B5FD, #7DD3FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #C9C4E8;
        margin-top: 0.4rem;
        font-size: 0.98rem;
    }

    .stat-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.22);
        border: 1px solid rgba(168, 132, 255, 0.4);
        font-size: 0.8rem;
        margin-right: 0.5rem;
        color: #DCD6FF;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(168, 132, 255, 0.15);
        border-radius: 16px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
    }

    .source-card {
        background: rgba(56, 189, 248, 0.07);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #B9E2F5;
    }
    .source-card b { color: #7DD3FC; }

    .stButton>button {
        background: linear-gradient(90deg, #7C3AED, #38BDF8);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }

    .stFileUploader, .stTextInput>div>div>input {
        border-radius: 10px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


@st.cache_resource(show_spinner=False)
def load_embedder():
    """Loads the sentence-transformers model once per app process."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_llm(groq_api_key: str):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
        temperature=0.2,
    )


def load_documents(uploaded_files):
    """Reads uploaded files into LangChain Document objects."""
    docs = []
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(tmp_path).load()
            else:  # .txt, .md and other plain-text formats
                loaded = TextLoader(tmp_path, encoding="utf-8").load()
            for d in loaded:
                d.metadata["source"] = f.name
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Could not read **{f.name}**: {e}")
        finally:
            os.unlink(tmp_path)
    return docs


def build_index(uploaded_files):
    with st.spinner("Reading files, chunking, and embedding locally…"):
        raw_docs = load_documents(uploaded_files)
        if not raw_docs:
            st.error("No readable content found in the uploaded files.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)

        embedder = load_embedder()
        st.session_state.vectorstore = FAISS.from_documents(chunks, embedder)
        st.session_state.chunk_count = len(chunks)
        st.session_state.indexed_files = [f.name for f in uploaded_files]
        st.session_state.messages = []
    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")


PROMPT = ChatPromptTemplate.from_template(
    """You are MindMerge, a personal knowledge assistant. Answer the user's
question using ONLY the context below, drawn from their own documents.
If the context does not contain the answer, say you don't have enough
information in the indexed documents rather than guessing.

Context:
{context}

Question: {question}

Give a clear, well-organized answer grounded in the context above."""
)


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source', 'file')}] {d.page_content}" for d in docs)


def get_rag_chain(groq_api_key: str, k: int = 4):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
    llm = load_llm(groq_api_key)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain


# ─────────────────────────────────────────────────────────────────────────
# Sidebar — document management
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 MindMerge")
    st.caption("Your second brain, powered by RAG")

    st.markdown("---")
    st.markdown("**1. Upload your knowledge base**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⚡ Process Documents", use_container_width=True, disabled=not uploaded_files):
        build_index(uploaded_files)

    if st.session_state.vectorstore is not None:
        st.markdown(
            f'<span class="stat-pill">📄 {len(st.session_state.indexed_files)} files</span>'
            f'<span class="stat-pill">🧩 {st.session_state.chunk_count} chunks</span>',
            unsafe_allow_html=True,
        )
        with st.expander("Indexed files"):
            for name in st.session_state.indexed_files:
                st.write(f"• {name}")

        if st.button("🗑️ Clear index", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.indexed_files = []
            st.session_state.chunk_count = 0
            st.rerun()

    st.markdown("---")
    st.markdown("**2. LLM connection**")
    env_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API key",
        value=env_key,
        type="password",
        help="Loaded from .env if present. You can also paste one here for this session only.",
    )
    if groq_api_key:
        st.caption("✅ Groq key detected")
    else:
        st.caption("⚠️ Add a key to .env or paste one above to enable answers")

    top_k = st.slider("Chunks retrieved per question (k)", 2, 8, 4)

    st.markdown("---")
    st.caption(
        "Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (runs locally)\n\n"
        "LLM: Groq `openai/gpt-oss-120b` · Vector store: FAISS (in-memory)"
    )

# ─────────────────────────────────────────────────────────────────────────
# Main area — hero + chat
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>MindMerge</h1>
        <p>Ask questions across your notes, PDFs, and bookmarks — get grounded answers, not generic web trivia.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.info("👈 Upload documents and click **Process Documents** in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources used"):
                    for s in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card"><b>{s["source"]}</b><br>{s["snippet"]}</div>',
                            unsafe_allow_html=True,
                        )

    question = st.chat_input("Ask something about your documents…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not groq_api_key:
                answer = "Please provide a Groq API key in the sidebar to generate answers."
                st.markdown(answer)
                sources = []
            else:
                with st.spinner("Retrieving relevant context and generating an answer…"):
                    try:
                        retriever, chain = get_rag_chain(groq_api_key, k=top_k)
                        docs = retriever.invoke(question)
                        answer = chain.invoke(question)
                    except Exception as e:
                        answer = f"Something went wrong calling the LLM: {e}"
                        docs = []
                st.markdown(answer)
                sources = [
                    {
                        "source": d.metadata.get("source", "unknown"),
                        "snippet": d.page_content[:220].replace("\n", " ") + "…",
                    }
                    for d in docs
                ]
                if sources:
                    with st.expander("📚 Sources used"):
                        for s in sources:
                            st.markdown(
                                f'<div class="source-card"><b>{s["source"]}</b><br>{s["snippet"]}</div>',
                                unsafe_allow_html=True,
                            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
