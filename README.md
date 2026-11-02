# MindMerge — Personal RAG Assistant (Prototype)

A single-file Streamlit prototype for a Final Year Project on Retrieval-Augmented
Generation. Upload your own PDFs/notes, ask questions, get answers grounded in
your documents instead of generic web knowledge.

## ⚠️ Security first — do this before anything else

You pasted real API keys into a chat earlier. Treat them as compromised:

1. Go to **Google AI Studio** → API keys → delete/regenerate the Gemini key.
2. Go to **Hugging Face** → Settings → Access Tokens → revoke the old token
   (this prototype doesn't even need it, since embeddings run locally).
3. Put the **new** keys only in a local `.env` file (see below). Never commit
   `.env` to git or paste it into a chat/screenshot again — `.gitignore` it.

## Architecture

```
Uploaded files (PDF/TXT/MD)
        │
        ▼
RecursiveCharacterTextSplitter  (chunk_size=800, overlap=120)
        │
        ▼
HuggingFaceEmbeddings  (sentence-transformers/all-MiniLM-L6-v2, runs locally on CPU)
        │
        ▼
FAISS  (in-memory vector index, no external DB/server)
        │
        ▼
Retriever (top-k similarity search)
        │
        ▼
LangChain LCEL chain → Gemini 1.5 Flash → grounded answer + shown sources
```

Everything — UI, chunking, embedding, retrieval, and generation — runs inside
one Streamlit process. There's no separate backend server and no external
vector database; FAISS's index simply lives in memory for the session.

## Setup

```bash
cd mindmerge_rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your NEW Gemini API key

streamlit run app.py
```

The first run will download the `all-MiniLM-L6-v2` model (~90MB) from Hugging
Face automatically — after that it's cached locally and works fully offline
for embeddings (only the Gemini call needs internet).

## Using it

1. Open the sidebar, upload one or more PDF/TXT/MD files.
2. Click **Process Documents** — this chunks and embeds them into FAISS.
3. Confirm your Gemini API key is picked up (green check in the sidebar), or
   paste one directly for a quick test.
4. Ask questions in the chat box. Each answer shows the **source chunks**
   that were retrieved, so you can verify it's actually grounded.

## Notes for your report

- **Why local embeddings instead of an API call?** Avoids HF Inference API
  rate limits/model-availability issues and keeps the pipeline fully working
  offline for the embedding step — a common design tradeoff worth mentioning
  in your evaluation section.
- **Why FAISS over Pinecone for the prototype?** No account/API key/network
  dependency, in-memory, ideal for a local FYP demo. Swapping `FAISS.from_documents`
  for a `PineconeVectorStore` call is a one-line change if you want to extend
  this to a persistent, cloud-hosted store later.
- **Extending it:** chat memory across turns, multi-user persistence (save/load
  the FAISS index with `save_local`/`load_local`), citation highlighting, or
  swapping Gemini for a local model via Ollama are natural "future work" items.
