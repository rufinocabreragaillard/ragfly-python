# RAGfly Python SDK

Official Python client for [RAGfly](https://ragfly.ai) — retrieval infrastructure for your AI agents.

## Install

```bash
pip install ragfly
```

## Quick start

```python
from ragfly import RAGfly

client = RAGfly(api_key="rfly_...")

# Ask a question (RAG end-to-end)
resp = client.ask("What are the Q1 sales figures?")
print(resp.answer)

# Streaming
for chunk in client.ask("Summarize active contracts", stream=True):
    print(chunk.delta, end="", flush=True)

# Semantic search (retrieval only)
results = client.search("maintenance contracts", limit=5)
for doc in results.documents:
    print(doc.nombre, doc.similitud_max)
```

## API Keys

Generate an API key from [app.ragfly.ai](https://app.ragfly.ai) → Settings → API Keys.

## Methods

| Method | Description |
|--------|-------------|
| `client.ask(question, *, stream=False, conversation_id=None)` | RAG end-to-end: retrieve + generate |
| `client.search(query, *, limit=10, min_similitud=0.0)` | Hybrid retrieval (vector + lexical + rerank) |
| `client.list_documents(*, page=1, page_size=20)` | List corpus documents |

## Links

- Docs: https://api.ragfly.ai/docs
- Site: https://ragfly.ai
