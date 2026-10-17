# End-to-End RAG Pipeline Guide

The **Nexora AI End-to-End RAG Pipeline** connects research document retrieval, evidence context formatting, answer generation, and reference verification into a unified, traceable research intelligence workflow.

---

## 1. Core Principles

- **Evidence First**: All generated answers are strictly grounded in retrieved evidence chunks.
- **Traceable Attribution**: Supporting research papers and chunk excerpts are tagged with citation labels (`[E1]`, `[E2]`, etc.).
- **Truthful Status**: Unanswerable queries or empty retrieval results report `INSUFFICIENT_EVIDENCE` without fabricating citations or hallucinating findings.
- **Explicit Mock vs. Live Demarcation**: Offline deterministic generation is clearly flagged with `is_mock: true` and `model: offline-deterministic-mock-v1`. Real LLM outputs report `is_mock: false`.
- **Zero-Dependency Offline Mode**: Works completely offline out of the box with zero external API keys, paid services, or network calls.

---

## 2. Architecture

```text
User Research Question
          │
          ▼
┌────────────────────────────────────────┐
│           RAG Pipeline                 │
├────────────────────────────────────────┤
│ 1. Retrieval Layer                     │
│    - Mode: 'bm25', 'hybrid', 'vector'  │
│    - Ranks & filters candidate chunks  │
├────────────────────────────────────────┤
│ 2. Evidence Context Builder            │
│    - Formats EvidenceItem & [E#] labels│
│    - Truthfully detects empty evidence │
├────────────────────────────────────────┤
│ 3. Answer Generation Provider          │
│    - DeterministicMockGenerator (100%  │
│      offline, CI/test ready)           │
│    - LLMAnswerGenerator (OpenAI / live)│
├────────────────────────────────────────┤
│ 4. Citation Verification               │
│    - Verifies [E#] syntactic integrity │
│    - Maps citations to AttributedSource│
└────────────────────────────────────────┘
          │
          ▼
Structured RAGResponse
  ├── answer (grounded text with [E#])
  ├── cited_sources (actively cited papers)
  ├── sources (all retrieved candidates)
  ├── is_grounded (verification status)
  └── is_mock (truthful provider flag)
```

---

## 3. Quickstart CLI Commands

### Direct Question (PowerShell)
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m nexora.rag.cli "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights?"
```

### Specify Retrieval Strategy & Top-K Candidates
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m nexora.rag.cli --query "What BLEU score did the Transformer achieve?" --retriever bm25 --top-k 3
```

### JSON Output Mode
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m nexora.rag.cli --query "FlashAttention SRAM HBM tiling" --json
```

### Interactive Question Shell
Launch the interactive question-and-answer prompt:
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m nexora.rag.cli
```

### List Curated Benchmark Questions
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m nexora.rag.cli --sample-questions
```

---

## 4. Python API Usage

```python
from nexora.rag import create_offline_rag_pipeline, RAGRequest

# 1. Initialize offline RAG pipeline over curated pilot benchmark
pipeline = create_offline_rag_pipeline(default_mode="bm25")

# 2. Ask a research question
response = pipeline.answer("How does LoRA freeze pre-trained weights and reduce trainable parameters?")

print(f"Status: {response.status}")
print(f"Answer:\n{response.answer}\n")
print(f"Grounded: {response.is_grounded}")

# 3. Inspect cited sources
for source in response.cited_sources:
    print(f"{source.label} - {source.paper_title}")
    print(f"  Chunk:   {source.chunk_id}")
    print(f"  Score:   {source.score}")
    print(f"  Excerpt: {source.text_excerpt}\n")
```

---

---

## 5. REST API Usage & Startup

The RAG pipeline is exposed via a REST HTTP API with comprehensive request validation, honest source attribution, and OpenAPI documentation.

### Starting the HTTP Server

Run the API server with Uvicorn (zero configuration, runs 100% offline):

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\uvicorn.exe nexora.api.app:app --host 127.0.0.1 --port 8000
```

Alternatively, use the explicit offline app factory:
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\uvicorn.exe "nexora.api.app:create_offline_app()" --factory --host 127.0.0.1 --port 8000
```

Once running, interactive OpenAPI / Swagger UI documentation is available at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **OpenAPI Schema**: `http://127.0.0.1:8000/openapi.json`

---

### Service Health Check

Check service health (`GET /health`):

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response (`200 OK`)**:
```json
{
  "status": "ok"
}
```

---

### Grounded Research Query (`POST /api/v1/research/query`)

Send a research question with optional retrieval strategy and parameters:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights and reduce trainable parameters?",
    "retrieval_mode": "hybrid",
    "top_k": 3
  }'
```

**Response (`200 OK`)**:
```json
{
  "query": "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights and reduce trainable parameters?",
  "answer": "Based on the retrieved research evidence, the following findings address your query:\n\nFrom \"LoRA: Low-Rank Adaptation of Large Language Models\" [E1]:\nLoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and GPU memory requirement by 3 times.\n\nSummary of Evidence:\n- [E1] LoRA freezes pre-trained weights and injects trainable rank decomposition matrices, reducing parameters by 10,000x.",
  "sources": [
    {
      "label": "[E1]",
      "document_id": "arxiv_2106.09685",
      "chunk_id": "arxiv_2106.09685_chunk_001",
      "paper_id": "arxiv_2106.09685",
      "paper_title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "score": 0.032787,
      "text": "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and GPU memory requirement by 3 times.",
      "text_excerpt": "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable...",
      "metadata": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis"],
        "year": 2021
      }
    }
  ],
  "cited_sources": [
    {
      "label": "[E1]",
      "document_id": "arxiv_2106.09685",
      "chunk_id": "arxiv_2106.09685_chunk_001",
      "paper_id": "arxiv_2106.09685",
      "paper_title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "score": 0.032787,
      "text": "LoRA freezes the pre-trained model weights...",
      "text_excerpt": "LoRA freezes the pre-trained model weights...",
      "metadata": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models"
      }
    }
  ],
  "is_grounded": true,
  "is_mock": true,
  "model": "offline-deterministic-mock-v1",
  "status": "success",
  "error": null,
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 84,
    "total_tokens": 212
  },
  "retrieval_mode": "hybrid",
  "evidence_count": 1
}
```

---

### Unanswerable Queries / Insufficient Evidence

When a query cannot be answered from the corpus:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the breeding cycle of emperor penguins in Antarctica during winter?",
    "retrieval_mode": "bm25"
  }'
```

**Response (`200 OK`)**:
```json
{
  "query": "What is the breeding cycle of emperor penguins in Antarctica during winter?",
  "answer": "I could not find sufficient evidence in the indexed research papers to answer this question.",
  "sources": [],
  "cited_sources": [],
  "is_grounded": true,
  "is_mock": true,
  "model": "offline-deterministic-mock-v1",
  "status": "insufficient_evidence",
  "error": null,
  "usage": null,
  "retrieval_mode": "bm25",
  "evidence_count": 0
}
```

No citations are fabricated and no false claims are returned.

---

## 6. Running Tests

Run the dedicated RAG API test suite:
```powershell
.venv\Scripts\pytest.exe tests/test_api_rag.py -v
```

Run all API test suites:
```powershell
.venv\Scripts\pytest.exe tests/test_api_app.py tests/test_api_routes.py tests/test_api_rag.py tests/test_api_document_ingest.py -v
```

Run the complete RAG and retrieval verification:
```powershell
.venv\Scripts\pytest.exe tests/test_retrieval_bm25.py tests/test_retrieval_demo.py tests/test_rag_pipeline.py tests/test_api_rag.py -v
```

