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

## 5. Running Tests

Run the dedicated RAG pipeline test suite:
```powershell
.venv\Scripts\pytest.exe tests/test_rag_pipeline.py -v
```

Run the combined retrieval, demo, and RAG suites:
```powershell
.venv\Scripts\pytest.exe tests/test_retrieval_bm25.py tests/test_retrieval_demo.py tests/test_rag_pipeline.py -v
```
