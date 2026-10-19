🚀 Nexora AI

AI-Powered Research Intelligence Platform

Nexora AI is an advanced AI-powered research intelligence platform designed to help researchers, students, developers, and knowledge workers understand, analyze, compare, and discover insights across scientific literature.

Instead of treating research papers as isolated documents, Nexora AI aims to build an intelligent layer that connects papers, concepts, methods, datasets, algorithms, findings, limitations, and research directions.

«Research Papers → Evidence → Understanding → Connections → Research Intelligence»

---

🧠 Vision

Scientific research is growing faster than humans can efficiently read and connect.

Nexora AI aims to transform large collections of research papers into an intelligent research environment where users can:

- 🔎 Search research literature semantically
- 📄 Understand research papers automatically
- 🧩 Extract important research information
- 🔬 Compare multiple papers
- 📚 Ask questions and receive evidence-grounded answers
- 🔗 Discover relationships between research concepts
- 📈 Analyze research trends
- 🕳️ Identify documented research gaps
- 🧠 Build research knowledge graphs
- 📝 Generate structured research plans

The long-term goal is to create a research intelligence system rather than a simple chatbot.

---

✨ Core Capabilities

1. 📄 Research Paper Intelligence

Nexora AI will process research papers and extract structured information such as:

- Research problem
- Objectives
- Methodology
- Algorithms
- Models
- Datasets
- Experiments
- Evaluation metrics
- Results
- Limitations
- Future work
- References
- Authors
- Publication metadata

---

2. 🔎 Semantic Research Search

Traditional keyword search can miss relevant research.

Nexora AI will use embedding-based semantic retrieval to understand the meaning of research queries.

Example:

Query:
"How are transformer models used for medical image analysis?"

Nexora AI:
→ Finds semantically relevant papers
→ Retrieves relevant sections
→ Ranks supporting evidence
→ Provides citations

---

3. 🤖 Evidence-Grounded Research Assistant

Users will be able to ask questions about their research collection.

Example:

What datasets are commonly used for this problem?

Which papers use transformer-based approaches?

What limitations are repeatedly mentioned?

How do these two methodologies differ?

The system will retrieve relevant evidence before generating an answer.

Design principle

«Retrieve evidence first. Generate conclusions second.»

This helps reduce unsupported AI-generated claims.

---

🔬 Research Intelligence

Nexora AI is designed to go beyond basic Retrieval-Augmented Generation (RAG).

The planned intelligence layer includes:

📊 Paper Analysis

Automatically structure individual papers into research components.

🔄 Paper Comparison

Compare papers across:

- Methodology
- Algorithms
- Datasets
- Experiments
- Results
- Evaluation metrics
- Limitations

🕳️ Research Gap Discovery

Analyze documented:

- Limitations
- Future work
- Missing experiments
- Dataset limitations
- Methodological limitations

Nexora AI will distinguish between:

Documented research gap
        ↓
Supported by paper evidence

AI-generated research suggestion
        ↓
Requires further validation

The system should not invent research gaps and present them as established facts.

📈 Research Trend Analysis

Analyze collections of papers to identify:

- Frequently occurring topics
- Emerging concepts
- Popular methodologies
- Algorithms
- Datasets
- Research directions

---

🤖 Planned AI Agent Architecture

Nexora AI will eventually use specialized AI agents instead of relying on one general-purpose assistant.

🔎 Research Retrieval Agent

Responsible for:

- Semantic search
- Evidence retrieval
- Relevant document selection

📄 Paper Analyst Agent

Responsible for:

- Paper understanding
- Methodology extraction
- Dataset identification
- Results analysis
- Limitation extraction

⚖️ Paper Comparison Agent

Responsible for:

- Multi-paper comparison
- Methodology comparison
- Dataset comparison
- Result comparison

🕳️ Research Gap Agent

Responsible for:

- Finding documented limitations
- Analyzing future work
- Identifying research opportunities
- Separating evidence from AI suggestions

📈 Research Trend Agent

Responsible for:

- Topic analysis
- Keyword analysis
- Methodology trends
- Research evolution

📝 Research Planner Agent

Responsible for helping construct:

- Research problems
- Research questions
- Literature review direction
- Dataset strategy
- Methodology
- Experiment plan
- Evaluation metrics

---

🧠 Knowledge Graph

A future version of Nexora AI will represent research relationships as a knowledge graph.

Example:

Paper
  │
  ├── uses → Algorithm
  │             │
  │             └── belongs to → Method
  │
  ├── uses → Dataset
  │
  ├── studies → Research Topic
  │
  ├── cites → Paper
  │
  └── reports → Result

This enables Nexora AI to reason about relationships between research entities rather than treating documents as independent text files.

---

🏗️ System Architecture

                 ┌─────────────────────────┐
                 │       User / UI         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      API Layer          │
                 │       FastAPI           │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Research Intelligence  │
                 │        Layer            │
                 └────────────┬────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │ Retrieval      │  │ Paper Analysis │  │ Agent System   │
 │ Agent          │  │ Agent          │  │                │
 └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                 ┌─────────────────────────┐
                 │       RAG Pipeline      │
                 └────────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌────────────┐   ┌────────────┐
        │ Embedding│    │ Vector DB  │   │ Knowledge  │
        │ Model    │    │            │   │ Graph      │
        └──────────┘    └────────────┘   └────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │   Research Documents    │
                 │          PDFs            │
                 └─────────────────────────┘

---

🔄 Research Processing Pipeline

Research Paper
      ↓
PDF Ingestion
      ↓
Text / Tables / Metadata Extraction
      ↓
Cleaning & Preprocessing
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
RAG
      ↓
Research Intelligence
      ↓
Agents + Knowledge Graph
      ↓
Evidence-Grounded Research Output

---

🛠️ Technology Stack

The technology stack will evolve as the system grows.

Programming

- Python

Backend

- FastAPI
- Pydantic

Data Processing

- Pandas
- NumPy
- scikit-learn

Document Processing

- PyMuPDF
- PDF extraction
- Text preprocessing

AI / ML

- Large Language Models
- Embedding models
- Retrieval-Augmented Generation
- Semantic search
- Agentic AI
- Model evaluation
- Domain adaptation
- Future experimentation with LoRA / PEFT

Retrieval

- Vector embeddings
- FAISS initially
- Future vector database integration

Database

- PostgreSQL

Knowledge Representation

- Knowledge Graph technology

Frontend

Planned:

- React

API

- REST API
- FastAPI

DevOps

Planned:

- Docker
- GitHub Actions
- Automated testing
- Cloud deployment

---

📁 Project Structure

The repository will gradually evolve toward:

Nexora-AI/
│
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
│
├── docs/
│   ├── architecture.md
│   └── roadmap.md
│
├── src/
│   └── nexora/
│       ├── __init__.py
│       ├── ingestion/
│       ├── preprocessing/
│       ├── retrieval/
│       ├── intelligence/
│       ├── agents/
│       ├── knowledge_graph/
│       ├── evaluation/
│       └── api/
│
├── tests/
│
├── data/
│
└── .github/
    └── workflows/

The structure will be expanded only when each component becomes necessary.

---

🧪 Evaluation & Reliability

Nexora AI is designed with strict reliability, verification, and evaluation mechanisms across the retrieval-to-answer pipeline:

### 1. Retrieval Hardening & Query Caching
- **Vector Validation**: Query vectors are validated for type safety (must be `list[float]`), non-emptiness, and finite numeric values (`math.isfinite`).
- **Dimension Matching**: Query vector dimensions are strictly checked against the expected vector store / embedding service dimension to fail fast before search operations.
- **Query Embedding Caching**: In-memory query vector caching in `RetrievalService` avoids redundant embedding API calls for repeated queries.
- **Failure Isolation**: Infrastructure failures (e.g., vector store unreachable, timeout) propagate cleanly to `ResearchServiceError` and map to safe HTTP 500 responses without exposing tracebacks or internal credentials.

### 2. Evidence Grounding & Citation Filtering
- **Strict Citation Grounding**: The LLM is provided with numbered evidence blocks (`[E1]`, `[E2]`, etc.). Post-generation citation filtering (`filter_unsupplied_citations`) automatically strips any hallucinated citation labels outside the supplied range.
- **Zero-Evidence Safety**: When no relevant documents are retrieved, the prompt explicitly instructs the LLM that no evidence was found, and any hallucinated citations are automatically removed.
- **Source Traceability**: Every evidence item preserves `document_id`, `chunk_id`, `paper_id`, relevance score, and filename, enabling complete traceability from citations back to underlying PDF chunks.
- **Untrusted Content Containment**: Retrieved paper text is treated as untrusted data and strictly isolated within the prompt context, preventing prompt injection attacks from hijacking system instructions.
- **Citation Semantics & Scope**: Citation validity and filtering verify citation-reference integrity (that generated citation labels like `[E1]` syntactically resolve to supplied evidence chunks). They do not constitute semantic natural language inference or prove full factual correctness of generated claims.

### 3. Quantitative Evaluation Suite (`nexora.retrieval.evaluation`)
A deterministic benchmark evaluation suite provides reproducible, quantitative metrics:
- **Retrieval Metrics**:
  - **MRR (Mean Reciprocal Rank)**: Measures rank of the first relevant document chunk.
  - **Recall@K & Precision@K**: Evaluates retrieval accuracy across top-$K$ candidates.
  - **Empty Retrievals**: Tracks occurrences of queries yielding no relevant chunks.
- **Citation Reference Metrics**:
  - **Citation Validity Rate**: Fraction of cited labels that map to valid, supplied evidence chunks (measures reference resolution integrity; does not prove semantic claim support).
  - **Evidence Coverage Rate**: Fraction of supplied evidence items referenced in the answer.
  - **Grounded Answers Rate (Reference Grounding)**: Percentage of answers containing zero hallucinated or unsupplied citation labels.

---

🔐 Security Principles

Security will be considered throughout development.

Secrets

API keys and credentials must never be committed to GitHub.

Use environment variables:

.env

and provide only safe templates:

.env.example

Data

Research documents should be handled carefully and access controls will be added as the platform evolves.

AI Safety

Nexora AI should distinguish between:

Evidence
   ↓
Documented fact

Analysis
   ↓
Reasoned interpretation

Suggestion
   ↓
AI-generated possibility

These should not be presented as equivalent.

---

📱 Phone-First Development Strategy

Nexora AI is being developed with a phone-first workflow.

The development environment will use cloud-based tools when local hardware is insufficient.

Phone

Used for:

- GitHub management
- Code editing
- Project management
- Testing
- Documentation
- Monitoring

GitHub

Used as the central project source of truth:

- Source code
- Documentation
- Issues
- Version history
- CI/CD
- Project roadmap

Cloud

Used when additional computing resources are required for:

- Model inference
- Embeddings
- Vector processing
- Experiments
- Deployment

---

🚧 Development Roadmap

Stage 0 — Foundation

- [x] Create GitHub repository
- [ ] Professional README
- [ ] Project architecture
- [ ] Repository structure
- [ ] Environment configuration
- [ ] Development documentation

---

Stage 1 — Document Intelligence

- [ ] PDF ingestion
- [ ] Text extraction
- [ ] Metadata extraction
- [ ] Text preprocessing
- [ ] Chunking
- [ ] Document validation

---

Stage 2 — RAG Engine

- [ ] Embedding generation
- [ ] Vector database
- [ ] Semantic search
- [ ] Retrieval pipeline
- [ ] LLM integration
- [ ] Evidence-based answers
- [ ] Source citations

---

Stage 3 — Research Intelligence

- [x] Structured paper analysis
- [x] Multi-paper comparison
- [ ] Research summaries
- [x] Limitation extraction
- [x] Future-work extraction
- [ ] Research trend analysis
- [ ] Documented research-gap analysis

---

Stage 4 — Agentic AI

- [ ] Retrieval Agent
- [ ] Paper Analyst Agent
- [ ] Comparison Agent
- [ ] Research Gap Agent
- [ ] Trend Agent
- [ ] Research Planner Agent
- [ ] Agent orchestration
- [ ] Tool calling

---

Stage 5 — Knowledge Graph

- [ ] Paper entities
- [ ] Author entities
- [ ] Dataset entities
- [ ] Algorithm entities
- [ ] Concept entities
- [ ] Citation relationships
- [ ] Research relationship graph

---

Stage 6 — Advanced AI

- [ ] RAG evaluation
- [ ] Retrieval evaluation
- [ ] Hallucination detection
- [ ] Answer quality evaluation
- [ ] Domain adaptation
- [ ] Advanced embedding experiments
- [ ] LoRA / PEFT experiments where practical

---

Stage 7 — Production Platform

- [ ] FastAPI backend
- [ ] React frontend
- [ ] PostgreSQL
- [ ] Authentication
- [ ] API architecture
- [ ] Docker
- [ ] Automated testing
- [ ] GitHub Actions
- [ ] Cloud deployment
- [ ] Monitoring

---

## 📡 Research Intelligence API Specification (Milestone 12C-24)

Nexora AI provides a robust, evidence-grounded research intelligence engine for single-paper deep analysis and multi-paper comparative synthesis. All extraction operations preserve strict citation traceability back to ingested paper chunks and prevent AI hallucinations when evidence is absent.

### Standardized Research Dimensions
Both analysis and multi-paper comparisons align across 7 standardized research dimensions:
1. `problem_and_objectives`: Core research question, problem statement, and primary objectives.
2. `methodology_and_algorithms`: Theoretical approach, algorithmic foundations, and system architecture.
3. `models_and_datasets`: Neural architectures, model parameters, pretraining/finetuning datasets, and splits.
4. `experiments_and_metrics`: Experimental design, baseline comparisons, evaluation benchmarks, and quantitative metrics.
5. `results_and_findings`: Key empirical results, performance gains, efficiency tradeoffs, and discoveries.
6. `limitations`: Explicitly acknowledged assumptions, hardware dependencies, and boundary conditions.
7. `future_work`: Documented next steps, open questions, and recommended directions.

### Citation & Evidence Traceability
- **Direct Reporting**: Each dimension contains an explicit `status` (`"reported"`, `"not_reported"`, or `"uncertain"`).
- **Zero Hallucination Guarantee**: When evidence is missing or a paper contains no chunks, the dimension reports `status: "not_reported"` and an empty citation list without fabricating claims.
- **Traceable Attribution**: Each reported claim includes `evidence_labels` (e.g., `["[E1]", "[E2]"]`) and a `sources` array linking directly to chunk IDs, filenames, page numbers, and paper IDs.
- **Comparative Segregation**: When comparing multiple papers, each paper's individual extracted facts and citations are preserved under `per_paper[paper_id]`, strictly segregating original paper claims from the model's cross-paper synthesis (`comparison_summary`, `key_differences`, `key_similarities`).

---

### Endpoints

#### 1. Structured Paper Analysis — `POST /api/v1/papers/analyze`
Analyze a paper by specifying its ID in the JSON body.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/papers/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "options": {
      "model": "gpt-4o",
      "temperature": 0.0
    }
  }'
```

**Response (`200 OK`):**
```json
{
  "paper_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  "dimensions": {
    "problem_and_objectives": {
      "summary": "Scaling sequence models efficiently to long contexts without quadratic attention costs.",
      "status": "reported",
      "evidence_labels": ["[E1]"],
      "sources": [
        {
          "chunk_id": "chunk-uuid-1",
          "paper_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
          "filename": "mamba.pdf",
          "text": "Selective state spaces achieve linear-time computation for sequence modeling."
        }
      ],
      "details": {
        "computational_complexity": "O(N)"
      }
    },
    "methodology_and_algorithms": {
      "summary": "State space duality and selective scan mechanisms.",
      "status": "reported",
      "evidence_labels": ["[E1]"],
      "sources": [],
      "details": {}
    },
    "models_and_datasets": { "..." },
    "experiments_and_metrics": { "..." },
    "results_and_findings": { "..." },
    "limitations": { "..." },
    "future_work": {
      "summary": "Not reported in the provided evidence.",
      "status": "not_reported",
      "evidence_labels": [],
      "sources": [],
      "details": {}
    }
  },
  "evidence_count": 8,
  "model": "gpt-4o",
  "usage": {
    "input_tokens": 1240,
    "output_tokens": 380,
    "total_tokens": 1620
  }
}
```

#### 2. Structured Paper Analysis by Path — `POST /api/v1/papers/{paper_id}/analyze`
Convenience endpoint accepting `paper_id` in the URL path.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/papers/3fa85f64-5717-4562-b3fc-2c963f66afa6/analyze" \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### 3. Multi-Paper Comparison — `POST /api/v1/papers/compare`
Perform cross-paper analysis comparing 2 to 5 papers across all standardized dimensions.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/papers/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_ids": [
      "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "8ba62f12-2281-4231-90be-8d951e73bfa9"
    ],
    "options": {
      "model": "gpt-4o",
      "temperature": 0.0
    }
  }'
```

**Response (`200 OK`):**
```json
{
  "paper_ids": [
    "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "8ba62f12-2281-4231-90be-8d951e73bfa9"
  ],
  "dimensions": {
    "methodology_and_algorithms": {
      "dimension": "methodology_and_algorithms",
      "comparison_summary": "Paper 1 utilizes selective state space models (SSMs) whereas Paper 2 optimizes exact attention via GPU memory hierarchy tiling.",
      "key_differences": [
        "Paper 1 replaces attention with a linear recurrent mechanism.",
        "Paper 2 preserves standard softmax attention while avoiding HBM roundtrips."
      ],
      "key_similarities": [
        "Both target sub-quadratic scaling or significant throughput acceleration on long sequences."
      ],
      "per_paper": {
        "3fa85f64-5717-4562-b3fc-2c963f66afa6": {
          "summary": "Selective scan algorithm operating in linear time.",
          "status": "reported",
          "evidence_labels": ["[E1]"],
          "sources": [],
          "details": {}
        },
        "8ba62f12-2281-4231-90be-8d951e73bfa9": {
          "summary": "Tiled exact attention executed in SRAM blocks.",
          "status": "reported",
          "evidence_labels": ["[E2]"],
          "sources": [],
          "details": {}
        }
      }
    },
    "problem_and_objectives": { "..." },
    "models_and_datasets": { "..." },
    "experiments_and_metrics": { "..." },
    "results_and_findings": { "..." },
    "limitations": { "..." },
    "future_work": { "..." }
  },
  "overall_summary": "Both architectures advance efficient long-context sequence modeling from complementary angles: algorithmic recurrence (Mamba) versus hardware-aware kernel execution (FlashAttention).",
  "model": "gpt-4o",
  "usage": {
    "input_tokens": 2850,
    "output_tokens": 720,
    "total_tokens": 3570
  }
}
```

---

### Error Handling & Safeguards
- **`400 Bad Request`**: Returned for invalid UUIDs, less than 2 papers or duplicate paper IDs in comparison requests, or bounds violations (`code: "invalid_request"`).
- **`401 Unauthorized`**: Returned when `enforce_auth=True` and caller provides missing or invalid credentials (`code: "unauthorized"`).
- **`403 Forbidden`**: Returned when attempting to access a workspace, collection, paper, or document owned by another user (`code: "forbidden"`).
- **`404 Not Found`**: Returned if a requested paper or document UUID does not exist in relational persistence (`code: "not_found"`).
- **`413 Payload Too Large`**: Returned when uploaded files or remote PDFs exceed the 50 MB size limit (`code: "file_too_large"`).
- **`422 Unprocessable Entity`**: Returned by FastAPI validation for malformed JSON bodies or unsupported parameter types.
- **`429 Too Many Requests`**: Returned when inbound rate limits are exceeded, providing `Retry-After` and `X-RateLimit-*` headers (`code: "rate_limited"`).
- **`500 Internal Server Error`**: Wrapped application error envelopes (`analysis_failed`, `comparison_failed`, `discovery_failed`) returned when upstream services fail, without exposing credentials, tracebacks, or private internal state.
- **`502 Bad Gateway` / `504 Gateway Timeout`**: Returned for upstream scholarly provider network dropped connections or request timeouts.
- **Prompt Injection Containment**: Ingested document chunks are framed inside demarcated untrusted data blocks within the prompt to prevent instructions in scientific texts from hijacking the LLM extraction logic.

---

### Security, Authentication & Production Deployment Configuration

#### 0. Production Startup & Container Deployment
- **Production ASGI Factory**:
  ```bash
  uvicorn nexora.server:create_production_app --factory --host 0.0.0.0 --port 8000
  ```
  *(or `uvicorn nexora.server:app --host 0.0.0.0 --port 8000`)*

- **Containerized Deployment (Docker)**:
  Build and run the production image (Python 3.12 slim, non-root user `appuser:10001`):
  ```bash
  docker build -t nexora-ai:latest .
  docker run -d --name nexora-ai -p 8000:8000 --env-file .env nexora-ai:latest
  ```

- **Local & Staging Orchestration (Docker Compose)**:
  Launch the complete stack (PostgreSQL 16, Qdrant, migration runner, and Nexora AI backend):
  ```bash
  # 1. Copy sample environment and configure provider keys
  cp .env.example .env

  # 2. Boot stack in background
  docker compose up -d

  # 3. View status and logs
  docker compose ps
  docker compose logs -f app

  # 4. Tear down stack
  docker compose down
  ```
  *Architecture Note: In Compose, the `migrate` service runs `alembic upgrade head` to completion before `app` begins serving traffic. When `app` starts, its `/ready` probe automatically validates that the schema and migration head are consistent.*

- **Database Migrations**:
  Apply all schema migrations prior to launching application workers:
  ```bash
  alembic upgrade head
  ```
  *Note: The application creates connection pools lazily and connects to existing tables; it does not automatically execute DDL or schema migrations at runtime.*
  *In container environments, pre-flight migrations can be enabled via `RUN_MIGRATIONS=true` in `docker-entrypoint.sh`, or executed as a separate migration job/init-container.*

- **Qdrant Collection Initialization**:
  When running with `NEXORA_VECTOR_STORE=qdrant`, ensure the target collection exists with the vector dimensionality matching your embedding provider (e.g., 1536 for OpenAI `text-embedding-3-small`, 768 for Gemini, or 128 for `hash` / `hash-sha256` testing):
  *Note: The application connects to an existing collection and does not automatically create or resize collections during startup. If the target collection does not exist or has an incompatible dimensionality, vector store operations and `/ready` probes will fail closed with `QdrantStoreError` or dimension mismatch errors.*

- **Required Production Environment Variables**:
  - `NEXORA_DATABASE_URL` (or `DATABASE_URL`): PostgreSQL connection URL (e.g., `postgresql://user:password@localhost:5432/nexora`).
  - `NEXORA_LLM_API_KEY`: API key for the LLM provider (or `OPENAI_API_KEY`).
  - `NEXORA_LLM_MODEL`: Target model (e.g., `gpt-4o-mini`).
  - `NEXORA_VECTOR_STORE`: `qdrant` (or `memory` for ephemeral dev/testing).
  - `NEXORA_QDRANT_URL`: Required when vector store is `qdrant` (e.g., `http://localhost:6333`).
  - `NEXORA_EMBEDDING_PROVIDER`: `openai`, `gemini`, or `hash`.
  - `NEXORA_EMBEDDING_API_KEY`: Required if using external embedding providers (`openai` / `gemini`).
  - `APP_ENV`: Set to `production` in production deployments.

- **Reproducible Dependency Management**:
  - `requirements.txt`: Minimal runtime-only dependencies with validated compatibility bounds.
  - `requirements-dev.txt`: Development, linting, and testing dependencies (`-r requirements.txt` + `pytest`).
  - `requirements.lock`: Exact, byte-for-byte pinned versions used for reproducible production container builds.
  - To update locked dependencies: test compatibility in a clean virtual environment, run `pytest`, and freeze the tested packages into `requirements.lock`.

- **Fail-Safe Startup & Production Safeguards**:
  - `create_production_app` validates all required configuration at boot time and fails closed with descriptive errors, keeping credentials and passwords completely out of exception messages and logs.
  - The offline demo factory (`nexora.api.app:create_offline_app` and `nexora.api.app:app`) is safeguarded against accidental execution in production: if `APP_ENV=production`, `ENVIRONMENT=production`, or `NEXORA_APP_MODE=production` is detected, it raises `InvalidConfigurationError` immediately to prevent unauthenticated execution.

#### 1. API Authentication & Multi-Tenant Authorization
- **Credential Transport**: Pass credentials via `Authorization: Bearer <api_key>` or `X-API-Key: <api_key>`.
- **Hashed Storage**: Only cryptographic SHA-256 digests of API keys are stored in the database. Raw keys are never stored, logged, or exposed.
- **Ownership Verification**: All collection, workspace, document, paper, analysis, comparison, and verification endpoints strictly verify ownership before executing any reads or writes.
- **Production Default**: Production composition roots (`AppSettings.from_env()`) enforce authentication by default (`NEXORA_ENFORCE_AUTH=true`). Startup fails closed with `InvalidConfigurationError` if auth is enforced without database session or auth service infrastructure.
- **Offline / Local Mode**: Offline RAG pipelines (`create_offline_app`) run with `enforce_auth=False`, requiring zero credentials, external servers, or network access for testing.

#### 2. Scholarly Discovery & Safe Remote Ingestion
- **Endpoints**:
  - `GET /api/v1/discovery/search`: Query external scholarly literature (arXiv) with metadata normalization and category filtering.
  - `POST /api/v1/discovery/import`: Explicit import into a verified user-owned collection.
- **Defense-in-Depth Protections**:
  - **Scheme Enforcement**: HTTPS strictly enforced.
  - **Host Allowlisting**: Only verified scholarly hosts (`arxiv.org`, `export.arxiv.org`) permitted.
  - **SSRF & DNS Rebinding**: Resolves IP addresses before connecting and validates against loopback (`127.0.0.0/8`, `::1`), private networks (RFC 1918), link-local cloud metadata endpoints (`169.254.169.254`), and RFC 6052 NAT64 prefixes.
  - **Redirect Destination Validation**: Every 3xx redirect destination is re-validated before following (maximum 3 hops).
  - **Streaming Cap**: Preflights `Content-Length` and streams chunks with an absolute 50 MB cutoff.
  - **Content Validation**: Rejects HTML error/captcha responses; verifies `%PDF-` magic byte signatures.
  - **Lifecycle Cleanup**: Temporary download files are guaranteed to be cleaned up on disk on all success and failure paths.

#### 3. Inbound Rate Limiting & Abuse Prevention
- **Configurable Tiers**: Inbound rate limits (`NEXORA_RATE_LIMIT_*`) enforce sliding-window quotas based on endpoint cost:
  - `heavy_analysis` (10 req/min): Multi-paper comparisons, structured analysis, research gaps, claim verification.
  - `research_query` (30 req/min): Grounded hybrid RAG queries.
  - `discovery_search` (20 req/min): External arXiv search queries.
  - `discovery_import` (10 req/min): Safe PDF download, parsing, and collection import.
  - `document_ingest` (10 req/min): Direct document upload, parsing, and vector indexing.
  - `default` (120 req/min): Standard metadata reads.
  - Unmetered: `/health`, `/docs`.
- **Tamper-Resistant Caller Tracking**: Authenticated requests are keyed by a 16-character SHA-256 digest prefix of their API key (raw secrets are never logged or stored in counter keys). Unauthenticated requests fall back to client IP, with anti-spoofing protection (`X-Forwarded-For` only trusted from explicitly configured `NEXORA_RATE_LIMIT_TRUSTED_PROXIES`).
- **Multi-Worker & Distributed Enforcement**:
  - `memory` (default): In-process sliding-window counter for local development and unit tests. In multi-worker deployments (e.g. Uvicorn `--workers N`), each worker maintains its own counters.
  - `redis`: Distributed rate limiting backed by Redis via atomic Lua scripts (`_LUA_SLIDING_WINDOW`). Guarantees atomic prune-and-check operations without race conditions across multiple workers and container replicas.
  - **Configuration**: Set `NEXORA_RATE_LIMIT_BACKEND=redis` and configure `NEXORA_RATE_LIMIT_REDIS_URL` (or `REDIS_URL`, e.g. `redis://redis:6379/0`).
  - **Fail-Closed Security Policy**: `NEXORA_RATE_LIMIT_REDIS_FAIL_CLOSED=true` (default) ensures requests are rejected safely if Redis becomes unreachable or times out, preventing API abuse during infrastructure degradation.
  - **Fail-Fast Validation**: If `backend="redis"` is configured without a valid `redis_url` or injected storage, application startup fails closed immediately.

#### 4. CORS Security
- **Secure by Default**: CORS is disabled by default (`NEXORA_CORS_ENABLED=false`).
- **No Wildcard with Credentials**: The wildcard origin `*` is strictly forbidden when `NEXORA_CORS_ALLOW_CREDENTIALS=true`.
- **Origin Validation**: Permitted origins must explicitly specify `http://` or `https://` schemes.

#### 5. Database Connection Pooling & Concurrency
- **PostgreSQL Requirement**: In production, `DatabaseConfig` enforces a PostgreSQL dialect (`postgresql://` or `postgresql+<driver>://`), redacting credentials from logs and exceptions. SQLite is used for fast, isolated in-memory unit/integration test doubles via dedicated SQLite session factories.
- **Configuration Precedence**: `NEXORA_DATABASE_URL` takes precedence over `DATABASE_URL`.
- **Configurable Pool Parameters**:
  - `NEXORA_DATABASE_POOL_SIZE` (default: 5, recommended: 10–20 per ASGI worker process).
  - `NEXORA_DATABASE_MAX_OVERFLOW` (default: 10, handles burst concurrency).
  - `NEXORA_DATABASE_POOL_TIMEOUT` (default: 30.0s, wait time before raising on pool exhaustion).
  - `NEXORA_DATABASE_POOL_RECYCLE` (default: 1800s, recycles idle connections before backend server timeout).
- **Session Lifecycle**: Requests and worker jobs manage isolated sessions via the `database_transaction` context manager, ensuring commits only occur on clean exits and rollbacks/closes happen deterministically on any failure or cancellation.
- **Auto-Initialization**: When `database` configuration is provided in `AppSettings`, `build_http_app` automatically provisions the SQLAlchemy engine and `sessionmaker` factory.

#### 6. Health, Readiness & Liveness Architecture
- **Liveness Probe**: `GET /health`
  - Ultra-lightweight; touches zero network or database dependencies.
  - Returns `{"status": "ok"}` (HTTP 200) as long as the ASGI event loop and worker process are running.
- **Readiness Probe**: `GET /ready` (alias: `GET /health/ready`)
  - Deep dependency validation for container orchestrators and load balancers:
    - **Database**: Validates connectivity (`SELECT 1`), verifies presence of all 10 core schema tables, and confirms applied Alembic migration revision matches expected application heads (`044c24ad4c33`). Fails closed on missing tables, outdated revisions, or connection drops.
    - **Authentication**: Validates that required auth tables and service are operational when `enforce_auth=true`.
    - **Research Service**: Confirms RAG query handlers and LLM service are wired.
    - **Vector Store**: Verifies cluster connectivity via `vector_store.count()` and validates vector dimension compatibility against the configured embedding model.
  - Returns HTTP 200 (`status: "ready"`) when all dependencies respond.
  - Returns HTTP 503 (`status: "unavailable"`) if any required dependency degrades or schema mismatch is detected, preventing traffic routing to unmigrated or degraded replicas.

#### 7. Bounded Durable Document Recovery
- **Background Reindexing**: Interrupted or failed document ingestion is repaired via `DocumentRecoveryService.recover_batch()`.
- **Bounded Batch Execution**: Strictly enforces `1 <= batch_size <= 100` (`MAX_BATCH_SIZE`) to prevent synchronous event loop or worker thread starvation.
- **Crash Claiming**: Claims unindexed documents using timed leases (`claimed_at`, `claimed_by`) with automatic expiry (`claim_timeout_seconds=300.0`) so crashed workers release locks safely.
- **Idempotency**: Vectors are keyed deterministically by chunk UUID (`str(chunk.id)`), guaranteeing duplicate recovery runs overwrite in place without creating orphaned vectors.

#### 8. Request Correlation, Structured Logging & Graceful Lifecycle
- **Request Correlation**: Outermost ASGI `RequestIDMiddleware` validates incoming `X-Request-ID` / `X-Correlation-ID` headers against `^[a-zA-Z0-9_\-\.]{1,64}$`, generating secure `req_<hex>` identifiers when absent or invalid. Guarantees that `X-Request-ID` is echoed on all HTTP responses (including 200 OK, 4xx client errors, 429 rate limit exceeded, and 500 internal errors).
- **Context Variable Tracking**: Propagates request IDs across async coroutines and threadpools via Python `contextvars`, making correlation IDs instantly available to loggers and route handlers without manual plumbing.
- **Structured JSON Logging**: `StructuredJsonFormatter` produces single-line machine-readable JSON logs containing `timestamp` (ISO-8601 UTC), `level`, `severity`, `logger`, `request_id`, and `message`.
- **Sensitive Data Redaction**: Automatically scrubs Bearer tokens, API keys, passwords, database credentials, and Redis passwords from log messages and exception tracebacks prior to emission.
- **Graceful Lifespan Teardown**: ASGI `lifespan` handler deterministically cleans up owned resources on process termination:
  - Disposes database connection pool (`engine.dispose()`).
  - Closes vector store / retrieval clients.
  - Closes Redis rate limiting storage connections.
  - Closes discovery HTTP clients (`ArxivClient`, `SafePDFDownloader`).
  - Executes registered custom shutdown hooks (`app.state.shutdown_hooks`).
  - **Fault Isolation**: Each subsystem cleanup is isolated in its own `try...except` block, ensuring a failure or timeout in one resource does not abort the teardown of remaining components.

---

🎯 Project Objectives

Nexora AI aims to demonstrate practical skills in:

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Large Language Models
- Retrieval-Augmented Generation
- Semantic Search
- Vector Databases
- Agentic AI
- Knowledge Graphs
- Data Engineering
- Backend Engineering
- API Development
- Software Architecture
- AI Evaluation
- Production AI Systems

---

💡 Why Nexora AI?

Nexora AI is intentionally designed to demonstrate more than basic AI API usage.

The project focuses on:

AI Models
     +
Data Processing
     +
Retrieval
     +
Reasoning
     +
Agents
     +
Knowledge Graphs
     +
Evaluation
     +
Production Engineering

The goal is to build the system incrementally and understand every major component rather than hiding the complexity behind a single AI API.

---

🧑‍💻 Development Philosophy

Nexora AI follows these principles:

1. Evidence First

AI outputs should be grounded in available evidence whenever possible.

2. Modular Architecture

Each major capability should be independently testable.

3. Evaluation Before Complexity

New AI components should be evaluated before adding unnecessary complexity.

4. Security by Default

Secrets and credentials must never be committed to the repository.

5. Incremental Development

Build one working component at a time.

6. Research Integrity

The system should clearly distinguish between:

- Evidence
- Analysis
- Hypothesis
- AI-generated suggestion

---

📊 Current Development Status

Project: Nexora AI
Version: 0.1.0
Status: 🚧 Active Development
Current Phase: Foundation / Architecture

Nexora AI is currently being developed incrementally, beginning with the core architecture and document intelligence pipeline.

---

🌟 Long-Term Vision

The long-term objective is to evolve Nexora AI into an intelligent research workspace capable of transforming large-scale scientific literature into structured, searchable, connected, and actionable research intelligence.

Millions of Research Documents
             ↓
      Document Intelligence
             ↓
       Semantic Retrieval
             ↓
         RAG Engine
             ↓
     Research Intelligence
             ↓
        AI Agents
             ↓
      Knowledge Graph
             ↓
      Research Planning
             ↓
   Human Researcher

---

📌 Important Note

Nexora AI is an evolving research engineering project.

AI-generated outputs should be independently verified before being used for academic, scientific, medical, financial, or other high-impact decisions.

---

👨‍💻 Author

Saiyed Simnan

IT Student | AI & Data Science Enthusiast

Interested in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Software Engineering
- Research Systems
- AI Applications

---

⭐ Project

If you find the project interesting, consider giving the repository a ⭐ on GitHub.

Nexora AI — From Research Papers to Research Intelligence.

---

📜 License

License will be selected as the project architecture and distribution model are finalized.
