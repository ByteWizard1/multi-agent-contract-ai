## Agentic-Ai

### Overview

**Agentic-Ai** is an AI backend for contract analysis built with a multi‑agent architecture.  
It answers natural‑language questions about contracts by retrieving relevant clauses, reasoning over them with an LLM, evaluating answer quality, optionally self‑correcting, and sending email alerts for critical risks.

High‑level pipeline:
- **Retrieval Agent** → gets relevant clauses from a Pinecone vector database
- **Reasoning Agent** → uses Claude (via AWS Bedrock) to draft an answer grounded in those clauses
- **Evaluation Agent** → scores the answer on factuality, completeness, and reliability (FRAMES)
- **Self‑Correction Agent** → improves low‑factuality answers
- **Alert Agent** → sends email alerts via AWS SES when risk keywords are detected

The main API is exposed via FastAPI and containerized with Docker.

---

### Core technologies

- **Backend**
  - FastAPI (ASGI app)
  - Uvicorn (server)
  - Pydantic v2 (request/response models)
- **LLM & embeddings (AWS Bedrock)**
  - Claude 3.5 Sonnet (`anthropic.claude-3-5-sonnet-20241022-v2:0`) for reasoning, evaluation, and self‑correction
  - Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) for semantic embeddings
- **Vector database**
  - Pinecone (`pinecone-client==3.x`) for storing and searching contract clause embeddings
- **Agents & tools**
  - LangChain `Tool` wrappers for retrieval and reasoning agents
- **Email / alerts**
  - AWS SES (Simple Email Service) for alert emails
- **Infrastructure**
  - Docker + Docker Compose
  - Structured JSON logging

---

### Repository structure

```text
.
├─ backend/
│  ├─ app/
│  │  ├─ main.py                 # FastAPI application entrypoint
│  │  ├─ routers/
│  │  │  ├─ health_router.py     # Health check endpoint
│  │  │  └─ query_router.py      # /query endpoint (agent pipeline orchestration)
│  │  ├─ agents/
│  │  │  ├─ retrieval_agent.py   # Pinecone + embeddings retrieval
│  │  │  ├─ reasoning_agent.py   # LLM reasoning using Claude
│  │  │  ├─ evaluation_agent.py  # FRAMES evaluation
│  │  │  ├─ self_correction_agent.py # Self‑correction using evaluation feedback
│  │  │  └─ alert_agent.py       # Alerting logic using SES
│  │  ├─ services/
│  │  │  ├─ bedrock_client.py    # AWS Bedrock runtime client
│  │  │  ├─ pinecone_client.py   # Pinecone client wrapper
│  │  │  └─ ses_client.py        # AWS SES client wrapper
│  │  ├─ models/
│  │  │  ├─ query_models.py      # Request models (e.g., QueryRequest)
│  │  │  └─ response_models.py   # Response models (clauses, evaluation, etc.)
│  │  └─ utils/
│  │     ├─ config.py            # Settings loaded from environment variables
│  │     └─ logger.py            # JSON structured logging
│  ├─ docker-compose.yml         # Backend service definition
│  ├─ Dockerfile                 # Backend Docker image
│  └─ requirements.txt           # Backend Python dependencies
├─ data_pipeline/
│  └─ ingestion/
│     ├─ create_index.py         # Pinecone index creation script
│     ├─ ingest.py               # Ingests CSV clauses into Pinecone via Bedrock embeddings
│     ├─ legal_docs.csv          # Sample/legal clauses dataset (example data)
│     └─ ingested_preview.json   # Preview of ingested records
└─ README.md
```

---

### API behavior

#### `POST /query`

- **Request body** (`QueryRequest`):
  - `question: string` – user’s natural‑language question, e.g.  
    `"What are the termination clauses in the contract?"`
- **Processing steps**:
  1. **Retrieval**  
     - Generate embedding for `question` via Titan Embeddings  
     - Query Pinecone for top‑K similar clauses (`TOP_K_CLAUSES` in `config.py`)
  2. **Reasoning**  
     - Send query + retrieved clauses to Claude Sonnet  
     - Produce a draft answer
  3. **Evaluation (FRAMES)**  
     - Ask Claude to evaluate factuality, completeness, reliability (0.0–1.0)
     - Parse JSON scores into `EvaluationResult`
  4. **Self‑correction (optional)**  
     - If `factuality < FACTUALITY_THRESHOLD` (default 0.7), generate a corrected answer
  5. **Alerting (optional)**  
     - If query/answer contain risk keywords (e.g. `"termination"`, `"breach"`, `"penalty"`), send an email alert via SES
- **Response body** (`QueryResponse`):
  - `final_answer: string`
  - `retrieved_clauses: RetrievedClause[]`
  - `draft_answer: string | null`
  - `evaluation: EvaluationResult | null`
  - `self_corrected: boolean`
  - `alert_triggered: boolean`

---

### Environment configuration

The backend reads configuration from environment variables via `config.py`:

- **AWS / Bedrock / SES**
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION` (default: `us-east-1`)
  - `SES_FROM_EMAIL` (sender + default recipient for alerts)
- **Pinecone**
  - `PINECONE_API_KEY` (required for any retrieval)
  - `PINECONE_INDEX_NAME` (default: `contract_clauses_dataset`)
- **Models / behavior**
  - `BEDROCK_LLM_MODEL_ID` (Claude model ID)
  - `BEDROCK_EMBEDDING_MODEL_ID` (Titan embeddings model ID)
  - `FACTUALITY_THRESHOLD` (float, default `0.7`)
  - `TOP_K_CLAUSES` (int, default `5`)
- **Logging**
  - `LOG_LEVEL` (e.g. `INFO`, `DEBUG`)

When running via Docker Compose (from `backend/`), these are passed through from your local environment:

```yaml
environment:
  - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
  - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
  - AWS_REGION=us-east-1
  - PINECONE_API_KEY=${PINECONE_API_KEY}
  - LOG_LEVEL=INFO
```

Set them in your shell or via a `.env` file before starting the stack.

---

### Running the backend with Docker

From the `backend/` directory:

```bash
docker-compose up --build
```

This will:
- Build the backend image using `Dockerfile`
- Start the `agentic-ai-backend` container on port `8000`
- Mount the `app/` directory into the container for live‑code reloads (when used with dev configs)

Once running:
- FastAPI will be available at `http://localhost:8000`
- Open `http://localhost:8000/docs` for interactive Swagger UI

To stop:

```bash
docker-compose down
```

---

### Running ingestion scripts (optional)

The ingestion utilities under `data_pipeline/ingestion/`:

- Create a Pinecone index (if needed) using Bedrock embeddings
- Read contract clauses from `legal_docs.csv`
- Generate embeddings (via Bedrock)
- Upsert them into Pinecone

Usage pattern (conceptual):

```bash
cd data_pipeline/ingestion
python create_index.py   # create Pinecone index if not present
python ingest.py         # embed and upsert contract clauses
```

You must configure:
- `PINECONE_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

---

### Development notes

- Logging is structured JSON, suited for log aggregation systems.
- Agents are written as plain Python classes and some expose LangChain `Tool` wrappers, so you can:
  - Keep using the explicit pipeline in `query_router.py`, or
  - Plug them into a higher‑level LangChain `AgentExecutor` later.
- Tests under `backend/tests/` cover basic health and model validations; you can extend them with pipeline‑level tests.

---
