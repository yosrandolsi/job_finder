# Architecture — Autonomous Job Application Agent

This document explains **how the system is built and why**, for the report and
the demo. It covers the agents, the messages they exchange, the orchestration
model, and the cross-cutting concerns (MCP, harness/hooks/subagents, security,
observability).

---

## 1. What the system does

Given a **resume** (PDF or image) and a **desired job title**, the system:

1. Extracts a structured candidate profile from the resume.
2. Searches for relevant job postings.
3. Scores and ranks every posting against the candidate.
4. **Pauses** and shows the full ranked list to the user.
5. Waits for the user to **select one job**.
6. Generates a tailored cover letter **only** for that selected job.

The point of the project is to demonstrate **multi-agent collaboration with a
human in the loop**, not to be a production app. Every design choice below
serves *traceability* and *explainability*.

---

## 2. High-level architecture

```
                          ┌───────────────────────────────────────────────┐
                          │              ORCHESTRATOR (LangGraph)          │
                          │   drives the pipeline, logs every A2A message  │
                          └───────────────────────────────────────────────┘
                                             │
   PDF / image  ─────────────►  ┌────────────▼─────────────┐
                                │  Profile Extraction Agent │  pdfplumber / PyPDF2 (PDF)
                                │                           │  Groq vision model (image)
                                └────────────┬─────────────┘  + Groq LLM structuring
                                             │ ProfileSchema
                                ┌────────────▼─────────────┐        ┌──────────────────────┐
                                │     Job Search Agent      │◄─MCP──►│  job-search MCP server│
                                │  (QueryNormalizerSubagent)│ stdio  │  Adzuna → Arbeitnow   │
                                └────────────┬─────────────┘        └──────────────────────┘
                                             │ list[JobPosting]
                                ┌────────────▼─────────────┐
                                │   Matching / Eval Agent   │  local embeddings (MiniLM)
                                │   (MatchScoringSubagent)  │  + Groq LLM judge per job
                                └────────────┬─────────────┘
                                             │ list[MatchResult] (ranked)
                                   ══════════▼══════════   ◄── PHASE A ENDS HERE
                                   ║   LangGraph INTERRUPT ║       (interrupt_before)
                                   ║   human selects a job ║   ── ranked list shown to user
                                   ══════════▲══════════   ◄── PHASE B RESUMES HERE
                                             │ selected_job_id
                                ┌────────────▼─────────────┐
                                │     Cover Letter Agent    │  Groq LLM
                                └────────────┬─────────────┘
                                             │ CoverLetterResult
                                             ▼
                                   Presentation (Streamlit / CLI)
```

---

## 3. Components and responsibilities

Each agent is a **separate module** with a single responsibility and a typed
input → typed output. They never share mutable state; they only pass Pydantic
messages. This is what makes the collaboration traceable.

| Component | File | Input → Output | Notes |
|-----------|------|----------------|-------|
| **Orchestrator** | `orchestrator/graph.py` | drives all agents | LangGraph state machine + Phase A/B interrupt |
| **Profile Extraction Agent** | `agents/profile_extraction.py` | `bytes + filename` → `ProfileSchema` | Multimodal: PDF text or vision-model image reading |
| **Job Search Agent** | `agents/job_search.py` | `ProfileSchema + title` → `list[JobPosting]` | Calls the MCP tool; normalizes the query first |
| **Matching / Eval Agent** | `agents/matching.py` | `ProfileSchema + list[JobPosting]` → `list[MatchResult]` | Embedding score + LLM-judge score per posting |
| **Cover Letter Agent** | `agents/cover_letter.py` | `CoverLetterRequest` → `CoverLetterResult` | Runs only in Phase B, for one selected job |
| **MCP job-search server** | `mcp_server/job_search_server.py` | `search_jobs(query, location)` tool | Adzuna primary, Arbeitnow fallback |
| **Presentation** | `ui/app.py`, `ui/cli.py` | — | Streamlit UI (primary) + CLI fallback |

**Shared infrastructure** (`common/`):

| Module | Purpose |
|--------|---------|
| `config.py` | Loads `.env` once for the whole process |
| `security.py` | Length caps, input sanitization, prompt-injection guard, log redaction |
| `llm_client.py` | Groq wrapper: multi-key fallback, JSON/text/vision calls, token metering |
| `embeddings.py` | Local sentence-transformers cosine similarity (no API key) |
| `harness.py` | Uniform agent execution wrapper + lifecycle hooks |
| `metrics.py` | Operational per-agent + per-LLM metrics, console summary |
| `quality.py` | Quality metrics (completeness, ranking separation, cover-letter checks) |
| `logging_utils.py` | Redacted A2A message trace |

---

## 4. Agent-to-Agent (A2A) message contracts

Every message that crosses an agent boundary is a **Pydantic model** defined in
`schemas/models.py`. This is the backbone of the design: because the contracts
are explicit and validated, the collaboration is *observable* and a bad payload
cannot silently corrupt a downstream agent.

```
ProfileSchema        name, summary, skills, years_experience, past_titles,
                     education, certifications, languages, flags
JobPosting           id, title, company, description, location, url, source
MatchResult          job, similarity_score(0-1), llm_score(0-100),
                     rationale, combined_score(0-1)
CoverLetterRequest   profile, selected_job
CoverLetterResult    job_id, letter_text
TraceEntry           step, from_agent, to_agent, schema_name, preview
```

Every model uses `extra="forbid"` (reject unknown fields), clamps string/list
lengths, and range-checks numeric scores. So whether a payload comes from an
LLM, an external job API, or a large PDF, it is **normalized at the boundary**.

**The trace.** The orchestrator calls `log_message()` at every hand-off. Each
entry is redacted (long text truncated, lists previewed) and appended to a
`trace` list that travels through the graph state and is rendered in the UI's
"Agent collaboration trace" panel — this is the visible proof of collaboration.

---

## 5. Two-phase orchestration with a human-in-the-loop interrupt

The orchestrator is a **LangGraph `StateGraph`** with a `MemorySaver`
checkpointer. The key feature is a **real interrupt**, not a UI convention:

```
START
  → extract_profile
  → search_jobs
  → evaluate_matches
  → [interrupt_before = "generate_cover_letter"]   ◄── PHASE A stops here
  → generate_cover_letter                          ◄── PHASE B resumes here
  → END
```

- **`run_phase_a(...)`** invokes the graph. It runs extraction → search →
  matching and then **genuinely pauses** before the Cover Letter Agent. The
  graph state (including the ranked list) is persisted under a `thread_id`. No
  cover-letter work happens.
- The ranked list is returned to the UI. The user picks a job.
- **`run_phase_b(thread_id, selected_job_id)`** updates the persisted state
  with the chosen id and **resumes the same thread**, running only the Cover
  Letter Agent.

Because the pause is expressed in the **graph structure**
(`interrupt_before=["generate_cover_letter"]`), the human-in-the-loop step is a
first-class part of the architecture — this is what distinguishes it from plain
agent-to-agent automation.

Error handling is also explicit: conditional edges route to `END` if a node
sets an `error`, so downstream agents never run on incomplete state.

**State object** (`AgentState`): `pdf_bytes`, `resume_filename`,
`desired_title`, `location`, `profile`, `job_pool`, `ranked_results`,
`selected_job_id`, `cover_letter`, `trace`, `error`.

> Privacy note: `MemorySaver` keeps checkpoints in **in-process memory only**.
> An uploaded resume and its extracted profile are never written to disk.

---

## 6. MCP — job search as a tool

Job search is exposed through the **Model Context Protocol** rather than a
direct function call, to demonstrate tool-use over a real protocol boundary.

- **Server** (`mcp_server/job_search_server.py`): a `FastMCP` server exposing
  one tool, `search_jobs(query, location, max_results)`. It runs over **stdio
  only** — spawned as a local subprocess, never listening on a network socket.
- **Client** (`mcp_server/client.py`): the Job Search Agent spawns the server
  as a subprocess and calls the tool via JSON-RPC over stdio. The agent and the
  tool are therefore genuinely decoupled processes.

**Provider strategy (resilience):**

```
search_jobs(query, location)
   │
   ├─ Adzuna (needs API keys)
   │    ├─ try: query + location
   │    ├─ retry: query only        (location may not exist in this country index)
   │    └─ retry: any-word match     (title phrasing may not match posting titles)
   │
   └─ Arbeitnow (no key) — token-overlap scoring, location as a bonus not a filter
```

This progressive relaxation is why a query like *"analyste de données" +
"Casablanca"* returns results instead of an empty list.

---

## 7. Harness, hooks, and subagents

Every agent runs through a **uniform execution harness** (`common/harness.py`),
so cross-cutting behavior lives in exactly one place.

- **Harness** — `run_agent` / `run_agent_sync` wrap an agent call: they time it,
  record metrics, open a tracing span, and fire lifecycle hooks. The
  orchestrator **never calls an agent directly**; it always goes through the
  harness.
- **Hooks** — callbacks registered on `before_agent`, `after_agent`, and
  `on_error`. The default hooks do console logging and metrics recording, but
  new behavior (e.g. a guardrail) can be added without touching any agent code.
  A failing hook can never take down the pipeline.
- **Subagents** — small, single-purpose LLM helpers invoked *inside* a main
  agent, run through the same harness (quietly, so they don't flood the
  console):
  - `QueryNormalizerSubagent` (inside Job Search) — turns a free-form,
    possibly non-English title into standard English search keywords
    (*"développeur web"* → *"web developer"*).
  - `MatchScoringSubagent` (inside Matching) — one LLM-judge call per posting.

```
Orchestrator ──harness──► JobSearchAgent
                              └─harness──► QueryNormalizerSubagent (quiet)
Orchestrator ──harness──► MatchingAgent
                              └─harness──► MatchScoringSubagent × N (quiet)
```

---

## 8. Multimodal resume ingestion

The Profile Extraction Agent accepts **PDF or any common image format**:

```
file bytes + filename
   │
   ├─ .pdf  ─────► pdfplumber (fallback PyPDF2) ─► raw text
   │
   └─ image ─────► Pillow normalize → PNG, downscale
                    └─► Groq VISION model transcribes the image → raw text
   │
   └────────────► Groq LLM structures raw text → ProfileSchema (JSON)
```

Whatever the source, the extracted text goes through the **same LLM structuring
step**, so the rest of the pipeline is unchanged. This is "multimodal" in the
sense that image resumes (photos or scans) are read by a vision model — there is
no OCR library involved.

---

## 9. Matching / evaluation logic

For each posting, the Matching Agent computes **two independent scores**:

| Score | How | Cost |
|-------|-----|------|
| `similarity_score` (0–1) | Cosine similarity between the profile text (incl. the LLM-written `summary`) and the job text, using `all-MiniLM-L6-v2` embeddings run **locally** | free, no key |
| `llm_score` (0–100) + `rationale` | A Groq LLM judges fit and explains why (matches / gaps) | 1 LLM call/job |

```
combined_score = 0.5 · similarity_score + 0.5 · (llm_score / 100)
```

The full list is sorted by `combined_score` and returned — **not** just the top
match, because Phase A must show the whole ranking for the user to choose from.
Using two different scoring methods also enables the evaluation layer to check
whether they agree (§11).

---

## 10. Security (OWASP LLM Top 10)

Four rules from the **OWASP Top 10 for LLM Applications** are implemented and
documented in [`SECURITY.md`](SECURITY.md):

- **LLM01 Prompt Injection** — all third-party text (resume, job descriptions)
  is wrapped by `untrusted_block()` with explicit "data, not instructions"
  delimiters; every system prompt reinforces this; LLM output is re-validated
  against schemas regardless.
- **LLM02 Insecure Output Handling** — Pydantic validation at every boundary;
  job URLs are scheme-checked (only `http(s)`) at the schema *and* the UI;
  external strings are `html.escape()`d before rendering.
- **LLM06 Sensitive Information Disclosure** — trace logs are redacted; API
  keys are env-only and never logged; no on-disk persistence.
- **LLM10 Unbounded Consumption** — caps on file size (10 MB), resume text
  (12k chars), job descriptions (4k chars), result counts, per-call
  `max_tokens`, and HTTP timeouts; token usage is metered.

Plus **MCP isolation** (stdio-only, sanitized inputs, allowlisted hosts) and
**API-key fallback** for availability.

---

## 11. Observability and evaluation

All **console-only** (deliberately not in the UI). Two kinds of metric:
*operational* (cost/speed, recorded live) and *quality* (computed from the
outputs).

**Operational** (`common/metrics.py`) — after each phase, a table prints
per-agent calls/errors/latency and per-model LLM calls/tokens/key-rotations.

**Quality** (`common/quality.py`, pure functions, no LLM) — each targets a
place the pipeline can actually fail:

| Metric | Measures | Why |
|--------|----------|-----|
| **Profile completeness** | fraction of profile fields filled + flag count | catches a weak/partial extraction quantitatively |
| **Score separation** | stdev of `combined_score` + top-minus-median gap | a ranking where every job scores ~80 doesn't discriminate |
| **Spearman rank agreement** | do embedding and LLM scores *order* jobs the same? | ranking is about order — Spearman is correct where Pearson on raw scores misleads |
| **Cover-letter checks** | word count, first-person voice, no 3rd-person self-reference | guards the letter format (incl. the first-person requirement) cheaply |

**LLM-as-judge** (`scripts/evaluate_langsmith.py`, traced by LangSmith) — two
independent judges catch hallucination in the two generated outputs:

- **Rationale groundedness** — is each match rationale grounded in the profile
  *and* the job description (no invented skills / requirements)?
- **Cover-letter faithfulness** — does the letter claim only credentials that
  exist in the profile (no fabricated experience)?

**LangSmith tracing** — when `LANGSMITH_TRACING=true`, the whole LangGraph run
(nodes, harness agent spans, every Groq call, and the judge calls) appears as a
run tree in the LangSmith UI.

---

## 12. Resilience / fallback strategy

| Failure | Fallback |
|---------|----------|
| Groq key rate-limited / invalid | `GROQ_API_KEYS` list → client rotates to the next key |
| Adzuna unconfigured / errored / too-narrow query | progressive relaxation, then Arbeitnow (no key) |
| PDF has no extractable text | error tells the user to upload an image instead (vision path) |
| `pdfplumber` fails | falls back to PyPDF2 |
| A single job posting is malformed | dropped individually, pipeline continues |
| An LLM scoring call fails | that posting gets score 0 + an explanatory rationale, pipeline continues |

---

## 13. Technology stack

| Concern | Choice | Why |
|---------|--------|-----|
| Agent orchestration | **LangGraph** | explicit state graph + native interrupt for human-in-the-loop |
| LLM | **Groq** `llama-3.3-70b-versatile` (+ vision model) | free tier, fast |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` | local, no key, no cost |
| Job search | **Adzuna** (primary) + **Arbeitnow** (no-key fallback) | free tiers |
| Tool protocol | **MCP** (Python SDK, stdio) | real tool-use boundary |
| Contracts | **Pydantic v2** | typed, validated A2A messages |
| PDF | **pdfplumber** / **PyPDF2** | text extraction, no OCR |
| Images | **Pillow** + Groq vision | multimodal ingestion |
| Tracing / eval | **LangSmith** | run-tree observability + LLM-as-judge |
| UI | **Streamlit** (+ CLI fallback) | quick, demoable |

---

## 14. Folder structure

```
schemas/        Pydantic A2A message contracts
common/         config, security, llm_client, embeddings, harness, metrics, logging
mcp_server/     MCP job-search server (Adzuna/Arbeitnow) + stdio client
agents/         one module per agent (single responsibility, typed I/O)
orchestrator/   LangGraph state graph + Phase A/B interrupt
ui/             Streamlit app + CLI fallback
scripts/        standalone smoke tests + LangSmith evaluation
sample_data/    synthetic sample-resume generator
SECURITY.md     OWASP LLM Top 10 mapping
ARCHITECTURE.md this document
README.md       setup + run instructions
```

---

## 15. Out of scope (possible extensions)

- **Feedback / RLHF loop** — logging which cover letters the user edits or
  accepts, and using that signal to tune the Matching Agent's scoring or the
  Cover Letter Agent's tone. Deliberately excluded from this POC.
- **OCR library** — not needed; image resumes go through the vision model.
- **Persistent storage / auth / multi-user** — the checkpointer is in-memory by
  design for privacy and simplicity.
```
