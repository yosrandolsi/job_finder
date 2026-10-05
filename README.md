# Autonomous Job Application Agent (Multi-Agent POC)

A proof-of-concept multi-agent system, built for a Generative AI course
assignment, that turns a PDF resume + a desired job title into a ranked list
of matching jobs and, once a human picks one, a tailored cover letter.

It demonstrates:
- **Agent collaboration** through explicit Pydantic message contracts (A2A),
  with every inter-agent message logged for a visible trace.
- **MCP** (Model Context Protocol) as the transport for the job-search tool.
- **Human-in-the-loop orchestration** via a real LangGraph interrupt between
  "show me ranked matches" and "write the cover letter for this one."
- **Harness / hooks / subagents**: every agent runs through a uniform
  execution harness (`common/harness.py`) with lifecycle hooks
  (before/after/error) doing logging + metrics; small single-purpose
  subagents (query normalizer, per-job match scorer) run through the same
  harness.
- **Multimodal resume ingestion**: PDFs are parsed as text; images in any
  common format (photo or scan of a resume) are read by the Groq vision
  model.
- **Observability & evaluation**: per-agent latency + token metrics printed
  to the console after each phase, LangSmith tracing of the whole graph and
  every LLM call, and an offline LLM-as-judge evaluation script.
- **Security**: 4 rules from the OWASP Top 10 for LLM Applications,
  documented in [SECURITY.md](SECURITY.md).

> **For a full design walkthrough** (agents, A2A contracts, the interrupt,
> MCP, harness/hooks/subagents, security, and the metrics rationale), see
> [ARCHITECTURE.md](ARCHITECTURE.md).

## Architecture

```
                 ┌────────────────────────┐
 PDF / image  →  │ Profile Extraction      │ → ProfileSchema
                 │ Agent (PDF text OR       │   (multimodal ingestion)
                 │ Groq vision, then Groq)  │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐        ┌─────────────────────┐
                 │ Job Search Agent         │◄──MCP─►│ job-search MCP      │
                 │                          │  stdio │ server (Adzuna /    │
                 └────────────┬────────────┘        │ Arbeitnow fallback) │
                              │ list[JobPosting]      └─────────────────────┘
                 ┌────────────▼────────────┐
                 │ Matching/Evaluation      │  embeddings (local) +
                 │ Agent                    │  LLM-judged score
                 └────────────┬────────────┘
                              │ list[MatchResult], ranked
                    ══════════▼══════════   PHASE A ends here
                    ║   LangGraph        ║   (interrupt_before)
                    ║   INTERRUPT        ║   → shown to the user
                    ══════════▲══════════   PHASE B resumes with
                              │ selected_job_id     selected_job_id
                 ┌────────────┴────────────┐
                 │ Cover Letter Agent       │ → CoverLetterResult
                 └──────────────────────────┘
```

The orchestrator (`orchestrator/graph.py`) is a single LangGraph
`StateGraph` with a `MemorySaver` checkpointer and
`interrupt_before=["generate_cover_letter"]`. `run_phase_a()` invokes the
graph and it genuinely stops before the Cover Letter Agent runs -- no
cover-letter work happens until `run_phase_b()` resumes the same
`thread_id` with a `selected_job_id`. This is a graph-level pause, not just
a UI convention.

## Folder structure

```
schemas/         Pydantic A2A message contracts (ProfileSchema, JobPosting, MatchResult, ...)
common/          Shared infrastructure:
                   config.py        .env loading
                   security.py      prompt-injection guard, input sanitization, log redaction
                   llm_client.py    Groq wrapper (multi-key fallback, JSON/text/vision, token metering)
                   embeddings.py    local sentence-transformers cosine similarity
                   harness.py       uniform agent execution wrapper + lifecycle hooks
                   metrics.py       operational per-agent / per-LLM metrics (console)
                   quality.py       quality metrics (completeness, ranking separation, letter checks)
                   logging_utils.py redacted A2A message trace
mcp_server/      MCP server exposing search_jobs (Adzuna primary, Arbeitnow fallback) + its stdio client
agents/          One module per agent, each with a single typed input -> typed output function
orchestrator/    LangGraph StateGraph wiring the agents, with the Phase A/B interrupt
ui/              Streamlit app (app.py) + CLI fallback (cli.py)
sample_data/     Synthetic sample-resume generator
scripts/         Standalone smoke tests + the LangSmith evaluation script
ARCHITECTURE.md  Full design walkthrough
SECURITY.md      OWASP LLM Top 10 mapping
```

## Setup

1. **Python 3.11+** and a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt   # optional, only for generating a sample resume
   ```

2. **Get free API keys:**
   - **Groq** (LLM inference, required): sign up at
     [console.groq.com](https://console.groq.com/keys), create an API key.
     The free tier has a **daily token budget per account** (see
     Troubleshooting) — fine for a demo, but heavy repeated testing can
     exhaust it.
   - **Adzuna** (job search, optional): sign up at
     [developer.adzuna.com](https://developer.adzuna.com/), create an app to
     get an `app_id` and `app_key`. If you skip this, the Job Search Agent
     automatically falls back to the no-key **Arbeitnow** API.

3. **Configure environment:**
   ```bash
   copy .env.example .env
   ```
   Fill in `GROQ_API_KEY` (required) and `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`
   (optional). For rate-limit resilience you can instead set
   `GROQ_API_KEYS=key1,key2,...` (keys from **different** Groq accounts) and
   the client rotates to the next key when one is exhausted.

4. **(Optional) generate a sample resume** to demo without a real PDF:
   ```bash
   python sample_data/generate_sample_resume.py
   ```

## Running the demo

**Streamlit UI (primary):**
```bash
streamlit run ui/app.py
```
In the sidebar: upload a resume (**PDF or image** — png/jpg/webp/etc.), enter
a desired title, click **Find matching jobs**. Review the extracted profile,
the ranked job cards, and the agent trace; pick a job from the dropdown, then
**Generate letter**. (Per-agent latency + token metrics print to the
**console**, not the UI.)

> After editing any `.env` value or any file outside `ui/app.py`, **restart
> Streamlit** — it loads the environment and non-UI modules only once at
> startup (see Troubleshooting).

**CLI fallback:**
```bash
python -m ui.cli --resume sample_data/sample_resume.pdf --title "Data Analyst"
```
Accepts an image path too (e.g. `--resume path/to/cv.png`).

**Standalone component tests** (useful while developing / for the report):
```bash
python scripts/test_job_search.py "data analyst" "Paris"
python scripts/test_profile_extraction.py
```

**Performance evaluation (console only, not in the UI):**
```bash
python scripts/evaluate_langsmith.py sample_data/sample_resume.pdf "Data Analyst"
```
Runs the full pipeline and prints, in order:
1. **Operational metrics** — per-agent latency + per-model LLM token usage.
2. **Profile extraction quality** — field completeness + flags raised.
3. **Ranking quality** — score separation and the **embedding-vs-LLM rank
   agreement** (Spearman); flags a ranking that barely discriminates.
4. **LLM-as-judge** — *rationale groundedness* (are match rationales grounded,
   not hallucinated?) and *cover-letter faithfulness* (does the generated
   letter invent no credentials?), plus first-person/length checks on the
   letter.

When LangSmith tracing is enabled, the whole evaluation (pipeline + every
judge call) also appears as a run tree in the LangSmith UI.

> Each run makes ~26 LLM calls. If you're near the Groq daily token budget,
> run it sparingly (see Troubleshooting).

**LangSmith tracing (optional):** set `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` in `.env` (free account at
[smith.langchain.com](https://smith.langchain.com)) to see the full
LangGraph run tree -- graph nodes, harness agent spans, and every Groq call
-- in the LangSmith UI.

**API-key fallback:** set `GROQ_API_KEYS=key1,key2,...` instead of a single
`GROQ_API_KEY` and the LLM client automatically rotates to the next key when
one hits its rate limit.

## Data schemas

See `schemas/models.py` for the full definitions:
`ProfileSchema`, `JobPosting`, `MatchResult`, `CoverLetterRequest`,
`CoverLetterResult`, `TraceEntry`. Every agent's input/output is one of
these; all models use `extra="forbid"` and clamp string/list lengths on the
way in.

## Security notes

This is a POC, but the following are deliberately in place rather than
deferred, since the assignment calls for it:

- **Prompt-injection guarding**: resume text and job descriptions are
  untrusted third-party content. Anywhere they're embedded in an LLM prompt,
  they're wrapped with `common.security.untrusted_block()`, which delimits
  the content and explicitly instructs the model not to follow instructions
  found inside it.
- **Schema validation at every A2A boundary**: all inter-agent messages are
  Pydantic models with `extra="forbid"` and length-capped fields, so a
  malformed or oversized payload (from an LLM, an external job API, or a
  large PDF) can't silently propagate or blow up downstream cost.
- **MCP server isolation**: `mcp_server/job_search_server.py` runs over
  stdio only, spawned as a local subprocess -- it never listens on a network
  socket, so it has no remote attack surface in this POC. All its logging
  goes to stderr (never stdout, which is the JSON-RPC channel).
- **No URL injection**: search query/location are cleaned with
  `sanitize_single_line()` and always passed to `requests` via `params=`
  (never string-concatenated into a URL). The one value that does go into a
  URL path (Adzuna's country code) comes from an env var, not user input,
  and is still checked against an allowlist.
- **Secrets handling**: `GROQ_API_KEY` / `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`
  are read from environment variables only, never logged, never included in
  any schema or trace output. `.env` is gitignored.
- **Bounded cost/size**: PDF uploads are capped at 10MB, resume text sent to
  the LLM is capped (~12k chars), job descriptions are capped (~4k chars),
  and the number of postings evaluated per search is capped (15).
- **No persistence**: the LangGraph checkpointer (`MemorySaver`) keeps state
  in-process memory only -- an uploaded resume and its extracted profile are
  never written to disk.

## Troubleshooting

**A code/`.env` change didn't take effect (e.g. new key, but same error).**
Streamlit reloads only `ui/app.py` on save; it does **not** re-read `.env` or
re-import other modules (`orchestrator/`, `agents/`, `common/`). Restart the
server: `Ctrl+C`, then `streamlit run ui/app.py`. If a server was left running
in the background, kill leftovers first:
```powershell
Get-Process | Where-Object { $_.Name -like '*streamlit*' } | Stop-Process -Force
```
(Multiple stale servers is also why the app can end up on a different port
like 8502/8503 while your browser talks to an old one.)

**`429 ... tokens per day (TPD): Limit 100000` from Groq.**
This is a **daily token budget tied to your Groq _organization_, not the
individual key.** A new key from the *same account* shares the same exhausted
budget (the error's `org_...` id stays the same). Options:
- wait for the daily reset, or
- put a key from a **genuinely different Groq account** in
  `GROQ_API_KEYS=key1,key2` so the client falls back to fresh quota, or
- reduce spend: lower `MAX_JOBS` in `agents/job_search.py` (each posting costs
  one LLM scoring call) and avoid re-running the eval script repeatedly.

**"No jobs found."** Usually a location the configured Adzuna country doesn't
index. Set `ADZUNA_COUNTRY` in `.env` to a valid 2-letter code (`us`, `gb`,
`fr`, `de`, ...); the agent also relaxes the query and falls back to Arbeitnow
automatically.

**"No extractable text found in the PDF."** The PDF is a scan/image with no
text layer. Upload it as an **image** instead — images go through the vision
model.

## Limitations / out of scope

- No OCR library: text-based PDFs are parsed locally; scanned/photo resumes
  must be uploaded as images, which are read by the vision model instead.
- **Non-English input** is largely handled — resume fields are normalized to
  English during extraction, and non-English job titles are translated before
  search — but embedding similarity uses an English-centric model
  (`all-MiniLM-L6-v2`), so cross-lingual similarity (English profile vs. a
  French job description) is weaker; the LLM-judge score compensates. Cover
  letters are written in English.
- No feedback/RLHF loop (e.g. learning from which cover letters the user
  liked) -- explicitly out of scope for this POC. A natural next step would
  be logging user selections/edits and using them to fine-tune the Matching
  Agent's scoring or the Cover Letter Agent's tone.
- Groq free tier has a **daily token budget per account** (see
  Troubleshooting); heavy repeated runs can exhaust it.
- Adzuna/Arbeitnow coverage varies by country/region; results depend on
  what those free APIs index.
