"""
Orchestrator: a LangGraph StateGraph wiring the four agents together in two
explicit phases, with a real interrupt point between them.

Phase A: extract_profile -> search_jobs -> evaluate_matches -> [INTERRUPT]
Phase B: (resumed with selected_job_id) -> generate_cover_letter -> END

The interrupt is implemented with `interrupt_before=["generate_cover_letter"]`
on the compiled graph plus a MemorySaver checkpointer keyed by thread_id: the
graph genuinely pauses (state is persisted, no cover-letter work happens)
until `run_phase_b` is called with a human-selected job_id. This is what
makes the human-in-the-loop step visible in the graph structure rather than
being an implicit UI-only pause.

Note on state: MemorySaver keeps checkpoints in-process memory only (never
written to disk), so an uploaded resume's bytes and the extracted profile
never persist beyond the running process -- a deliberate simplicity/privacy
choice for this POC.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.cover_letter import generate_cover_letter
from agents.job_search import search_jobs
from agents.matching import evaluate_matches
from agents.profile_extraction import extract_profile
from common import metrics
from common.harness import run_agent
from common.logging_utils import log_message
from schemas.models import CoverLetterRequest, JobPosting, ProfileSchema


class AgentState(TypedDict, total=False):
    pdf_bytes: bytes          # raw resume file bytes (PDF or image)
    resume_filename: str      # original filename, used to route multimodal ingestion
    desired_title: str
    location: str
    profile: Optional[dict]
    job_pool: list[dict]
    ranked_results: list[dict]
    selected_job_id: Optional[str]
    cover_letter: Optional[dict]
    trace: list[dict]
    error: Optional[str]


async def _extract_profile_async(pdf_bytes: bytes, filename: str):
    return await asyncio.to_thread(extract_profile, pdf_bytes, filename)


async def _node_extract_profile(state: AgentState) -> dict:
    trace = state.get("trace", [])
    try:
        profile = await run_agent(
            "ProfileExtractionAgent",
            _extract_profile_async,
            state["pdf_bytes"],
            state.get("resume_filename", "resume.pdf"),
        )
    except Exception as e:
        log_message(trace, "Orchestrator", "ProfileExtractionAgent", "error", str(e))
        return {"trace": trace, "error": f"Profile extraction failed: {e}"}
    log_message(trace, "ProfileExtractionAgent", "Orchestrator", "ProfileSchema", profile)
    return {"profile": profile.model_dump(), "trace": trace}


async def _node_search_jobs(state: AgentState) -> dict:
    trace = state.get("trace", [])
    profile = ProfileSchema(**state["profile"])
    try:
        jobs = await run_agent(
            "JobSearchAgent", search_jobs, profile, state["desired_title"], state.get("location", "")
        )
    except Exception as e:
        log_message(trace, "Orchestrator", "JobSearchAgent", "error", str(e))
        return {"trace": trace, "error": f"Job search failed: {e}"}
    log_message(trace, "JobSearchAgent", "Orchestrator", "list[JobPosting]", f"{len(jobs)} postings")
    return {"job_pool": [j.model_dump() for j in jobs], "trace": trace}


async def _node_evaluate_matches(state: AgentState) -> dict:
    trace = state.get("trace", [])
    profile = ProfileSchema(**state["profile"])
    jobs = [JobPosting(**j) for j in state.get("job_pool", [])]
    if not jobs:
        return {"ranked_results": [], "trace": trace, "error": "No job postings were found for this title/location."}
    try:
        results = await run_agent(
            "MatchingAgent", asyncio.to_thread, evaluate_matches, profile, jobs
        )
    except Exception as e:
        log_message(trace, "Orchestrator", "MatchingAgent", "error", str(e))
        return {"trace": trace, "error": f"Matching failed: {e}"}
    log_message(trace, "MatchingAgent", "Orchestrator", "list[MatchResult]", f"{len(results)} ranked results")
    return {"ranked_results": [r.model_dump() for r in results], "trace": trace}


async def _node_generate_cover_letter(state: AgentState) -> dict:
    trace = state.get("trace", [])
    profile = ProfileSchema(**state["profile"])
    selected_id = state.get("selected_job_id")
    match = next((r for r in state.get("ranked_results", []) if r["job"]["id"] == selected_id), None)
    if match is None:
        return {"trace": trace, "error": f"selected_job_id '{selected_id}' not found in ranked results."}

    selected_job = JobPosting(**match["job"])
    request = CoverLetterRequest(profile=profile, selected_job=selected_job)
    log_message(trace, "Orchestrator", "CoverLetterAgent", "CoverLetterRequest", request)

    try:
        result = await run_agent(
            "CoverLetterAgent", asyncio.to_thread, generate_cover_letter, request
        )
    except Exception as e:
        log_message(trace, "Orchestrator", "CoverLetterAgent", "error", str(e))
        return {"trace": trace, "error": f"Cover letter generation failed: {e}"}

    log_message(trace, "CoverLetterAgent", "Orchestrator", "CoverLetterResult", result)
    return {"cover_letter": result.model_dump(), "trace": trace}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_profile", _node_extract_profile)
    graph.add_node("search_jobs", _node_search_jobs)
    graph.add_node("evaluate_matches", _node_evaluate_matches)
    graph.add_node("generate_cover_letter", _node_generate_cover_letter)

    graph.add_edge(START, "extract_profile")
    # Short-circuit to END on error instead of running downstream agents on
    # incomplete state -- explicit routing, not a silent no-op.
    graph.add_conditional_edges(
        "extract_profile", lambda s: END if s.get("error") else "search_jobs"
    )
    graph.add_conditional_edges(
        "search_jobs", lambda s: END if s.get("error") else "evaluate_matches"
    )
    graph.add_edge("evaluate_matches", "generate_cover_letter")
    graph.add_edge("generate_cover_letter", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["generate_cover_letter"])


_GRAPH = build_graph()


def new_thread_id() -> str:
    return str(uuid.uuid4())


async def run_phase_a(
    pdf_bytes: bytes,
    desired_title: str,
    location: str,
    thread_id: str,
    resume_filename: str = "resume.pdf",
) -> AgentState:
    """Extraction -> Search -> Matching. Stops at the interrupt point."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "pdf_bytes": pdf_bytes,
        "resume_filename": resume_filename,
        "desired_title": desired_title,
        "location": location,
        "trace": [],
    }
    result = await _GRAPH.ainvoke(initial_state, config)
    metrics.registry.print_summary("PHASE A METRICS (console only)")
    return result


async def run_phase_b(thread_id: str, selected_job_id: str) -> AgentState:
    """Resumes the same thread with a human-selected job_id and runs only
    the Cover Letter Agent."""
    config = {"configurable": {"thread_id": thread_id}}
    await _GRAPH.aupdate_state(config, {"selected_job_id": selected_job_id})
    result = await _GRAPH.ainvoke(None, config)
    metrics.registry.print_summary("PHASE A+B METRICS (console only)")
    return result
