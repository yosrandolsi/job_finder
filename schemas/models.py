"""
Pydantic message contracts exchanged between agents.

Every agent's input/output conforms to exactly one of these models. This is
the A2A (agent-to-agent) security boundary for this POC: each model uses
`extra="forbid"` (reject unexpected fields) and clamps/caps string and list
lengths on the way in, so a malformed or oversized payload from an LLM, an
external job API, or a large PDF cannot silently balloon downstream cost or
break a receiving agent's assumptions.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SHORT = 200
MAX_TEXT = 6000
MAX_LIST_ITEMS = 50


def _clamp(value: Optional[str], max_len: int) -> Optional[str]:
    if value is None:
        return value
    return str(value)[:max_len]


class ProfileSchema(BaseModel):
    """Structured candidate profile produced by the Profile Extraction Agent."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    summary: Optional[str] = None  # 2-3 sentence professional summary, used for embedding similarity
    skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    past_titles: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)  # missing/ambiguous fields noted during extraction

    @field_validator("name", mode="before")
    @classmethod
    def _cap_name(cls, v):
        return _clamp(v, MAX_SHORT)

    @field_validator("summary", mode="before")
    @classmethod
    def _cap_summary(cls, v):
        return _clamp(v, 800)

    @field_validator("skills", "past_titles", "education", "certifications", "languages", "flags", mode="before")
    @classmethod
    def _cap_list(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        return [str(item)[:MAX_SHORT] for item in v][:MAX_LIST_ITEMS]


class JobPosting(BaseModel):
    """A single job posting, normalized from whichever provider returned it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    company: str
    description: str
    location: str
    url: str
    source: str = "unknown"  # "adzuna" | "arbeitnow"

    @field_validator("id", "title", "company", "location", "source", mode="before")
    @classmethod
    def _cap_short(cls, v):
        return _clamp("" if v is None else str(v), MAX_SHORT)

    @field_validator("description", mode="before")
    @classmethod
    def _cap_desc(cls, v):
        return _clamp("" if v is None else str(v), MAX_TEXT)

    @field_validator("url", mode="before")
    @classmethod
    def _cap_url(cls, v):
        # OWASP LLM02 (insecure output handling): job APIs are untrusted --
        # drop any non-http(s) URL (javascript:, data:, file:, ...) at the
        # schema boundary so it can never reach the UI.
        url = _clamp("" if v is None else str(v), 500)
        return url if url.startswith(("https://", "http://")) else ""


class MatchResult(BaseModel):
    """Ranking output from the Matching/Evaluation Agent for one posting."""

    model_config = ConfigDict(extra="forbid")

    job: JobPosting
    similarity_score: float = Field(ge=0.0, le=1.0)   # embedding cosine similarity
    llm_score: int = Field(ge=0, le=100)               # LLM-judged fit score
    rationale: str
    combined_score: float = Field(ge=0.0, le=1.0)      # blend used for ranking

    @field_validator("rationale", mode="before")
    @classmethod
    def _cap_rationale(cls, v):
        return _clamp("" if v is None else str(v), 600)


class CoverLetterRequest(BaseModel):
    """Input to the Cover Letter Agent -- only built after the user selects a job."""

    model_config = ConfigDict(extra="forbid")

    profile: ProfileSchema
    selected_job: JobPosting


class CoverLetterResult(BaseModel):
    """Output of the Cover Letter Agent."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    letter_text: str


class TraceEntry(BaseModel):
    """One logged inter-agent message, for the visible collaboration trace."""

    model_config = ConfigDict(extra="forbid")

    step: int
    from_agent: str
    to_agent: str
    schema_name: str
    preview: dict | list | str | int | float | bool | None = None
