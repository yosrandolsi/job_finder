"""
Streamlit presentation layer -- modern light design.

Phase A: upload resume + desired title -> ranked job match cards.
Human-in-the-loop: user picks a job.
Phase B: tailored cover letter for that job only.

All externally-sourced strings (job titles, companies, rationales, ...) are
HTML-escaped before being rendered inside styled markdown blocks, since job
APIs and LLM output are untrusted content.

Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import asyncio
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import common.config  # noqa: E402,F401  (loads .env)
import streamlit as st  # noqa: E402

from orchestrator.graph import new_thread_id, run_phase_a, run_phase_b  # noqa: E402

st.set_page_config(page_title="Job Match AI", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #A855F7 100%);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; font-size: 1.9rem; margin: 0 0 .4rem 0; }
    .hero p { color: #E0E7FF; margin: 0; font-size: 1rem; }

    .step-pill {
        display: inline-block;
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px;
        padding: .25rem .8rem;
        margin-right: .45rem;
        margin-top: .9rem;
        font-size: .8rem;
        color: white;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: .8rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, .05);
    }
    .card h4 { margin: 0 0 .15rem 0; color: #0F172A; font-size: 1.05rem; }
    .card .meta { color: #64748B; font-size: .85rem; margin-bottom: .55rem; }
    .card .rationale { color: #334155; font-size: .9rem; line-height: 1.45; }

    .rank-badge {
        display: inline-flex; align-items: center; justify-content: center;
        background: #EEF2FF; color: #4F46E5;
        border-radius: 8px; font-weight: 700; font-size: .8rem;
        padding: .15rem .55rem; margin-right: .5rem;
    }
    .score-pill {
        display: inline-block; border-radius: 999px;
        padding: .18rem .7rem; font-size: .78rem; font-weight: 600;
        margin-right: .4rem; margin-bottom: .5rem;
    }
    .score-high { background: #DCFCE7; color: #15803D; }
    .score-mid  { background: #FEF9C3; color: #A16207; }
    .score-low  { background: #FEE2E2; color: #B91C1C; }
    .source-pill { background: #F1F5F9; color: #475569; }

    .skill-chip {
        display: inline-block; background: #EEF2FF; color: #4338CA;
        border-radius: 999px; padding: .18rem .7rem; font-size: .78rem;
        margin: 0 .3rem .35rem 0;
    }

    .letter-box {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #4F46E5;
        border-radius: 12px; padding: 1.4rem 1.6rem;
        color: #1E293B; font-size: .95rem; line-height: 1.65;
        white-space: pre-wrap;
    }

    div[data-testid="stSidebarContent"] { background: #FFFFFF; }
    .stButton > button[kind="primary"] {
        border-radius: 10px; font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
if "phase_a_result" not in st.session_state:
    st.session_state.phase_a_result = None
if "cover_letter" not in st.session_state:
    st.session_state.cover_letter = None


def _score_class(score: float) -> str:
    if score >= 0.6:
        return "score-high"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <h1>🎯 Job Match AI</h1>
    <p>Upload your resume, pick a target role — AI agents extract your profile,
    find matching jobs, rank them, and write a tailored cover letter for the one <b>you</b> choose.</p>
    <div>
        <span class="step-pill">1 · Profile extraction</span>
        <span class="step-pill">2 · Job search (MCP)</span>
        <span class="step-pill">3 · Match &amp; rank</span>
        <span class="step-pill">4 · You choose</span>
        <span class="step-pill">5 · Cover letter</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar -- inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Your application")
    resume_file = st.file_uploader(
        "Resume (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"],
        help="PDF resumes are parsed as text; images (including photos/scans) are read by the vision model.",
    )
    desired_title = st.text_input("Desired job title", placeholder="e.g. Data Analyst, développeur web…")
    location = st.text_input("Location (optional)", placeholder="e.g. Paris, Remote…")

    run_clicked = st.button(
        "🚀  Find matching jobs",
        type="primary",
        use_container_width=True,
        disabled=not resume_file or not desired_title,
    )
    if not resume_file or not desired_title:
        st.caption("Upload a resume and enter a title to start.")

    st.divider()
    if st.button("↺  New session", use_container_width=True):
        st.session_state.thread_id = new_thread_id()
        st.session_state.phase_a_result = None
        st.session_state.cover_letter = None
        st.rerun()

# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------
if run_clicked and resume_file and desired_title:
    with st.status("Running the agent pipeline…", expanded=True) as status:
        st.write("🔍 Extracting your profile from the resume…")
        pdf_bytes = resume_file.read()
        try:
            result = asyncio.run(
                run_phase_a(
                    pdf_bytes,
                    desired_title,
                    location,
                    st.session_state.thread_id,
                    resume_filename=resume_file.name,
                )
            )
            st.session_state.phase_a_result = result
            st.session_state.cover_letter = None
            if result.get("error"):
                status.update(label="Pipeline stopped with an error", state="error")
            else:
                n = len(result.get("ranked_results") or [])
                status.update(label=f"Done — {n} jobs found and ranked ✅", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Phase A failed: {e}")

result = st.session_state.phase_a_result

if not result:
    st.info("👈 Start in the sidebar: upload your resume and enter the job title you're aiming for.")
else:
    if result.get("error"):
        st.error(result["error"])

    # ------------------------------ Profile ------------------------------
    profile = result.get("profile")
    if profile:
        st.markdown("#### 👤 Extracted profile")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Name", profile.get("name") or "Unknown")
        years = profile.get("years_experience")
        c2.metric("Experience", f"{years} yrs" if years is not None else "—")
        c3.metric("Skills", len(profile.get("skills", [])))
        c4.metric("Past roles", len(profile.get("past_titles", [])))

        if profile.get("summary"):
            st.markdown(
                f'<div class="card"><div class="rationale">{html.escape(profile["summary"])}</div></div>',
                unsafe_allow_html=True,
            )
        if profile.get("skills"):
            chips = "".join(f'<span class="skill-chip">{html.escape(s)}</span>' for s in profile["skills"][:25])
            st.markdown(chips, unsafe_allow_html=True)

        if profile.get("flags"):
            with st.expander(f"⚠️ Extraction notes ({len(profile['flags'])})"):
                for f in profile["flags"]:
                    st.markdown(f"- {f}")
        with st.expander("Full profile JSON"):
            st.json(profile)

    # ------------------------------ Ranked jobs ------------------------------
    ranked = result.get("ranked_results") or []
    if ranked:
        st.markdown(f"#### 💼 Ranked job matches · {len(ranked)}")

        for i, r in enumerate(ranked, start=1):
            job = r["job"]
            combined = r["combined_score"]
            title = html.escape(job["title"])
            company = html.escape(job["company"])
            loc_txt = html.escape(job["location"])
            source = html.escape(job.get("source", ""))
            rationale = html.escape(r["rationale"])
            url = job["url"]
            # OWASP LLM02 (insecure output handling): only render http(s)
            # links -- a malicious posting could otherwise inject e.g. a
            # javascript: URL through the job API.
            safe_url = url if url.startswith(("https://", "http://")) else ""
            link = f' · <a href="{html.escape(safe_url)}" target="_blank">View posting ↗</a>' if safe_url else ""

            st.markdown(
                f"""
<div class="card">
    <h4><span class="rank-badge">#{i}</span>{title}</h4>
    <div class="meta">🏢 {company} &nbsp;·&nbsp; 📍 {loc_txt}{link}</div>
    <div>
        <span class="score-pill {_score_class(combined)}">Match {round(combined * 100)}%</span>
        <span class="score-pill score-pill source-pill">AI score {r["llm_score"]}/100</span>
        <span class="score-pill score-pill source-pill">Similarity {round(r["similarity_score"] * 100)}%</span>
        <span class="score-pill score-pill source-pill">{source}</span>
    </div>
    <div class="rationale">{rationale}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        # --------------------- Selection + Phase B ---------------------
        st.markdown("#### ✍️ Get your tailored cover letter")
        options = {
            f'#{i} — {r["job"]["title"]} @ {r["job"]["company"]}': r["job"]["id"]
            for i, r in enumerate(ranked, start=1)
        }
        sel_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
        with sel_col:
            choice_label = st.selectbox("Choose the job you want to apply to", list(options.keys()))
        with btn_col:
            generate_clicked = st.button("Generate letter", type="primary", use_container_width=True)

        if generate_clicked:
            selected_id = options[choice_label]
            with st.spinner("The Cover Letter Agent is writing…"):
                try:
                    phase_b_result = asyncio.run(run_phase_b(st.session_state.thread_id, selected_id))
                    if phase_b_result.get("error"):
                        st.error(phase_b_result["error"])
                    else:
                        st.session_state.cover_letter = phase_b_result.get("cover_letter")
                        st.session_state.phase_a_result["trace"] = phase_b_result.get("trace", result.get("trace"))
                except Exception as e:
                    st.error(f"Cover letter generation failed: {e}")

    if st.session_state.cover_letter:
        st.markdown(
            f'<div class="letter-box">{html.escape(st.session_state.cover_letter["letter_text"])}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.download_button(
            "⬇️  Download as .txt",
            st.session_state.cover_letter["letter_text"],
            file_name="cover_letter.txt",
        )

    # ------------------------------ Trace ------------------------------
    with st.expander("🔬 Agent collaboration trace (inter-agent messages)"):
        for entry in result.get("trace", []):
            st.markdown(f"**#{entry['step']} {entry['from_agent']} → {entry['to_agent']}** · `{entry['schema']}`")
            preview = entry["preview"]
            # st.json() only accepts JSON-shaped data; plain-string previews
            # (e.g. "15 postings") must be rendered as text or they raise a
            # client-side JSON parse error.
            if isinstance(preview, (dict, list)):
                st.json(preview)
            else:
                st.code(str(preview), language=None)
