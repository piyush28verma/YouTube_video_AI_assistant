import os
import time
import streamlit as st
from dotenv import load_dotenv

from transcript_loader import (
    fetch_transcript,
    parse_custom_transcript,
    SAMPLE_VIDEOS,
    extract_video_id,
)
from rag_pipeline import RAGPipeline
from config import SUPPORTED_CHAT_MODELS

# ── Load environment variables ──
load_dotenv(override=True)

# ── Page Configuration ──
st.set_page_config(
    page_title="YouTube Video Assistant — RAG-Powered Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM ULTRA HI-FI CSS — Dark Glassmorphic Theme with Animations
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-deep: #050816;
        --bg-card: rgba(15, 23, 42, 0.65);
        --bg-card-hover: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.06);
        --glass-border-hover: rgba(129, 140, 248, 0.35);
        --accent-indigo: #818cf8;
        --accent-violet: #a78bfa;
        --accent-cyan: #22d3ee;
        --accent-emerald: #34d399;
        --accent-amber: #fbbf24;
        --accent-rose: #fb7185;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* ── Global Overrides ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(180deg, #050816 0%, #0a0f1e 40%, #0d1321 100%) !important;
    }

    /* Hide default Streamlit cruft */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Animated Hero Title ── */
    .hero-container {
        text-align: left;
        padding: 0 0 8px 0;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 30%, #f472b6 60%, #38bdf8 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.02rem;
        font-weight: 400;
        margin-top: 4px;
        line-height: 1.5;
    }
    .hero-subtitle strong {
        color: var(--accent-indigo);
        font-weight: 600;
    }

    /* ── Tech Badge Chips ── */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 20px 0;
    }
    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.10);
        color: var(--accent-indigo);
        border: 1px solid rgba(99, 102, 241, 0.25);
        backdrop-filter: blur(6px);
        transition: all 0.25s ease;
    }
    .badge-chip:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.10);
        color: var(--accent-emerald);
        border-color: rgba(16, 185, 129, 0.25);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.10);
        color: var(--accent-amber);
        border-color: rgba(245, 158, 11, 0.25);
    }
    .badge-rose {
        background: rgba(251, 113, 133, 0.10);
        color: var(--accent-rose);
        border-color: rgba(251, 113, 133, 0.25);
    }
    .badge-cyan {
        background: rgba(34, 211, 238, 0.10);
        color: var(--accent-cyan);
        border-color: rgba(34, 211, 238, 0.25);
    }

    /* ── Pulse Indicator ── */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }
    .pulse-dot-green { background: var(--accent-emerald); box-shadow: 0 0 8px var(--accent-emerald); }
    .pulse-dot-amber { background: var(--accent-amber); box-shadow: 0 0 8px var(--accent-amber); }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }

    /* ── Glassmorphic Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px -4px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-indigo), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--glass-border-hover);
        box-shadow: 0 12px 40px -4px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .metric-card:hover::before { opacity: 1; }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.15;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-val-sm {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-lbl {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }
    .metric-icon {
        font-size: 1.3rem;
        margin-bottom: 6px;
        display: block;
    }

    /* ── Evidence Grounding Cards ── */
    .evidence-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.8) 0%, rgba(30,41,59,0.5) 100%);
        border-left: 3px solid var(--accent-indigo);
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        border-top: 1px solid var(--glass-border);
        border-right: 1px solid var(--glass-border);
        border-bottom: 1px solid var(--glass-border);
        transition: border-left-color 0.3s ease, box-shadow 0.3s ease;
    }
    .evidence-card:hover {
        border-left-color: var(--accent-cyan);
        box-shadow: 0 4px 20px -4px rgba(34,211,238,0.1);
    }
    .evidence-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .evidence-ts {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--accent-cyan);
    }
    .evidence-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
    }
    .score-high   { background: rgba(16,185,129,0.15); color: var(--accent-emerald); border: 1px solid rgba(16,185,129,0.3); }
    .score-medium { background: rgba(99,102,241,0.15); color: var(--accent-indigo); border: 1px solid rgba(99,102,241,0.3); }
    .score-low    { background: rgba(245,158,11,0.15); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.3); }
    .evidence-body {
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--text-secondary);
        border-left: 2px solid rgba(255,255,255,0.04);
        padding-left: 12px;
    }

    /* ── Chapter Timeline ── */
    .chapter-card {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 14px 18px;
        background: var(--bg-card);
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid var(--glass-border);
        transition: all 0.25s ease;
    }
    .chapter-card:hover {
        border-color: var(--glass-border-hover);
        background: var(--bg-card-hover);
        transform: translateX(4px);
    }
    .chapter-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
        color: white;
        flex-shrink: 0;
    }
    .chapter-ts {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-cyan);
        font-weight: 600;
        font-size: 0.85rem;
        min-width: 60px;
    }
    .chapter-title {
        font-weight: 500;
        color: var(--text-primary);
        font-size: 0.92rem;
        flex: 1;
    }
    .chapter-link {
        color: var(--accent-indigo);
        font-weight: 600;
        text-decoration: none;
        font-size: 0.8rem;
        white-space: nowrap;
        transition: color 0.2s ease;
    }
    .chapter-link:hover { color: var(--accent-cyan); }

    /* ── Transcript Segment Row ── */
    .transcript-row {
        padding: 10px 14px;
        margin-bottom: 6px;
        background: rgba(15,23,42,0.35);
        border-radius: 8px;
        border-left: 3px solid var(--accent-cyan);
        display: flex;
        gap: 12px;
        align-items: flex-start;
        transition: background 0.2s ease;
    }
    .transcript-row:hover { background: rgba(30,41,59,0.5); }
    .transcript-ts-link {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-cyan);
        text-decoration: none;
        font-weight: 600;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .transcript-text {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* ── Pipeline Architecture Visual ── */
    .arch-step {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 14px 18px;
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: 10px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .arch-step:hover { border-color: var(--glass-border-hover); }
    .arch-num {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 800;
        color: white;
        flex-shrink: 0;
    }
    .arch-title { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .arch-desc { font-size: 0.8rem; color: var(--text-muted); margin-top: 2px; }

    /* ── Sidebar Overrides ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070c1a 0%, #0c1225 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }

    /* ── Smooth button transitions ── */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        letter-spacing: 0.01em;
    }

    /* ── Custom dividers ── */
    .divider-glow {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(129,140,248,0.3), transparent);
        margin: 24px 0;
        border: none;
    }

    /* ── Scrollable container tweaks ── */
    [data-testid="stVerticalBlock"] > div:has(> .transcript-row) {
        scrollbar-width: thin;
        scrollbar-color: rgba(129,140,248,0.3) transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════
for key, default in {
    "pipeline": None,
    "stats": None,
    "active_url": "",
    "chat_history": [],
    "video_summary": None,
    "chapters": None,
    "raw_docs": [],
    "selected_sample_key": None,
    "total_queries": 0,
    "session_start": time.time(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration & Controls
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.02em;">🎬 YouTube Video</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #818cf8; letter-spacing: -0.02em; margin-top: -4px;">Assistant</div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px; line-height: 1.4;">
                RAG-Powered Q&A Engine<br/>
                <span style="color: #94a3b8;">LangChain • Gemini • FAISS • Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Key
    st.markdown("##### API Key")
    env_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Set GOOGLE_API_KEY in .env or paste here. Offline mode works without a key.",
    )
    active_api_key = api_key_input.strip() if api_key_input else env_api_key

    if active_api_key:
        st.markdown(
            '<div style="margin: 4px 0 12px 0;"><span class="pulse-dot pulse-dot-green"></span><span style="font-size:0.8rem; font-weight:600; color:#34d399;">Gemini API Connected</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="margin: 4px 0 12px 0;"><span class="pulse-dot pulse-dot-amber"></span><span style="font-size:0.8rem; font-weight:600; color:#fbbf24;">Offline / Demo Mode</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("No API key. Running with offline heuristic retrieval.")

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    # ── 1-Click Demo Presets ──
    st.markdown("##### 🌟 Instant Demo Presets")
    st.caption("Load a curated video transcript for instant testing:")

    demo_options = {
        "zjkBMFhNj_g": "🧠 Karpathy — Intro to LLMs (1hr)",
        "UF8uR6Z6KLc": "🎓 Steve Jobs — Stanford '05 (15m)",
        "wjZofJX0v4U": "📐 3B1B — Attention Explained (28m)",
    }

    selected_demo_id = st.selectbox(
        "Select preset:",
        options=list(demo_options.keys()),
        format_func=lambda x: demo_options[x],
        index=0,
        label_visibility="collapsed",
    )

    if st.button("🚀 Load & Index Preset", use_container_width=True, type="primary"):
        st.session_state.selected_sample_key = selected_demo_id
        st.session_state.active_url = f"https://www.youtube.com/watch?v={selected_demo_id}"
        st.rerun()

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    # ── RAG Hyperparameter Studio ──
    st.markdown("##### ⚙️ RAG Configuration")
    with st.expander("Tune Pipeline Parameters", expanded=False):
        model_selection = st.selectbox(
            "LLM Model",
            options=SUPPORTED_CHAT_MODELS,
            index=0,
            help="High-throughput Gemini reasoning with verified candidate fallback.",
        )
        retrieval_mode = st.selectbox(
            "Search Strategy",
            options=["similarity", "mmr"],
            index=0,
            help="Similarity: nearest neighbors. MMR: relevance + diversity balance.",
        )
        chunk_size = st.slider("Chunk Size (chars)", 400, 1600, 800, 100)
        chunk_overlap = st.slider("Chunk Overlap (chars)", 50, 300, 150, 25)
        top_k = st.slider("Top-K Evidence Chunks", 2, 8, 4, 1)

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    # ── Session Analytics ──
    st.markdown("##### 📈 Session Analytics")
    elapsed_min = int((time.time() - st.session_state.session_start) / 60)
    st.markdown(
        f"""
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <div style="background: var(--bg-card); border: 1px solid var(--glass-border); border-radius: 8px; padding: 8px 12px; flex: 1; text-align: center; min-width: 70px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--accent-indigo);">{st.session_state.total_queries}</div>
                <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Queries</div>
            </div>
            <div style="background: var(--bg-card); border: 1px solid var(--glass-border); border-radius: 8px; padding: 8px 12px; flex: 1; text-align: center; min-width: 70px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan);">{elapsed_min}m</div>
                <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Session</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    if st.button("🔄 Reset Session", use_container_width=True):
        for k in ["pipeline", "stats", "active_url", "chat_history", "video_summary", "chapters", "raw_docs", "selected_sample_key"]:
            st.session_state[k] = [] if k in ("chat_history", "raw_docs") else ("" if k == "active_url" else None)
        st.session_state.total_queries = 0
        st.session_state.session_start = time.time()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# HELPER — Build / Rebuild Pipeline
# ══════════════════════════════════════════════════════════════════════
def process_video_source(url_or_id: str, custom_text: str = ""):
    with st.spinner("🔄 Extracting transcript → Chunking documents → Computing embeddings → Building FAISS index..."):
        try:
            if custom_text and custom_text.strip():
                docs, stats = parse_custom_transcript(custom_text)
            else:
                docs, stats = fetch_transcript(url_or_id)

            pipeline = RAGPipeline(
                api_key=active_api_key,
                model_name=model_selection,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                search_type=retrieval_mode,
            )
            pipeline.build_vector_store(docs, stats)

            st.session_state.pipeline = pipeline
            st.session_state.stats = stats
            st.session_state.raw_docs = docs
            st.session_state.active_url = url_or_id
            st.session_state.chat_history = []
            st.session_state.video_summary = None
            st.session_state.chapters = None
            st.session_state.total_queries = 0
            st.success("✅ Transcript extracted, embedded, and FAISS vector index built successfully!")
        except Exception as err:
            st.error(f"❌ {str(err)}")


# ── Auto-trigger preset from sidebar ──
if st.session_state.selected_sample_key and st.session_state.active_url:
    if (
        not st.session_state.stats
        or st.session_state.stats.get("video_id") != st.session_state.selected_sample_key
    ):
        process_video_source(st.session_state.selected_sample_key)
        st.session_state.selected_sample_key = None


# ══════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">YouTube Video Assistant</div>
        <div class="hero-subtitle">
            End-to-End <strong>Retrieval-Augmented Generation</strong> over long-form YouTube transcripts.
            Ask natural-language questions — get answers strictly grounded in timestamped evidence, with zero hallucination.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tech Stack Badges
st.markdown(
    """
    <div class="badge-row">
        <span class="badge-chip">Python</span>
        <span class="badge-chip">LangChain</span>
        <span class="badge-chip badge-cyan">Gemini API</span>
        <span class="badge-chip badge-rose">FAISS</span>
        <span class="badge-chip badge-green">Streamlit</span>
        <span class="badge-chip">gemini-embedding-001</span>
        <span class="badge-chip badge-amber">Grounding Guardrails</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Video Input Bar
col_input, col_btn = st.columns([4.5, 1])
with col_input:
    url_input = st.text_input(
        "Enter YouTube Video URL or ID",
        value=st.session_state.active_url if st.session_state.active_url else "",
        placeholder="https://www.youtube.com/watch?v=... or paste any YouTube URL / Video ID",
        label_visibility="collapsed",
    )
with col_btn:
    load_clicked = st.button("Index Video", use_container_width=True, type="primary")

# Custom Transcript Paste
with st.expander("Paste Custom Transcript (optional)"):
    st.caption("Supports plain text, timestamped lines, or SRT/VTT formatted text.")
    custom_transcript_input = st.text_area(
        "Paste transcript text:",
        height=100,
        placeholder="[00:00] Welcome to today's lecture on...\n[00:45] The first concept we need to understand is...",
        label_visibility="collapsed",
    )
    if st.button("Index Custom Transcript"):
        if custom_transcript_input.strip():
            process_video_source("custom_upload", custom_transcript_input)
        else:
            st.warning("Paste some transcript text first.")

# Trigger processing
if load_clicked and url_input.strip() and url_input.strip() != st.session_state.active_url:
    process_video_source(url_input.strip())


# Main content area
if st.session_state.stats and st.session_state.pipeline:
    stats = st.session_state.stats
    pipeline = st.session_state.pipeline
    diag = pipeline.get_index_stats()

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    # Video Preview + Metadata
    col_vid, col_meta = st.columns([1.5, 2.5])

    with col_vid:
        vid_id = stats.get("video_id", "")
        if vid_id and vid_id != "custom":
            st.markdown(
                f"""
                <div style="border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08);
                     box-shadow: 0 12px 40px -8px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.03);">
                    <iframe width="100%" height="230"
                        src="https://www.youtube.com/embed/{vid_id}"
                        title="Video Player" frameborder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen>
                    </iframe>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.image(stats.get("thumbnail_url", ""), use_container_width=True)

    with col_meta:
        st.markdown(f"### {stats.get('title', 'Video')}")
        st.markdown(f"**By:** `{stats.get('author_name', 'YouTube Creator')}`")
        if stats.get("youtube_url"):
            st.markdown(f"[Watch on YouTube ↗]({stats['youtube_url']})")

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = [
            ("⏱️", stats.get("duration", "N/A"), "Duration", m1),
            ("📝", f"{stats.get('total_words', 0):,}", "Words", m2),
            ("🧩", str(diag.get("total_vectors", 0)), "Chunks", m3),
            ("📐", f"{diag.get('dimension', 768)}d", "Embed Dim", m4),
            ("🔍", diag.get("search_type", "sim").upper()[:3], "Strategy", m5),
        ]
        for icon, val, label, col in metrics:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <span class="metric-icon">{icon}</span>
                        <div class="metric-val-sm">{val}</div>
                        <div class="metric-lbl">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    tab_chat, tab_summary, tab_chapters, tab_transcript, tab_arch = st.tabs([
        "Chat",
        "Summary",
        "Chapters",
        "Transcript",
        "Architecture",
    ])

    # Tab 1: Chat
    with tab_chat:
        st.markdown("#### Grounded Q&A")
        st.caption(
            "Answers are synthesized strictly from retrieved transcript context. "
            "Inspect evidence below for timestamps and relevance scores."
        )

        # Suggested Questions
        vid_id = stats.get("video_id", "")
        if vid_id in SAMPLE_VIDEOS and "sample_questions" in SAMPLE_VIDEOS[vid_id]:
            sample_qs = SAMPLE_VIDEOS[vid_id]["sample_questions"]
            st.markdown(
                "<div style='font-size:0.78rem; font-weight:600; color:#64748b; margin:12px 0 6px 0; text-transform:uppercase; letter-spacing:0.06em;'>Suggested Questions</div>",
                unsafe_allow_html=True,
            )
            q_cols = st.columns(2)
            for idx, q_text in enumerate(sample_qs):
                with q_cols[idx % 2]:
                    if st.button(f"→ {q_text}", key=f"sq_{idx}", use_container_width=True):
                        with st.spinner("Retrieving evidence & synthesizing grounded answer..."):
                            result = pipeline.ask(q_text)
                            st.session_state.chat_history.append(result)
                            st.session_state.total_queries += 1
                            st.rerun()

        # Chat History
        if not st.session_state.chat_history:
            st.info("👋 Ask a question below, or click a suggested question above to get started!")
        else:
            for item in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(item["question"])

                with st.chat_message("assistant"):
                    st.markdown(item["answer"])

                    # Grounding Badges
                    conf = item.get("confidence", "Medium")
                    avg_sim = int(round(item.get("avg_similarity", 0.0) * 100))
                    latency = item.get("latency_ms", 0.0)
                    n_sources = len(item.get("sources", []))

                    conf_badge = "badge-green" if conf == "High" else ("badge-chip" if conf == "Medium" else "badge-amber")
                    mode_badge = "badge-amber" if item.get("offline_mode") else "badge-cyan"
                    mode_text = "Offline" if item.get("offline_mode") else "Gemini"

                    st.markdown(
                        f"""
                        <div class="badge-row" style="margin-top: 8px;">
                            <span class="badge-chip {conf_badge}">🛡️ {conf} Confidence · {avg_sim}%</span>
                            <span class="badge-chip">⏱️ {latency}ms</span>
                            <span class="badge-chip">📚 {n_sources} Sources</span>
                            <span class="badge-chip {mode_badge}">🤖 {mode_text}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Evidence Inspector
                    sources = item.get("sources", [])
                    if sources:
                        with st.expander(f"🔎 Evidence Grounding Inspector — {n_sources} Retrieved Chunks", expanded=False):
                            for src in sources:
                                relevance = src.get("relevance_percent", 0)
                                score_class = "score-high" if relevance >= 60 else ("score-medium" if relevance >= 35 else "score-low")
                                st.markdown(
                                    f"""
                                    <div class="evidence-card">
                                        <div class="evidence-header">
                                            <span class="evidence-ts">
                                                ⏱️ <a href="{src.get('youtube_url', '#')}" target="_blank" style="color: var(--accent-cyan); text-decoration: none;">{src.get('time_range', 'N/A')}</a>
                                                &nbsp;·&nbsp; Chunk #{src.get('chunk_id', 0)}
                                            </span>
                                            <span class="evidence-score {score_class}">{relevance}% Match</span>
                                        </div>
                                        <div class="evidence-body">"{src.get('text', '')}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

        # Chat Input
        user_query = st.chat_input("Ask a question about this video transcript...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching FAISS index → Retrieving evidence → Generating grounded answer..."):
                    result = pipeline.ask(user_query)
                    st.markdown(result["answer"])
                    st.session_state.chat_history.append(result)
                    st.session_state.total_queries += 1
                    st.rerun()

    # Tab 2: Summary
    with tab_summary:
        st.markdown("#### Summary")
        st.caption("Structured overview covering key concepts, quotes, and takeaways.")

        if st.session_state.video_summary:
            st.markdown(st.session_state.video_summary)
        else:
            if st.button("Generate Summary", type="primary", use_container_width=False):
                with st.spinner("Analyzing transcript and synthesizing summary..."):
                    summary_result = pipeline.generate_summary()
                    st.session_state.video_summary = summary_result.get("summary", "Summary unavailable.")
                    st.rerun()
            else:
                st.info("Click the button to generate an AI summary of the video transcript.")

    # Tab 3: Chapters
    with tab_chapters:
        st.markdown("#### Chapters & Timeline")
        st.caption("Detected topic milestones with direct timestamp navigation.")

        if not st.session_state.chapters:
            if st.button("Generate Chapters", type="primary"):
                with st.spinner("Segmenting transcript into topic milestones..."):
                    st.session_state.chapters = pipeline.generate_chapters()
                    st.rerun()
            else:
                st.info("Click above to segment the video into timestamped chapters.")
        else:
            for i, ch in enumerate(st.session_state.chapters):
                st.markdown(
                    f"""
                    <div class="chapter-card">
                        <div class="chapter-num">{i + 1}</div>
                        <span class="chapter-ts">{ch['timestamp']}</span>
                        <span class="chapter-title">{ch['title']}</span>
                        <a class="chapter-link" href="{ch['youtube_url']}" target="_blank">Watch ↗</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Tab 4: Transcript Explorer
    with tab_transcript:
        st.markdown("#### Transcript Explorer")
        st.caption("Browse timestamped segments with real-time keyword filtering.")

        search_filter = st.text_input(
            "Filter by keyword:",
            placeholder="e.g. hallucination, attention, transformer, calligraphy...",
        )

        raw_docs = st.session_state.raw_docs
        filtered_docs = [
            d for d in raw_docs
            if not search_filter or search_filter.lower() in d.page_content.lower()
        ]

        st.caption(f"Showing **{len(filtered_docs)}** of {len(raw_docs)} segments")

        with st.container(height=450):
            for doc in filtered_docs:
                ts = doc.metadata.get("timestamp", "00:00")
                ts_end = doc.metadata.get("timestamp_end", ts)
                yt_link = doc.metadata.get("youtube_url", "#")
                content = doc.page_content.split("]")[-1].strip()

                st.markdown(
                    f"""
                    <div class="transcript-row">
                        <a class="transcript-ts-link" href="{yt_link}" target="_blank">[{ts}–{ts_end}]</a>
                        <span class="transcript-text">{content}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Tab 5: Architecture & Stats
    with tab_arch:
        st.markdown("#### Pipeline Architecture & Diagnostics")
        st.caption("Technical breakdown of the retrieval pipeline.")

        col_arch, col_diag = st.columns([1.6, 1])

        with col_arch:
            st.markdown("##### Pipeline Stages")
            stages = [
                ("1", "#818cf8", "Transcript Extraction", "youtube-transcript-api + oEmbed metadata + multi-language fallback"),
                ("2", "#c084fc", "Document Chunking", f"RecursiveCharacterTextSplitter — {chunk_size} chars, {chunk_overlap} overlap"),
                ("3", "#38bdf8", "Embedding Generation", "Google gemini-embedding-001 (or offline vectorizer) → dense vectors"),
                ("4", "#f472b6", "FAISS Vector Indexing", f"IndexFlatL2 in-memory store — {diag.get('total_vectors', 0)} vectors indexed"),
                ("5", "#34d399", "Similarity Search", f"Top-{top_k} evidence retrieval ({retrieval_mode.upper()})"),
                ("6", "#fbbf24", "Grounded Synthesis", "Context-aware prompt + anti-hallucination guardrail → Gemini LLM"),
            ]
            for num, color, title, desc in stages:
                st.markdown(
                    f"""
                    <div class="arch-step">
                        <div class="arch-num" style="background: {color};">{num}</div>
                        <div>
                            <div class="arch-title">{title}</div>
                            <div class="arch-desc">{desc}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_diag:
            st.markdown("##### Live Index Diagnostics")
            st.json(diag)

            st.markdown("##### Hallucination Guardrails")
            st.markdown(
                """
                1. **Time-Anchored Chunks** — Each chunk preserves exact `start → end` timestamps
                2. **Strict Grounding Directive** — LLM must cite context or declare "unanswerable"
                3. **Relevance Scoring** — FAISS distances → normalized 0–100% confidence
                4. **Evidence Inspector** — Auditable verbatim source passages per answer
                """
            )

        # Export
        st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)
        st.markdown("##### 💾 Export Session Report")
        chat_text = "\n\n".join([f"**Q:** {c['question']}\n**A:** {c['answer']}" for c in st.session_state.chat_history])
        export_content = f"""# YouTube Video Assistant — Session Report
**Video:** {stats.get('title')}
**Author:** {stats.get('author_name')}
**Duration:** {stats.get('duration')} | **Words:** {stats.get('total_words'):,}
**URL:** {stats.get('youtube_url')}
**Queries:** {st.session_state.total_queries}

## Summary
{st.session_state.video_summary or 'Not generated.'}

## Q&A History
{chat_text or 'No questions asked.'}
"""
        st.download_button(
            label="📥 Download Report (.md)",
            data=export_content,
            file_name=f"rag_report_{stats.get('video_id', 'video')}.md",
            mime="text/markdown",
        )
