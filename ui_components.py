"""Ultra Hi-Fi UI Components and Styles for Streamlit Interface.

Delivers dark glassmorphism, animated glow badges, pipeline architecture flow,
and evidence cards.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
from config import APP_NAME, VERSION


def inject_custom_css():
    """Inject premium CSS styling into Streamlit application."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg-deep: #050816;
            --bg-card: rgba(15, 23, 42, 0.70);
            --bg-card-hover: rgba(30, 41, 59, 0.80);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-border-hover: rgba(129, 140, 248, 0.40);
            --accent-indigo: #818cf8;
            --accent-violet: #a78bfa;
            --accent-cyan: #22d3ee;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --accent-rose: #fb7185;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        .stApp {
            background: linear-gradient(180deg, #050816 0%, #0a0f1e 40%, #0d1321 100%) !important;
            color: var(--text-primary);
        }

        /* ── Animated Hero Header ── */
        .hero-banner {
            background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.18), transparent 70%),
                        linear-gradient(135deg, rgba(15, 23, 42, 0.90), rgba(2, 6, 23, 0.95));
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 28px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }

        .hero-banner::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #3b82f6);
            background-size: 300% 100%;
            animation: gradientMove 6s ease infinite;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 30%, #c7d2fe 70%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            line-height: 1.2;
        }

        .hero-subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            max-width: 820px;
            line-height: 1.6;
            margin-bottom: 20px;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #cbd5e1;
            margin-right: 8px;
            margin-bottom: 6px;
        }

        /* ── Glass Metric Card ── */
        .glass-metric {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 16px 20px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-metric:hover {
            transform: translateY(-2px);
            border-color: var(--glass-border-hover);
            box-shadow: 0 12px 24px -10px rgba(99, 102, 241, 0.25);
        }

        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-sub {
            font-size: 0.72rem;
            color: var(--accent-indigo);
            margin-top: 3px;
        }

        /* ── Evidence Cards ── */
        .evidence-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 3px solid var(--accent-indigo);
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }

        .evidence-card:hover {
            background: rgba(30, 41, 59, 0.7);
            border-left-color: var(--accent-cyan);
        }

        .evidence-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .timestamp-badge {
            background: rgba(99, 102, 241, 0.2);
            color: #c7d2fe;
            border: 1px solid rgba(99, 102, 241, 0.4);
            padding: 2px 10px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 600;
            text-decoration: none;
        }

        .confidence-high { color: #34d399; font-weight: 700; }
        .confidence-med { color: #fbbf24; font-weight: 700; }
        .confidence-low { color: #fb7185; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header():
    """Render animated top hero header banner."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">🎬 {APP_NAME}</div>
            <div class="hero-subtitle">
                Production-grade Retrieval-Augmented Generation engine. Query long-form YouTube video transcripts
                with zero hallucinations, vector similarity search, and verbatim timestamp citations.
            </div>
            <div>
                <span class="badge-pill">LangChain 1.4+</span>
                <span class="badge-pill">Gemini Models</span>
                <span class="badge-pill">gemini-embedding-001</span>
                <span class="badge-pill">FAISS Vector Store</span>
                <span class="badge-pill">Grounding Guardrails</span>
                <span class="badge-pill" style="border-color: rgba(52, 211, 153, 0.4); color: #34d399;">v{VERSION}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_architecture():
    """Render interactive numbered pipeline architecture flow."""
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 18px 24px; margin-bottom: 24px;">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #818cf8; margin-bottom: 12px;">
                End-to-End RAG Pipeline
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
                <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border-left: 2px solid #818cf8;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">1. EXTRACTION</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #f1f5f9; margin: 4px 0;">YouTube Transcript API</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Timed captions & oEmbed metadata</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border-left: 2px solid #a78bfa;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">2. CHUNKING</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #f1f5f9; margin: 4px 0;">Recursive Splitter</div>
                    <div style="font-size: 0.72rem; color: #64748b;">800 char windows + 150 overlap</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border-left: 2px solid #22d3ee;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">3. EMBEDDINGS</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #f1f5f9; margin: 4px 0;">gemini-embedding-001</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Dense semantic vectors (or offline)</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border-left: 2px solid #34d399;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">4. VECTOR INDEX</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #f1f5f9; margin: 4px 0;">FAISS Vector Store</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Sub-millisecond L2 similarity search</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border-left: 2px solid #fb7185;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">5. GROUNDED LLM</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #f1f5f9; margin: 4px 0;">Gemini LLM Synthesis</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Grounded citations & zero hallucinations</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_chunk(chunk: Dict[str, Any], index: int):
    """Render individual retrieved source chunk card."""
    time_range = chunk.get("time_range", "00:00")
    relevance = chunk.get("relevance", "N/A")
    conf_level = chunk.get("confidence_level", "Medium")
    text = chunk.get("text", "")
    url = chunk.get("url", "")

    conf_class = (
        "confidence-high"
        if conf_level == "High"
        else "confidence-med"
        if conf_level == "Medium"
        else "confidence-low"
    )

    url_btn = (
        f'<a href="{url}" target="_blank" class="timestamp-badge">▶ {time_range}</a>'
        if url
        else f'<span class="timestamp-badge">{time_range}</span>'
    )

    st.markdown(
        f"""
        <div class="evidence-card">
            <div class="evidence-meta">
                <div>
                    <span style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; margin-right: 8px;">SOURCE CHUNK #{index}</span>
                    {url_btn}
                </div>
                <div style="font-size: 0.78rem;">
                    Relevance: <span class="{conf_class}">{relevance} ({conf_level})</span>
                </div>
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; font-style: italic;">
                "{text}"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
