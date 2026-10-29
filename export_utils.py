"""Export Utilities for Study Guides, Q&A Transcripts, and Research Briefings.

Generates production-quality Markdown and JSON reports for researchers,
students, and engineers.
"""

import json
import time
from typing import Any, Dict, List, Optional


def generate_study_guide_markdown(
    video_metadata: Dict[str, Any],
    summary_text: str = "",
    chapters: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    analytics_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate comprehensive, beautifully formatted Markdown study guide."""
    title = video_metadata.get("title", "YouTube Video")
    author = video_metadata.get("author_name", "Unknown Creator")
    url = video_metadata.get("youtube_url", "")
    duration = video_metadata.get("duration", "N/A")
    words = video_metadata.get("total_words", 0)
    generated_date = time.strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Study Guide: {title}\n")
    md.append(f"**Speaker / Channel:** {author}  ")
    if url:
        md.append(f"**Source Video:** [{url}]({url})  ")
    md.append(f"**Duration:** {duration} | **Total Word Count:** {words:,}  ")
    md.append(f"**Generated On:** {generated_date} via **LLM-Powered YouTube Video Assistant (RAG)**\n")
    md.append("---\n")

    # 1. Executive Summary
    if summary_text:
        md.append("## Executive Summary\n")
        md.append(f"{summary_text.strip()}\n\n---\n")

    # 2. Key Timeline Chapters
    if chapters:
        md.append("## Chapters & Timeline\n")
        for ch in chapters:
            ts = ch.get("timestamp", "00:00")
            ch_title = ch.get("title", "Chapter")
            desc = ch.get("description", "")
            md.append(f"- **`{ts}`** **{ch_title}**: {desc}")
        md.append("\n---\n")

    # 3. Grounded Q&A Transcripts
    if conversation_history:
        md.append("## Questions and Answers\n")
        for i, exchange in enumerate(conversation_history, 1):
            q = exchange.get("question", "")
            a = exchange.get("answer", "")
            conf = exchange.get("confidence", "N/A")
            sources = exchange.get("sources", [])

            md.append(f"### Q{i}: {q}\n")
            md.append(f"**Confidence Level:** {conf}  \n")
            md.append(f"{a}\n")

            if sources:
                md.append("\n**Retrieved Sources:**")
                for s in sources:
                    ts_range = s.get("time_range", "")
                    rel = s.get("relevance", "N/A")
                    txt = s.get("text", "").replace("\n", " ")
                    md.append(f"- `[{ts_range}]` *(Relevance: {rel})*: {txt[:160]}...")
            md.append("\n---\n")

    # 4. RAG Retrieval Telemetry
    if analytics_summary:
        md.append("## Retrieval Metrics\n")
        md.append(f"- **Total Queries Processed:** {analytics_summary.get('total_queries', 0)}")
        md.append(f"- **Average Retrieval & Generation Latency:** {analytics_summary.get('avg_latency_ms', 0)} ms")
        md.append(f"- **Average Retrieval Confidence:** {analytics_summary.get('avg_confidence', 0)}%")
        md.append(f"- **High Confidence Answers:** {analytics_summary.get('high_confidence_pct', 0)}%")
        md.append("\n---\n")

    md.append("*Generated with LLM-Powered YouTube Video Assistant (RAG) — Grounded AI Verification.*")
    return "\n".join(md)


def generate_session_json(
    video_metadata: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    analytics_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate structured JSON representation of the entire RAG session."""
    data = {
        "metadata": video_metadata,
        "exported_at": time.time(),
        "analytics": analytics_summary or {},
        "conversation": conversation_history or [],
    }
    return json.dumps(data, indent=2)
