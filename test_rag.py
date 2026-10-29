"""Comprehensive Unit & Integration Test Suite for YouTube Video Assistant (RAG).

Validates:
1. Video ID & URL parsing across multiple formats
2. Timestamp formatting and mathematical parsing
3. Sample datasets integrity and metadata resolution
4. Embeddings factory and OfflineFallbackEmbeddings vector normalization
5. RAG chunking and FAISS vector index creation
6. Semantic retrieval and L2 confidence score normalization
7. Grounded answer synthesis and citation generation
8. Session telemetry and performance analytics
9. Study guide Markdown and JSON exports
"""

import math
import sys

from config import PipelineConfig, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from video_metadata import (
    extract_video_id,
    format_timestamp,
    parse_timestamp_to_seconds,
    fetch_video_metadata,
)
from sample_data import SAMPLE_VIDEOS, get_sample_video, list_sample_videos
from embeddings import OfflineFallbackEmbeddings, create_embeddings_engine
from analytics import SessionAnalytics
from export_utils import generate_study_guide_markdown, generate_session_json
from transcript_loader import (
    fetch_transcript,
    parse_custom_transcript,
    create_documents_from_segments,
)
from rag_pipeline import RAGPipeline


def test_video_metadata_and_url_parser():
    print("[1/8] Testing Video URL & ID Extraction...")
    urls = [
        ("https://www.youtube.com/watch?v=zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("https://youtu.be/zjkBMFhNj_g?t=45", "zjkBMFhNj_g"),
        ("https://www.youtube.com/embed/zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("https://www.youtube.com/shorts/zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("https://www.youtube.com/live/zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("zjkBMFhNj_g", "zjkBMFhNj_g"),
    ]
    for url, expected in urls:
        extracted = extract_video_id(url)
        assert extracted == expected, f"Failed on {url}: expected {expected}, got {extracted}"

    print("  ✓ All YouTube URL formats parsed correctly")

    print("  Testing timestamp conversion...")
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3665) == "01:01:05"
    assert parse_timestamp_to_seconds("01:05") == 65.0
    assert parse_timestamp_to_seconds("01:01:05") == 3665.0
    print("  ✓ Timestamp conversions verified")


def test_sample_datasets():
    print("[2/8] Testing Benchmark Sample Datasets...")
    samples = list_sample_videos()
    assert len(samples) >= 3, "Expected at least 3 curated benchmark datasets"
    for s in samples:
        assert "video_id" in s
        assert "title" in s
        assert "segments" in s
        assert len(s["segments"]) > 0
        assert "sample_questions" in s
    print(f"  ✓ {len(samples)} curated sample datasets validated")


def test_embeddings_engine():
    print("[3/8] Testing Embeddings Engine...")
    embedder = OfflineFallbackEmbeddings(dimension=256)
    vec = embedder.embed_query("Large Language Models and Retrieval Augmented Generation")
    assert len(vec) == 256
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4, "Embedding vector is not unit normalized"

    docs_vecs = embedder.embed_documents(["First document test", "Second document test"])
    assert len(docs_vecs) == 2
    assert len(docs_vecs[0]) == 256

    engine, is_offline, label = create_embeddings_engine(api_key=None)
    assert is_offline is True
    assert "Offline" in label
    print("  ✓ OfflineFallbackEmbeddings and factory passed with unit normalization")


def test_transcript_loaders():
    print("[4/8] Testing Transcript Loaders...")
    docs, stats = fetch_transcript("zjkBMFhNj_g")
    assert len(docs) > 0
    assert stats["video_id"] == "zjkBMFhNj_g"
    assert stats["total_words"] > 0
    assert "duration" in stats
    print(f"  ✓ Curated fetch returned {len(docs)} segments, {stats['total_words']} words")

    custom_text = """[00:00] Welcome to this deep dive into neural networks.
[00:30] Today we will implement backpropagation from scratch in Python.
[01:15] Let us begin by defining the forward pass."""
    c_docs, c_stats = parse_custom_transcript(custom_text, title="Neural Nets from Scratch")
    assert len(c_docs) == 3
    assert c_stats["title"] == "Neural Nets from Scratch"
    print("  ✓ Custom timestamped transcript parsed into 3 segments")


def test_rag_vector_store_and_search():
    print("[5/8] Testing RAG Pipeline FAISS Vector Store...")
    docs, stats = fetch_transcript("zjkBMFhNj_g")
    pipeline = RAGPipeline(
        api_key=None,
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        search_type="similarity",
    )
    idx_stats = pipeline.build_vector_store(docs, stats)

    assert idx_stats["total_vectors"] > 0
    assert idx_stats["chunk_size"] == 500
    assert idx_stats["offline_mode"] is True

    # Vector similarity search with scores
    results = pipeline.retrieve_relevant_chunks("What is RLHF and reward model?")
    assert len(results) == 3
    for doc, dist in results:
        assert dist >= 0.0
        assert "timestamp" in doc.metadata

    print(f"  ✓ FAISS indexed {idx_stats['total_vectors']} chunks; similarity search verified")


def test_grounded_qa_and_guardrails():
    print("[6/8] Testing Grounded Q&A Synthesis...")
    docs, stats = fetch_transcript("zjkBMFhNj_g")
    pipeline = RAGPipeline(api_key=None, top_k=3)
    pipeline.build_vector_store(docs, stats)

    result = pipeline.ask("How does Karpathy explain hallucinations in language models?")
    assert "answer" in result
    assert len(result["sources"]) > 0
    assert "confidence" in result
    assert result["is_grounded"] is True
    assert result["latency_ms"] > 0

    first_source = result["sources"][0]
    assert "time_range" in first_source
    assert "relevance" in first_source
    print(f"  ✓ Grounded answer synthesized (Confidence: {result['confidence']}, Latency: {result['latency_ms']}ms)")


def test_intelligence_features():
    print("[7/8] Testing Intelligence Generators (Summary, Chapters, Suggestions)...")
    docs, stats = fetch_transcript("zjkBMFhNj_g")
    pipeline = RAGPipeline(api_key=None)
    pipeline.build_vector_store(docs, stats)

    summary = pipeline.generate_summary()
    assert "summary" in summary
    assert "Executive Summary" in summary["summary"]

    chapters = pipeline.generate_chapters()
    assert len(chapters) > 0
    assert "timestamp" in chapters[0]

    questions = pipeline.get_suggested_questions(summary["summary"])
    assert len(questions) >= 3
    print(f"  OK: generated summary, {len(chapters)} chapters, and {len(questions)} suggested questions")


def test_telemetry_and_exports():
    print("[8/8] Testing telemetry and exports...")
    analytics = SessionAnalytics()
    analytics.record_query(
        query="What is pre-training?",
        latency_ms=120.5,
        confidence_score=88.4,
        source_count=3,
        top_timestamp="01:45",
    )
    analytics.record_query(
        query="What are hallucinations?",
        latency_ms=95.2,
        confidence_score=68.0,
        source_count=4,
        top_timestamp="07:00",
    )
    summary = analytics.get_summary()
    assert summary["total_queries"] == 2
    assert summary["high_confidence_pct"] == 50.0
    assert summary["medium_confidence_pct"] == 50.0
    print(f"  OK: telemetry recorded 2 queries (avg latency: {summary['avg_latency_ms']}ms)")

    # Markdown and JSON export
    meta = {"title": "Intro to LLMs", "author_name": "Andrej Karpathy", "duration": "1:00:00", "total_words": 1500}
    conv = [{"question": "Q1", "answer": "A1", "confidence": "High", "sources": []}]
    md_report = generate_study_guide_markdown(meta, "Summary text", [], conv, summary)
    assert "Study Guide" in md_report

    json_report = generate_session_json(meta, conv, summary)
    assert '"title": "Intro to LLMs"' in json_report
    print("  OK: markdown and json reports generated")


if __name__ == "__main__":
    try:
        print("Running RAG test suite...")
        test_video_metadata_and_url_parser()
        test_sample_datasets()
        test_embeddings_engine()
        test_transcript_loaders()
        test_rag_vector_store_and_search()
        test_grounded_qa_and_guardrails()
        test_intelligence_features()
        test_telemetry_and_exports()
        print("\nAll 8 tests passed successfully.")
    except Exception as e:
        print(f"\nTest failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
