"""Session Telemetry and Performance Analytics for RAG Pipeline.

Tracks query latencies, vector retrieval confidence distributions,
source chunk citation counts, and session summary metrics.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD


@dataclass
class QueryTelemetryRecord:
    """Telemetry data captured for an individual RAG query."""

    query: str
    latency_ms: float
    confidence_score: float
    confidence_level: str
    source_count: int
    top_timestamp: str
    model_name: str
    created_at: float = field(default_factory=time.time)


class SessionAnalytics:
    """Manages session-level performance analytics and telemetry."""

    def __init__(self):
        self.records: List[QueryTelemetryRecord] = []
        self.total_tokens_estimated: int = 0

    def record_query(
        self,
        query: str,
        latency_ms: float,
        confidence_score: float,
        source_count: int,
        top_timestamp: str = "N/A",
        model_name: str = "gemini-1.5-flash",
    ) -> QueryTelemetryRecord:
        """Record telemetry for a single query invocation."""
        if confidence_score >= HIGH_CONFIDENCE_THRESHOLD:
            level = "High"
        elif confidence_score >= MEDIUM_CONFIDENCE_THRESHOLD:
            level = "Medium"
        else:
            level = "Low"

        record = QueryTelemetryRecord(
            query=query,
            latency_ms=round(latency_ms, 2),
            confidence_score=round(confidence_score, 1),
            confidence_level=level,
            source_count=source_count,
            top_timestamp=top_timestamp,
            model_name=model_name,
        )
        self.records.append(record)
        # Approximate 1 token per 4 characters
        self.total_tokens_estimated += len(query) // 4 + (source_count * 200)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregated metrics across all queries in this session."""
        if not self.records:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0,
                "high_confidence_pct": 0.0,
                "medium_confidence_pct": 0.0,
                "low_confidence_pct": 0.0,
                "total_tokens_estimated": 0,
            }

        total = len(self.records)
        avg_latency = sum(r.latency_ms for r in self.records) / total
        avg_conf = sum(r.confidence_score for r in self.records) / total

        high_count = sum(1 for r in self.records if r.confidence_level == "High")
        med_count = sum(1 for r in self.records if r.confidence_level == "Medium")
        low_count = sum(1 for r in self.records if r.confidence_level == "Low")

        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_confidence": round(avg_conf, 1),
            "high_confidence_pct": round((high_count / total) * 100, 1),
            "medium_confidence_pct": round((med_count / total) * 100, 1),
            "low_confidence_pct": round((low_count / total) * 100, 1),
            "total_tokens_estimated": self.total_tokens_estimated,
        }

    def get_records_as_dicts(self) -> List[Dict[str, Any]]:
        """Return list of telemetry records as dictionaries for tabular display."""
        return [asdict(r) for r in self.records]

    def clear(self):
        """Reset telemetry tracker."""
        self.records.clear()
        self.total_tokens_estimated = 0
