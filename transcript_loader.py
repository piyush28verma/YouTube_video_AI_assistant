"""Transcript Loader and YouTube Metadata Extractor for YouTube Video AI Assistant.

Handles:
- Universal YouTube URL/ID extraction via video_metadata
- Multi-language transcript extraction with YouTubeTranscriptApi
- Fallback to curated benchmark datasets from sample_data
- Custom transcript parsing (plain text, SRT/VTT formatted)
- Rich LangChain Document generation with precise timing metadata
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# Re-export key utilities for backward compatibility
from video_metadata import (
    extract_video_id,
    format_timestamp,
    parse_timestamp_to_seconds,
    fetch_video_metadata,
)
from sample_data import SAMPLE_VIDEOS, get_sample_video, list_sample_videos
from config import DEFAULT_PREFERRED_LANGUAGES


def create_documents_from_segments(
    segments: List[Dict[str, Any]],
    video_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Convert raw transcript segments into LangChain Documents with full timing metadata."""
    if metadata is None:
        metadata = fetch_video_metadata(video_id)

    docs: List[Document] = []
    full_text_list: List[str] = []
    total_duration = 0.0

    for i, item in enumerate(segments):
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))
        text = str(item.get("text", "")).strip()

        if not text:
            continue

        end = start + duration
        ts_start = format_timestamp(start)
        ts_end = format_timestamp(end)

        content = f"[{ts_start} - {ts_end}] {text}"
        doc_meta = {
            "video_id": video_id,
            "segment_index": i,
            "start": start,
            "end": end,
            "duration": duration,
            "timestamp": ts_start,
            "timestamp_end": ts_end,
            "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
            "video_title": metadata.get("title", f"Video {video_id}"),
        }

        docs.append(Document(page_content=content, metadata=doc_meta))
        full_text_list.append(text)
        if end > total_duration:
            total_duration = end

    full_text = " ".join(full_text_list)
    word_count = len(full_text.split())

    stats = {
        "video_id": video_id,
        "title": metadata.get("title", f"Video {video_id}"),
        "author_name": metadata.get("author_name", "YouTube Creator"),
        "thumbnail_url": metadata.get("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"),
        "youtube_url": metadata.get("youtube_url", f"https://www.youtube.com/watch?v={video_id}"),
        "segment_count": len(docs),
        "total_words": word_count,
        "duration": format_timestamp(total_duration),
        "duration_seconds": total_duration,
        "full_text": full_text,
        "is_sample": video_id in SAMPLE_VIDEOS,
    }

    return docs, stats


def fetch_transcript(
    video_url_or_id: str,
    preferred_languages: Optional[List[str]] = None,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Fetch transcript from YouTube or fallback to pre-bundled sample datasets.

    Supports automatic multi-language fallback, oEmbed metadata retrieval,
    and returns rich LangChain Documents.
    """
    if preferred_languages is None:
        preferred_languages = DEFAULT_PREFERRED_LANGUAGES

    video_id = extract_video_id(video_url_or_id)
    if not video_id:
        raise ValueError("Invalid YouTube video URL or ID. Please check the URL format.")

    metadata = fetch_video_metadata(video_id)

    # 1. Check if video matches one of our curated sample datasets
    if video_id in SAMPLE_VIDEOS:
        sample = SAMPLE_VIDEOS[video_id]
        metadata["title"] = sample.get("title", metadata["title"])
        metadata["author_name"] = sample.get("author_name", metadata["author_name"])
        return create_documents_from_segments(sample["segments"], video_id, metadata)

    # 2. Attempt online extraction via YouTubeTranscriptApi
    try:
        ytt = YouTubeTranscriptApi()
        raw_transcript = None

        try:
            # Direct fetch with language preference
            raw_transcript = ytt.fetch(video_id, languages=preferred_languages)
        except Exception:
            # Fallback: list transcripts and find first available transcript
            try:
                transcript_list = ytt.list(video_id)
                try:
                    found = transcript_list.find_transcript(preferred_languages)
                    raw_transcript = found.fetch()
                except Exception:
                    # If not in preferred languages, grab any available and translate if needed
                    for t in transcript_list:
                        if t.is_translatable and "en" in [tr["language_code"] for tr in t.translation_languages]:
                            raw_transcript = t.translate("en").fetch()
                            break
                        else:
                            raw_transcript = t.fetch()
                            break
            except Exception as inner_err:
                raise inner_err

        if not raw_transcript:
            raise ValueError("No transcript text could be retrieved for this video.")

        # Normalize transcript segments
        normalized_segments = []
        for item in raw_transcript:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                start = float(item.get("start", 0.0))
                duration = float(item.get("duration", 0.0))
            else:
                text = str(getattr(item, "text", "")).strip()
                start = float(getattr(item, "start", 0.0))
                duration = float(getattr(item, "duration", 0.0))

            if text:
                normalized_segments.append({"start": start, "duration": duration, "text": text})

        return create_documents_from_segments(normalized_segments, video_id, metadata)

    except (TranscriptsDisabled, NoTranscriptFound):
        raise ValueError(
            f"Subtitles/transcripts are disabled or unavailable for video '{video_id}'. "
            "You can paste custom transcript text below or select one of the curated demo videos from the sidebar."
        )
    except CouldNotRetrieveTranscript as err:
        raise ValueError(
            f"Could not retrieve transcript from YouTube: {str(err)}. "
            "Try a sample video from the sidebar or upload transcript text."
        )
    except Exception as err:
        # Check if user passed a sample video url query
        if video_id in SAMPLE_VIDEOS:
            return create_documents_from_segments(SAMPLE_VIDEOS[video_id]["segments"], video_id, metadata)
        raise ValueError(f"Failed to extract transcript: {str(err)}")


def parse_custom_transcript(
    raw_text: str,
    title: str = "Custom Video Transcript",
) -> Tuple[List[Document], Dict[str, Any]]:
    """Parse custom user-provided transcript text (plain text or timestamped SRT/VTT format)."""
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    if not lines:
        raise ValueError("The provided transcript text is empty.")

    segments = []
    current_time = 0.0

    # Match timestamp pattern like [00:15] or 00:15 or 00:00:15
    ts_pattern = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.*)$")

    for line in lines:
        # Ignore numeric SRT subtitle indices
        if line.isdigit():
            continue

        match = ts_pattern.match(line)
        if match:
            ts_str, text = match.groups()
            current_time = parse_timestamp_to_seconds(ts_str)
            segments.append({"start": current_time, "duration": 15.0, "text": text})
        else:
            # Estimate 15 seconds per line of text
            segments.append({"start": current_time, "duration": 15.0, "text": line})
            current_time += 15.0

    metadata = {
        "video_id": "custom",
        "title": title,
        "author_name": "User Upload",
        "thumbnail_url": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=800&q=80",
        "youtube_url": "",
    }
    return create_documents_from_segments(segments, "custom", metadata)
