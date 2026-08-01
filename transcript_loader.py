import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document


def extract_video_id(url_or_id):
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|\/|shorts\/|embed\/)([0-9A-Za-z_-]{11})", url_or_id)
    if match:
        return match.group(1)
    if len(url_or_id) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    return None


def format_timestamp(seconds):
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def fetch_transcript(video_url_or_id):
    video_id = extract_video_id(video_url_or_id)
    if not video_id:
        raise ValueError("Invalid YouTube video URL or ID.")

    ytt = YouTubeTranscriptApi()
    try:
        data = ytt.fetch(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        raise ValueError("Subtitles or transcripts are disabled for this video.")
    except Exception:
        try:
            data = ytt.list(video_id).find_transcript(["en"]).fetch()
        except Exception as err:
            raise ValueError(f"Failed to extract transcript: {str(err)}")

    docs = []
    full_text_list = []
    total_duration = 0.0

    for item in data:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            start = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
        else:
            text = str(getattr(item, "text", "")).strip()
            start = float(getattr(item, "start", 0.0))
            duration = float(getattr(item, "duration", 0.0))

        if not text:
            continue

        ts_str = format_timestamp(start)
        formatted_content = f"[{ts_str}] {text}"
        metadata = {
            "video_id": video_id,
            "start": start,
            "duration": duration,
            "timestamp": ts_str,
        }

        docs.append(Document(page_content=formatted_content, metadata=metadata))
        full_text_list.append(text)

        if start + duration > total_duration:
            total_duration = start + duration

    full_text = " ".join(full_text_list)
    stats = {
        "video_id": video_id,
        "segment_count": len(docs),
        "total_words": len(full_text.split()),
        "duration": format_timestamp(total_duration),
        "full_text": full_text,
    }

    return docs, stats
