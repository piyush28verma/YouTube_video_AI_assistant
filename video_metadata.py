"""YouTube Video Metadata and Timestamp Utilities.

Handles:
- Universal YouTube URL and Video ID extraction
- Direct timestamp math and string formatting
- YouTube oEmbed API metadata retrieval with resilient error handling
"""

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract standard 11-character YouTube video ID from various URL patterns or raw ID.

    Supports:
    - Standard watch URLs: https://www.youtube.com/watch?v=...
    - Shortened URLs: https://youtu.be/...
    - Embed URLs: https://www.youtube.com/embed/...
    - Shorts URLs: https://www.youtube.com/shorts/...
    - Live URLs: https://www.youtube.com/live/...
    - Direct 11-char ID
    """
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()

    # Raw 11-character alphanumeric ID
    if len(url_or_id) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})",
        r"(?:watch\?.*v=)([0-9A-Za-z_-]{11})",
        r"(?:live\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return None


def format_timestamp(seconds: float) -> str:
    """Format seconds into standard [HH:MM:SS] or [MM:SS] display string."""
    total_sec = max(0, int(seconds))
    mins, secs = divmod(total_sec, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Convert timestamp string (HH:MM:SS or MM:SS) back to numerical seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) == 3:
        return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    elif len(parts) == 2:
        return float(int(parts[0]) * 60 + int(parts[1]))
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return 0.0
    return 0.0


def fetch_video_metadata(video_id: str) -> Dict[str, Any]:
    """Fetch video metadata using YouTube's official oEmbed endpoint with graceful fallbacks.

    Returns:
        Dict with keys: video_id, title, author_name, author_url, thumbnail_url, youtube_url.
    """
    metadata: Dict[str, Any] = {
        "video_id": video_id,
        "title": f"YouTube Video ({video_id})",
        "author_name": "YouTube Creator",
        "author_url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "fallback_thumbnail_url": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
    }

    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(
            oembed_url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            metadata["title"] = data.get("title", metadata["title"])
            metadata["author_name"] = data.get("author_name", metadata["author_name"])
            metadata["author_url"] = data.get("author_url", metadata["author_url"])
            if "thumbnail_url" in data:
                metadata["thumbnail_url"] = data["thumbnail_url"]
    except Exception:
        # Graceful fallback if network is unreachable or blocked
        pass

    return metadata
