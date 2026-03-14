"""YouTube upload via the YouTube Data API v3.

Handles OAuth2 authentication (one-time browser consent, then refresh-token
based), resumable video uploads, and thumbnail setting.

Requires: google-api-python-client, google-auth-oauthlib
"""

from __future__ import annotations

import http.client
import json
import random
import sys
import time
from pathlib import Path

from .models import EditorError

# Lazy-imported so the rest of the package doesn't break when google libs
# aren't installed (they're only needed for the upload command).
_GOOGLE_LIBS_AVAILABLE: bool | None = None

# Scopes required: upload videos + set thumbnails.
_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Resumable-upload retry config.
_MAX_RETRIES = 10
_RETRIABLE_STATUS_CODES = (500, 502, 503, 504)

# Default token / secrets paths.
DEFAULT_CLIENT_SECRETS = Path("client_secrets.json")
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "lol-video-editor" / "youtube-token.json"


def _ensure_google_libs() -> None:
    """Import google client libs or raise a clear error."""
    global _GOOGLE_LIBS_AVAILABLE
    if _GOOGLE_LIBS_AVAILABLE is True:
        return
    try:
        import google.auth.transport.requests  # noqa: F401
        import google_auth_oauthlib.flow  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
        import googleapiclient.http  # noqa: F401
    except ImportError:
        _GOOGLE_LIBS_AVAILABLE = False
        raise EditorError(
            "YouTube upload requires google-api-python-client and "
            "google-auth-oauthlib.\n"
            "Install them with:\n"
            "  pip install google-api-python-client google-auth-oauthlib"
        )
    _GOOGLE_LIBS_AVAILABLE = True


def _get_authenticated_service(
    client_secrets: Path,
    token_path: Path,
):
    """Build an authenticated YouTube API service.

    On first run, opens a browser for OAuth consent and saves the refresh
    token.  Subsequent runs reuse the saved token silently.
    """
    _ensure_google_libs()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if creds is None or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secrets.exists():
                raise EditorError(
                    f"OAuth client secrets file not found: {client_secrets}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials.\n"
                    "See: https://console.cloud.google.com/apis/credentials"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets), _SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Persist the token for future runs.
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        print(f"Saved YouTube auth token: {token_path}", file=sys.stderr)

    return build("youtube", "v3", credentials=creds)


def _resumable_upload(request) -> dict:
    """Execute a resumable upload with exponential back-off retries."""
    response = None
    retries = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"  Upload {pct}% complete.", file=sys.stderr, flush=True)
        except http.client.HTTPException:
            if retries >= _MAX_RETRIES:
                raise
            retries += 1
            wait = random.uniform(0, 2**retries)
            print(
                f"  Upload interrupted, retrying in {wait:.1f}s (attempt {retries}/{_MAX_RETRIES})...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    return response


def upload_to_youtube(
    video_path: Path,
    *,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    category_id: str = "20",  # Gaming
    privacy: str = "private",
    thumbnail_path: Path | None = None,
    client_secrets: Path = DEFAULT_CLIENT_SECRETS,
    token_path: Path = DEFAULT_TOKEN_PATH,
) -> dict:
    """Upload a video to YouTube and optionally set a custom thumbnail.

    Returns the YouTube API ``videos.insert`` response dict (contains the
    video ``id``, ``snippet``, etc.).
    """
    _ensure_google_libs()
    from googleapiclient.http import MediaFileUpload

    if not video_path.exists():
        raise EditorError(f"Video file not found: {video_path}")

    youtube = _get_authenticated_service(client_secrets, token_path)

    body: dict = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    print(f"Uploading {video_path.name} to YouTube...", file=sys.stderr, flush=True)
    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=50 * 1024 * 1024,  # 50 MB chunks
    )
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )
    response = _resumable_upload(request)
    video_id = response["id"]
    print(
        f"Upload complete: https://www.youtube.com/watch?v={video_id}",
        file=sys.stderr,
        flush=True,
    )

    # Set custom thumbnail if provided.
    if thumbnail_path is not None:
        if not thumbnail_path.exists():
            print(
                f"Warning: thumbnail not found at {thumbnail_path}, skipping.",
                file=sys.stderr,
            )
        else:
            print("Setting custom thumbnail...", file=sys.stderr, flush=True)
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path)),
            ).execute()
            print("Thumbnail set.", file=sys.stderr, flush=True)

    return response
