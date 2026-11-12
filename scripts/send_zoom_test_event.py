"""Send a signed Zoom webhook to a running Meeting-to-Action server.

This tests the Zoom path without a Zoom account. It needs only Python's standard
library, and the same secret as the server's ZOOM_WEBHOOK_SECRET_TOKEN.

1. Check the answer to Zoom's URL validation challenge (works against a local or
   a deployed server):

       python scripts/send_zoom_test_event.py --secret <token> --validation

2. Simulate "the transcript of a meeting is ready" against a server running on
   this machine. The script serves a .vtt file locally and sends the event that
   points at it:

       python scripts/send_zoom_test_event.py --secret <token>

   The server only downloads from Zoom by default, so for this test add 127.0.0.1
   to ZOOM_DOWNLOAD_HOSTS in its .env (ZOOM_DOWNLOAD_HOSTS=zoom.us,zoomgov.com,127.0.0.1)
   and restart it.
"""

import argparse
import hashlib
import hmac
import http.server
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_TRANSCRIPT = Path(__file__).resolve().parent.parent / "sample_data" / "sample_transcript.vtt"
DOWNLOAD_TOKEN = "test-download-token"


def hmac_sha256(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def request_json(request: urllib.request.Request) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, {"detail": body.decode(errors="replace")}


def post_event(base_url: str, secret: str, event: dict) -> tuple[int, dict]:
    """Sign and send an event exactly as Zoom does."""
    body = json.dumps(event, separators=(",", ":")).encode()
    timestamp = str(int(time.time()))
    signature = "v0=" + hmac_sha256(secret, b"v0:" + timestamp.encode() + b":" + body)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/webhooks/zoom",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-zm-request-timestamp": timestamp,
            "x-zm-signature": signature,
        },
    )
    return request_json(request)


def check_validation(base_url: str, secret: str) -> bool:
    plain_token = uuid.uuid4().hex
    event = {"event": "endpoint.url_validation", "payload": {"plainToken": plain_token}, "event_ts": int(time.time() * 1000)}
    status, answer = post_event(base_url, secret, event)
    expected = {"plainToken": plain_token, "encryptedToken": hmac_sha256(secret, plain_token.encode())}
    if status == 200 and answer == expected:
        print("OK: the server answered the URL validation challenge correctly.")
        return True
    print(f"FAILED: HTTP {status}, answer: {answer}")
    if status == 401:
        print("The server rejected the signature: --secret differs from its ZOOM_WEBHOOK_SECRET_TOKEN.")
    return False


def serve_transcript(content: bytes) -> tuple[http.server.ThreadingHTTPServer, threading.Event]:
    """Serve the transcript on a free local port, like Zoom's download URL would."""
    downloaded = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.headers.get("Authorization") != f"Bearer {DOWNLOAD_TOKEN}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/vtt")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            downloaded.set()

        def log_message(self, *args) -> None:  # keep the output quiet
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, downloaded


def wait_for_job(base_url: str, job_id: str, api_key: str | None, timeout: float) -> dict | None:
    headers = {"X-API-Key": api_key} if api_key else {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}", headers=headers)
        status, job = request_json(request)
        if status == 401:
            print("The job endpoint needs the server's API key: pass it with --api-key to see the result.")
            return None
        if status != 200:
            print(f"Could not read the job: HTTP {status} {job}")
            return None
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(1)
    print("Timed out waiting for the job to finish.")
    return None


def simulate_transcript_event(base_url: str, secret: str, transcript: Path, api_key: str | None, timeout: float) -> bool:
    server, downloaded = serve_transcript(transcript.read_bytes())
    port = server.server_address[1]
    event = {
        "event": "recording.transcript_completed",
        "event_ts": int(time.time() * 1000),
        "download_token": DOWNLOAD_TOKEN,
        "payload": {
            "account_id": "test-account",
            "object": {
                "id": 81234567890,
                "uuid": uuid.uuid4().hex,  # a new meeting every run, so the server does not see a duplicate
                "topic": "Weekly Product Sync (test event)",
                "start_time": "2026-11-09T09:00:00Z",
                "timezone": "Africa/Tunis",
                "recording_files": [
                    {
                        "file_type": "TRANSCRIPT",
                        "file_extension": "VTT",
                        "recording_type": "audio_transcript",
                        "status": "completed",
                        "download_url": f"http://127.0.0.1:{port}/rec/download/sample.vtt",
                    }
                ],
            },
        },
    }
    try:
        status, answer = post_event(base_url, secret, event)
        print(f"Webhook answer: HTTP {status} {answer}")
        if status != 200 or answer.get("status") != "accepted":
            return False

        job = wait_for_job(base_url, answer["job_id"], api_key, timeout)
        if not downloaded.is_set():
            print("The server never downloaded the transcript from this machine.")
        if job is None:
            return False
        if job["status"] == "failed":
            print(f"Job failed: {job['error']}")
            return False
        result = job["result"]
        print(f"Job completed: {len(result['action_items'])} action item(s), Notion: {result['notion']['status']}.")
        for item in result["action_items"]:
            print(f"  - [{item['priority']}] {item['title']} (owner: {item['owner'] or 'none'}, due: {item['due_date'] or 'none'})")
        return True
    finally:
        server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a signed Zoom test webhook to Meeting-to-Action.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base URL of the server (default: %(default)s)")
    parser.add_argument("--secret", required=True, help="the server's ZOOM_WEBHOOK_SECRET_TOKEN")
    parser.add_argument("--validation", action="store_true", help="only send the URL validation challenge")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT, help="the .vtt file to serve")
    parser.add_argument("--api-key", help="the server's API_KEY, if it has one (used to read the job result)")
    parser.add_argument("--wait", type=float, default=300, help="seconds to wait for the job (default: %(default)s)")
    arguments = parser.parse_args()

    if arguments.validation:
        succeeded = check_validation(arguments.url, arguments.secret)
    else:
        succeeded = simulate_transcript_event(
            arguments.url, arguments.secret, arguments.transcript, arguments.api_key, arguments.wait
        )
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
