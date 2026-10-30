"""Live AML end-to-end smoke: uses a small synthetic image, no private input."""
import argparse
import base64
from io import BytesIO
import json
import os
import time
import uuid
import httpx
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18090")
    args = parser.parse_args()
    token = os.environ.get("MEMORY_API_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    image = Image.new("RGB", (400, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 40, 210, 180), fill="red")
    draw.text((240, 100), "BOX 42", fill="black", font_size=24)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    run = "smoke:" + uuid.uuid4().hex
    content = [{"type": "text", "text": "This is my storage box. I call it Ruby."},
               {"type": "image_url", "image_url": {"url": uri}}]
    payload = {"request_id": run+":add", "user_id": run, "session_id": run+":session",
               "messages": [{"role": "user", "timestamp": 1704067200000, "content": content}]}
    timings = {}
    with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=240) as client:
        start = time.monotonic()
        added = client.post("/add", json=payload)
        timings["add_seconds"] = round(time.monotonic()-start, 3)
        if added.status_code != 200:
            print(json.dumps({"passed": False, "stage": "add", "status": added.status_code,
                              "detail": added.json().get("detail"), "user_id": run}, ensure_ascii=False))
            return 1
        assert added.json() == {"success": True, "request_id": payload["request_id"],
                                "user_id": run, "session_id": payload["session_id"]}
        assert client.post("/add", json=payload).json() == added.json()
        start = time.monotonic()
        result = client.post("/search", json={"user_id": run, "query": "Ruby storage box", "top_k": 100})
        result.raise_for_status()
        timings["text_search_seconds"] = round(time.monotonic()-start, 3)
        rows = result.json()["data"]
        assert rows and len(rows) <= 100
        assert rows[0]["created_at"] == "2024-01-01T00:00:00Z"
        images = [p["image_url"]["url"] for r in rows for p in r["content"] if p["type"] == "image_url"]
        assert uri in images
        visual = client.post("/search", json={"user_id": run, "query": [
            {"type": "text", "text": "Find the memory related to this box."}, content[1]], "top_k": 3})
        visual.raise_for_status()
        assert visual.json()["data"]
        assert client.post("/search", json={"user_id": run+":other", "query": "Ruby", "top_k": 100}).json() == {"data": []}
        print(json.dumps({"passed": True, "user_id": run, "memories": len(rows),
                          "original_image_delivered": True, "image_query": "passed",
                          "isolation": "passed", "idempotency": "passed", **timings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
