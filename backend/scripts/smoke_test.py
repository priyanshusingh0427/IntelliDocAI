r"""Check a running IntelliDoc AI deployment from the outside, like a user would.

    ..\.venv\Scripts\python.exe scripts\smoke_test.py http://localhost:8080
    ..\.venv\Scripts\python.exe scripts\smoke_test.py http://localhost:8080 --allow-llm-down

Creates a throwaway account (smoke-<random>@example.com), uploads the X200 test manual,
asks a question and checks the cited page, then checks the web server setup (security
headers, single-page-app routing, API docs, upload size limit). Exit code 0 = all passed.
"""
import argparse
import secrets
import sys
import time
from pathlib import Path

import httpx

FIXTURE = Path(__file__).resolve().parents[2] / "frontend" / "e2e" / "fixtures" / "X200_Manual.pdf"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> bool:
    results.append((ok, label))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return ok


def wait_until_ready(client: httpx.Client, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            response = client.get("/api/v1/health/ready")
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last = exc.__class__.__name__
        time.sleep(3)
    raise SystemExit(f"Backend not reachable after {timeout}s ({last})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://localhost:8080")
    parser.add_argument("--allow-llm-down", action="store_true", help="do not fail if the LLM is offline")
    parser.add_argument("--wait", type=int, default=180, help="seconds to wait for the backend")
    parser.add_argument("--api-only", action="store_true", help="skip nginx checks (talking to the backend directly)")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=300)
    print(f"Smoke test against {args.base_url}")

    print("readiness")
    ready = wait_until_ready(client, args.wait)
    for name, component in ready["components"].items():
        required = not (name == "llm" and args.allow_llm_down)
        ok = component["ok"] or not required
        check(ok, f"{name}: {component['detail']}")

    if not args.api_only:
        check_web_server(client)
    else:
        check(client.get("/api/v1/documents").status_code == 401, "API requires login")

    user_flow(client, args)
    return summary()


def check_web_server(client: httpx.Client) -> None:
    print("web server")
    home = client.get("/")
    check(home.status_code == 200 and '<div id="root">' in home.text, "frontend served")
    check(home.headers.get("x-content-type-options") == "nosniff", "security headers present")
    check("default-src 'self'" in home.headers.get("content-security-policy", ""), "content security policy set")
    deep = client.get("/some/client/route")
    check(deep.status_code == 200 and '<div id="root">' in deep.text, "single-page-app routes fall back to index.html")
    check(client.get("/docs").status_code == 200, "API docs reachable at /docs")
    check(client.get("/api/v1/documents").status_code == 401, "API requires login")


def user_flow(client: httpx.Client, args: argparse.Namespace) -> None:
    print("user flow")
    email = f"smoke-{secrets.token_hex(4)}@example.com"
    password = secrets.token_urlsafe(12)
    reg = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    if not check(reg.status_code == 201, f"register {email}"):
        return
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    with FIXTURE.open("rb") as f:
        up = client.post("/api/v1/documents", files={"file": (FIXTURE.name, f, "application/pdf")}, headers=headers)
    doc = up.json() if up.status_code == 201 else {}
    check(doc.get("page_count") == 3 and doc.get("index_status") == "indexed", "upload, extract 3 pages and index")

    started = time.monotonic()
    ans = client.post("/api/v1/ask", json={"question": "How do I reset the X200?"}, headers=headers)
    body = ans.json() if ans.status_code == 200 else {}
    pages = [c["page_number"] for c in body.get("citations", [])]
    check(body.get("found") is True and 2 in pages, f"answer cites page 2 ({time.monotonic() - started:.1f}s, {body.get('llm_model')})")
    if body:
        print(f"        answer: {body['answer'][:120]!r}")
    off = client.post("/api/v1/ask", json={"question": "What is the capital of France?"}, headers=headers).json()
    check(off.get("found") is False, "off-topic question answered 'not found'")
    check(client.get("/api/v1/chats", headers=headers).json()[0]["message_count"] == 2, "question saved to chat history")

    # nginx rejects it before the backend; with --api-only the backend's own limit answers 413
    big = client.post(
        "/api/v1/documents",
        files={"file": ("big.pdf", b"%PDF" + b"0" * (12 * 1024 * 1024), "application/pdf")},
        headers=headers,
    )
    check(big.status_code == 413, f"12 MB upload rejected (HTTP {big.status_code})")

    if doc.get("id"):
        check(client.delete(f"/api/v1/documents/{doc['id']}", headers=headers).status_code == 204, "clean up test document")


def summary() -> int:
    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
