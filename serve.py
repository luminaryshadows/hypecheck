"""Read-only loopback preview; no model execution, directory listing or uploads."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote
import argparse

DOCS = Path(__file__).resolve().parent / "docs"
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/archive.html": ("archive.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/report.json": ("report.json", "application/json; charset=utf-8"),
    "/headlines.html": ("headlines.html", "text/html; charset=utf-8"),
    "/headlines.js": ("headlines.js", "text/javascript; charset=utf-8"),
    "/headlines.css": ("headlines.css", "text/css; charset=utf-8"),
    "/headline-report.json": ("headline-report.json", "application/json; charset=utf-8"),
    "/headline-context.json": ("headline-context.json", "application/json; charset=utf-8"),
    "/headline-predictions.json": ("headline-predictions.json", "application/json; charset=utf-8"),
}
CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


class Handler(BaseHTTPRequestHandler):
    server_version = "HypeCheck"
    sys_version = ""

    def end_headers(self):
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        target = FILES.get(unquote(urlsplit(self.path).path))
        if not target:
            self.send_error(404)
            return
        path, mime = target
        file = DOCS / path
        if file.is_symlink() or not file.resolve().is_relative_to(DOCS.resolve()) or not file.is_file():
            self.send_error(404)
            return
        body = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_HEAD = do_GET

    def do_POST(self):
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    print(f"Read-only preview: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
