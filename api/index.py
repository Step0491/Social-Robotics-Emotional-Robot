"""Vercel Python serverless entrypoint delegating to web_app handler."""

import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from web_app import create_default_agent, make_handler

_AGENT = create_default_agent()
_BaseHandler = make_handler(_AGENT)


class handler(_BaseHandler):
    """Vercel serverless HTTP handler supporting rewritten routes."""

    def _normalize_path(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        route = params.get("__route", [None])[0]
        if route:
            self.path = route
        elif parsed.path:
            self.path = parsed.path

    def do_GET(self) -> None:  # noqa: N802
        self._normalize_path()
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        self._normalize_path()
        super().do_POST()
