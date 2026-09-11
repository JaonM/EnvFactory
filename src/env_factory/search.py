"""Client for the local SearXNG search API."""

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SearchError(RuntimeError):
    """Raised when a SearXNG request fails or returns invalid data."""


@dataclass(frozen=True)
class SearchResult:
    """A normalized result returned by SearXNG."""

    title: str
    url: str
    content: str = ""
    engine: str = ""
    score: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SearchResult":
        """Build a result from a SearXNG result object."""

        score = value.get("score")
        return cls(
            title=str(value.get("title", "")),
            url=str(value.get("url") or ""),
            content=str(value.get("content", "")),
            engine=str(value.get("engine", "")),
            score=float(score) if score is not None else None,
        )


@dataclass(frozen=True)
class SearchResponse:
    """A normalized SearXNG search response."""

    query: str
    results: tuple[SearchResult, ...]
    suggestions: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    unresponsive_engines: tuple[tuple[str, str], ...] = ()


class SearXNGClient:
    """Call a SearXNG instance through its JSON search endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("SEARXNG_URL", "http://localhost:8080")).rstrip("/")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        categories: str | None = None,
        language: str | None = None,
        page: int = 1,
        time_range: str | None = None,
        safesearch: int | None = None,
    ) -> SearchResponse:
        """Search the configured SearXNG instance.

        ``categories`` may be a comma-separated value such as ``"general,news"``.
        """

        if not query.strip():
            raise ValueError("query must not be empty")
        if page < 1:
            raise ValueError("page must be greater than or equal to one")
        if safesearch is not None and safesearch not in (0, 1, 2):
            raise ValueError("safesearch must be 0, 1, or 2")

        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "pageno": page,
        }
        optional_params = {
            "categories": categories,
            "language": language,
            "time_range": time_range,
            "safesearch": safesearch,
        }
        params.update({key: value for key, value in optional_params.items() if value is not None})

        request = Request(
            f"{self.base_url}/search?{urlencode(params)}",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise SearchError(f"SearXNG returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise SearchError(f"Unable to reach SearXNG at {self.base_url}") from exc
        except json.JSONDecodeError as exc:
            raise SearchError("SearXNG returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise SearchError("SearXNG response must be a JSON object")

        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise SearchError("SearXNG response field 'results' must be a list")

        results = [
            SearchResult.from_dict(item)
            for item in raw_results
            if isinstance(item, dict)
        ]
        for infobox in payload.get("infoboxes", []):
            if not isinstance(infobox, dict) or not infobox.get("content"):
                continue
            urls = infobox.get("urls", [])
            first_url = ""
            if isinstance(urls, list) and urls and isinstance(urls[0], dict):
                first_url = str(urls[0].get("url") or "")
            results.append(
                SearchResult(
                    title=str(infobox.get("infobox") or infobox.get("id") or ""),
                    url=first_url,
                    content=str(infobox["content"]),
                    engine=str(infobox.get("engine") or "infobox"),
                )
            )

        raw_unresponsive = payload.get("unresponsive_engines", [])
        unresponsive = tuple(
            (str(item[0]), str(item[1]))
            for item in raw_unresponsive
            if isinstance(item, list) and len(item) >= 2
        )
        return SearchResponse(
            query=str(payload.get("query", query)),
            results=tuple(results),
            suggestions=tuple(str(item) for item in payload.get("suggestions", []) if item),
            answers=tuple(str(item) for item in payload.get("answers", []) if item),
            unresponsive_engines=unresponsive,
        )
