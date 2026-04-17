"""Production-oriented SynBioHub client abstractions.

The client keeps a small, explicit API surface while adding:
- configurable endpoints,
- optional auth support,
- retry-aware HTTP transport,
- robust response decoding,
- paginated SPARQL retrieval helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SynBioHubClientError(RuntimeError):
    """Base exception for SynBioHub client failures."""


class SynBioHubHTTPError(SynBioHubClientError):
    """Raised when SynBioHub returns a non-2xx response."""


class SynBioHubResponseError(SynBioHubClientError):
    """Raised when SynBioHub response payload cannot be decoded as expected."""


@dataclass(slots=True)
class SynBioHubEndpoints:
    """Endpoint paths used by :class:`SynBioHubClient`."""

    sparql: str = "/sparql"
    sbol: str = ""


@dataclass(slots=True)
class SynBioHubClient:
    """HTTP client for SynBioHub/SBOLHub-compatible servers."""

    base_url: str
    timeout: int = 30
    endpoints: SynBioHubEndpoints = field(default_factory=SynBioHubEndpoints)
    api_token: str | None = None
    auth: tuple[str, str] | None = None
    user_agent: str = "seqtrainer/0.1"
    verify_ssl: bool = True
    max_retries: int = 3
    backoff_factor: float = 0.5
    session: requests.Session | None = None
    _base_url: str = field(init=False, repr=False)
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._base_url = self.base_url.rstrip("/")
        self._session = self.session or requests.Session()
        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            status=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def base(self) -> str:
        """Normalized base URL for SynBioHub host."""
        return self._base_url

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.api_token:
            headers["X-authorization"] = self.api_token
        if extra:
            headers.update(extra)
        return headers

    def _full_url(self, endpoint_or_url: str) -> str:
        if endpoint_or_url.startswith("http://") or endpoint_or_url.startswith("https://"):
            return endpoint_or_url
        endpoint = endpoint_or_url if endpoint_or_url.startswith("/") else f"/{endpoint_or_url}"
        return f"{self.base}{endpoint}"

    def _request(
        self,
        method: str,
        endpoint_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        response = self._session.request(
            method=method,
            url=self._full_url(endpoint_or_url),
            params=params,
            data=data,
            json=json,
            headers=self._headers(headers),
            timeout=self.timeout,
            verify=self.verify_ssl,
            auth=self.auth,
        )
        if not response.ok:
            body_preview = response.text[:300]
            raise SynBioHubHTTPError(
                f"HTTP {response.status_code} calling {response.url}: {body_preview}"
            )
        return response

    @staticmethod
    def _decode_json(response: requests.Response) -> dict[str, Any] | list[Any]:
        try:
            return response.json()
        except ValueError as exc:
            raise SynBioHubResponseError(
                f"Expected JSON response from {response.url}, got: {response.text[:200]}"
            ) from exc

    @staticmethod
    def _add_limit_offset(query: str, limit: int, offset: int) -> str:
        upper = query.upper()
        if "LIMIT" in upper or "OFFSET" in upper:
            return query
        stripped = query.rstrip().rstrip(";")
        return f"{stripped}\nLIMIT {limit}\nOFFSET {offset}"

    def run_sparql(
        self,
        query: str,
        *,
        paginate: bool = False,
        page_size: int = 500,
        max_pages: int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Run SPARQL query and decode response.

        If ``paginate=True``, this method adds ``LIMIT/OFFSET`` automatically
        when absent and merges `results.bindings` pages.
        """
        if not paginate:
            response = self._request(
                "POST",
                self.endpoints.sparql,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
            )
            return self._decode_json(response)

        merged: dict[str, Any] | None = None
        page = 0
        offset = 0

        while True:
            if max_pages is not None and page >= max_pages:
                break
            paged_query = self._add_limit_offset(query, limit=page_size, offset=offset)
            payload = self.run_sparql(paged_query, paginate=False)
            if not isinstance(payload, dict):
                raise SynBioHubResponseError("Paginated SPARQL expected dict payload")

            bindings = payload.get("results", {}).get("bindings", [])
            if merged is None:
                merged = payload
            else:
                merged.setdefault("results", {}).setdefault("bindings", []).extend(bindings)

            if len(bindings) < page_size:
                break
            offset += page_size
            page += 1

        return merged or {"head": {"vars": []}, "results": {"bindings": []}}

    def fetch_sbol(
        self,
        uri: str,
        *,
        endpoint: str | None = None,
        accept: str = "application/rdf+xml,text/plain;q=0.9,*/*;q=0.1",
    ) -> str | bytes:
        """Fetch SBOL text/bytes from an absolute URI or configured endpoint."""
        target = uri
        if endpoint:
            endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
            target = f"{endpoint_path}/{uri.lstrip('/')}"
        elif self.endpoints.sbol and not uri.startswith(("http://", "https://")):
            target = f"{self.endpoints.sbol.rstrip('/')}/{uri.lstrip('/')}"

        response = self._request("GET", target, headers={"Accept": accept})
        ctype = response.headers.get("Content-Type", "").lower()
        if any(marker in ctype for marker in ("text", "xml", "json", "rdf")):
            return response.text
        return response.content

    def close(self) -> None:
        """Close underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "SynBioHubClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
