from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from seqtrainer.clients.synbiohub import (
    SynBioHubClient,
    SynBioHubEndpoints,
    SynBioHubHTTPError,
    SynBioHubResponseError,
)


class DummyResponse:
    def __init__(self, *, status_code: int = 200, json_payload: Any = None, text: str = "", headers: dict[str, str] | None = None, url: str = "https://example.org"):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = url
        self.content = text.encode("utf-8")

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._json_payload is None:
            raise ValueError("no json")
        return self._json_payload


def make_client_with_mocked_request(response: DummyResponse):
    mock_session = Mock()
    mock_session.request = Mock(return_value=response)
    return SynBioHubClient(base_url="https://hub.example.org", session=mock_session), mock_session


def test_run_sparql_basic_json():
    response = DummyResponse(json_payload={"head": {}, "results": {"bindings": []}})
    client, session = make_client_with_mocked_request(response)

    payload = client.run_sparql("SELECT * WHERE {?s ?p ?o}")

    assert payload["results"]["bindings"] == []
    called = session.request.call_args.kwargs
    assert called["method"] == "POST"
    assert called["url"].endswith("/sparql")


def test_run_sparql_pagination_merges_bindings():
    page1 = DummyResponse(json_payload={"head": {"vars": ["s"]}, "results": {"bindings": [{"s": {"value": "a"}}]}})
    page2 = DummyResponse(json_payload={"head": {"vars": ["s"]}, "results": {"bindings": []}})

    mock_session = Mock()
    mock_session.request = Mock(side_effect=[page1, page2])
    client = SynBioHubClient(base_url="https://hub.example.org", session=mock_session)

    payload = client.run_sparql("SELECT ?s WHERE {?s ?p ?o}", paginate=True, page_size=1)

    assert len(payload["results"]["bindings"]) == 1
    assert mock_session.request.call_count == 2


def test_fetch_sbol_text():
    response = DummyResponse(text="<rdf:RDF/>", headers={"Content-Type": "application/rdf+xml"}, json_payload=None)
    client, _ = make_client_with_mocked_request(response)

    data = client.fetch_sbol("https://hub.example.org/public/example")

    assert isinstance(data, str)
    assert "rdf" in data.lower()


def test_http_error_raises_custom_exception():
    bad = DummyResponse(status_code=500, text="server error", json_payload={"error": "x"})
    client, _ = make_client_with_mocked_request(bad)

    with pytest.raises(SynBioHubHTTPError):
        client.run_sparql("SELECT * WHERE {?s ?p ?o}")


def test_non_json_sparql_response_raises_response_error():
    not_json = DummyResponse(text="<html>oops</html>", headers={"Content-Type": "text/html"}, json_payload=None)
    client, _ = make_client_with_mocked_request(not_json)

    with pytest.raises(SynBioHubResponseError):
        client.run_sparql("SELECT * WHERE {?s ?p ?o}")


def test_token_header_is_applied():
    response = DummyResponse(json_payload={"head": {}, "results": {"bindings": []}})
    mock_session = Mock()
    mock_session.request = Mock(return_value=response)
    client = SynBioHubClient(
        base_url="https://hub.example.org",
        endpoints=SynBioHubEndpoints(sparql="/sparql"),
        api_token="token-123",
        session=mock_session,
    )

    client.run_sparql("SELECT * WHERE {?s ?p ?o}")
    headers = mock_session.request.call_args.kwargs["headers"]
    assert headers["X-authorization"] == "token-123"
