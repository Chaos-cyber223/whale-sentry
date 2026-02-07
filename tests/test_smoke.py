import pytest

from whalesentry.ingest.subgraph_client import SubgraphClient


def test_client_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        SubgraphClient("")


def test_client_applies_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class _DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict:
            return {"data": {}}

    class _DummySession:
        def post(self, *_args, **kwargs):  # type: ignore[no-untyped-def]
            captured["headers"] = kwargs.get("headers")
            return _DummyResponse()

    monkeypatch.setattr(
        "whalesentry.ingest.subgraph_client.requests.Session",
        lambda: _DummySession(),
    )

    client = SubgraphClient("https://example.com", headers={"Authorization": "Bearer x"})
    client.query("query { test }")

    assert captured["headers"]["Authorization"] == "Bearer x"
