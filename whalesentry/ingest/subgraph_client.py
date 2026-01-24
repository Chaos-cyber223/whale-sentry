"""Lightweight GraphQL client for Uniswap subgraphs."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Mapping, Optional

import requests


logger = logging.getLogger(__name__)


class SubgraphClientError(RuntimeError):
    """Base error raised when the subgraph query cannot be fulfilled."""


class SubgraphGraphQLError(SubgraphClientError):
    """Raised when the subgraph returns GraphQL-level errors."""


class SubgraphClient:
    """Thin wrapper around requests with retries and GraphQL error handling."""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("Subgraph endpoint is required")
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._headers = dict(headers or {})
        self._session = requests.Session()

    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not query:
            raise ValueError("query must not be empty")

        payload = {"query": query, "variables": variables or {}}
        attempt = 0
        wait = self.backoff

        while True:
            try:
                request_headers = {"Content-Type": "application/json", **self._headers}
                response = self._session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                    headers=request_headers,
                )
                response.raise_for_status()
            except requests.RequestException as exc:  # pragma: no cover - network dependent
                logger.warning(
                    "Subgraph request failed (%s/%s): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt >= self.max_retries:
                    raise SubgraphClientError("Failed to reach subgraph endpoint") from exc
                time.sleep(wait)
                wait *= 2
                attempt += 1
                continue

            try:
                payload_json = response.json()
            except json.JSONDecodeError as exc:
                raise SubgraphClientError("Subgraph response was not valid JSON") from exc

            if "errors" in payload_json:
                message = payload_json.get("errors")
                logger.warning("Subgraph responded with errors: %s", message)
                if attempt >= self.max_retries:
                    raise SubgraphGraphQLError(str(message))
                time.sleep(wait)
                wait *= 2
                attempt += 1
                continue

            data = payload_json.get("data")
            if data is None:
                raise SubgraphClientError("Subgraph response missing 'data' field")

            logger.debug(
                "Subgraph query succeeded in %s attempt(s)",
                attempt + 1,
            )
            return data
