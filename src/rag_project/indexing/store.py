"""Vector store client.

The deployment at LANCEDB_URI is a custom REST wrapper around LanceDB, not
LanceDB Cloud/Enterprise, so the `lancedb` Python client cannot talk to it.
This is a thin client for that wrapper's actual surface.

Auth is split on that service and the split is not obvious:
  * data endpoints  -> `x-api-key` header
  * /openapi.json   -> `?key=` query parameter
Only the former matters here.

`VectorStore` is a Protocol so tests can run against InMemoryStore without a
network, and so a future move to real LanceDB Enterprise is a swap of one class.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import httpx

from ..config import get_settings

# A 3072-dim vector serialises to roughly 60KB of JSON, so the binding
# constraint is nginx's client_max_body_size (1MB by default), not anything
# about the database. Batch size is derived from the vector width rather than
# hard-coded, and halves on a 413 so a smaller proxy limit self-corrects.
MAX_BODY_BYTES = 900_000
BYTES_PER_DIM = 20
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def batch_size_for(dim: int) -> int:
    return max(1, MAX_BODY_BYTES // (dim * BYTES_PER_DIM + 2_000))


class StoreError(RuntimeError):
    pass


@runtime_checkable
class VectorStore(Protocol):
    def exists(self) -> bool: ...
    def create(self, records: list[dict[str, Any]]) -> None: ...
    def insert(self, records: list[dict[str, Any]]) -> int: ...
    def search(
        self, vector: list[float], limit: int = 10, filter: str | None = None
    ) -> list[dict[str, Any]]: ...
    def count(self) -> int: ...
    def drop(self) -> None: ...


class RemoteStore:
    """Client for the self-hosted LanceDB REST wrapper."""

    def __init__(self, uri: str, api_key: str, table: str) -> None:
        self.base = uri.rstrip("/")
        self.table = table
        self._client = httpx.Client(
            headers={"x-api-key": api_key},
            timeout=TIMEOUT,
            follow_redirects=True,
        )

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self._client.request(method, f"{self.base}{path}", **kw)
        except httpx.HTTPError as e:
            raise StoreError(f"{method} {path} failed to reach the server: {e}") from e
        if r.status_code == 401:
            raise StoreError(
                "LanceDB rejected the API key. Check LANCEDB_API_KEY in .env — "
                "data endpoints use the `x-api-key` header."
            )
        if r.status_code == 404 and "/tables/" in path:
            raise StoreError(f"Table {self.table!r} not found on {self.base}.")
        if r.status_code >= 400:
            raise StoreError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    # --- table lifecycle ------------------------------------------------
    def list_tables(self) -> list[str]:
        return self._request("GET", "/tables").get("tables", [])

    def exists(self) -> bool:
        return self.table in self.list_tables()

    def create(self, records: list[dict[str, Any]]) -> None:
        """Create the table. The wrapper infers schema from these records, so
        the first batch must contain every column the table will ever have."""
        self._request(
            "POST", f"/tables/{self.table}", json={"records": records, "mode": "create"}
        )

    def drop(self) -> None:
        self._request("DELETE", f"/tables/{self.table}")

    # --- data -----------------------------------------------------------
    def insert(self, records: list[dict[str, Any]], progress: bool = False) -> int:
        if not records:
            return 0
        dim = len(records[0].get("vector") or [])
        size = batch_size_for(dim) if dim else 32

        n = 0
        i = 0
        while i < len(records):
            batch = records[i : i + size]
            try:
                self._request(
                    "POST", f"/tables/{self.table}/insert", json={"records": batch}
                )
            except StoreError as e:
                if "413" in str(e) and size > 1:
                    size = max(1, size // 2)
                    continue  # retry the same slice, smaller
                raise
            i += len(batch)
            n += len(batch)
            if progress and (n % (size * 20) == 0 or i >= len(records)):
                print(f"  inserted {n}/{len(records)} rows", flush=True)
        return n

    def search(
        self, vector: list[float], limit: int = 10, filter: str | None = None
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"vector": vector, "limit": limit}
        if filter:
            payload["filter"] = filter
        res = self._request("POST", f"/tables/{self.table}/search", json=payload)
        # The wrapper returns either a bare list or {"results": [...]}.
        if isinstance(res, dict):
            res = res.get("results") or res.get("records") or []
        return list(res)

    def count(self) -> int:
        res = self._request("GET", f"/tables/{self.table}/count")
        return res.get("count", res) if isinstance(res, dict) else int(res)

    def schema(self) -> Any:
        return self._request("GET", f"/tables/{self.table}/schema")


class InMemoryStore:
    """Offline stand-in with the same contract. Brute-force cosine."""

    def __init__(self, table: str = "memory") -> None:
        self.table = table
        self._rows: list[dict[str, Any]] = []
        self._created = False

    def exists(self) -> bool:
        return self._created

    def create(self, records: list[dict[str, Any]]) -> None:
        self._created = True
        self._rows.extend(records)

    def insert(self, records: list[dict[str, Any]]) -> int:
        self._rows.extend(records)
        return len(records)

    def search(
        self, vector: list[float], limit: int = 10, filter: str | None = None
    ) -> list[dict[str, Any]]:
        import math

        def cos(a: list[float], b: list[float]) -> float:
            num = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(y * y for y in b)) or 1e-9
            return num / (na * nb)

        scored = [
            {**{k: v for k, v in r.items() if k != "vector"},
             "_distance": 1.0 - cos(vector, r["vector"])}
            for r in self._rows
        ]
        scored.sort(key=lambda r: r["_distance"])
        return scored[:limit]

    def count(self) -> int:
        return len(self._rows)

    def drop(self) -> None:
        self._rows.clear()
        self._created = False


def get_store() -> VectorStore:
    s = get_settings()
    if not s.is_remote:
        return InMemoryStore(s.table)
    if not s.lancedb_api_key:
        raise StoreError("LANCEDB_URI is remote but LANCEDB_API_KEY is not set in .env")
    return RemoteStore(s.lancedb_uri, s.lancedb_api_key, s.table)
